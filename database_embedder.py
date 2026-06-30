"""
Embed a Parquet dataset using microsoft/harrier-oss-v1-0.6b.

This script:
1. Reads a Parquet file.
2. Joins selected text columns into one text per row.
3. Creates embeddings with SentenceTransformer.
4. Saves embeddings as .npy.
5. Saves metadata as .meta.json.

Authors: Christoph Ruff, ChatGPT
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from usearch.index import Index


class DatabaseEmbedder:
    """Embed a Parquet dataset with a certain model."""

    def __init__(
        self,
        parquet_file: str,
        output_prefix: str = "data/model",
        id_column: str = "id",
        text_columns: list[str] | None = None,
        model_name: str = "microsoft/harrier-oss-v1-0.6b",
        chunk_size: int = 10000,
        batch_size: int = 8,
        test_mode: bool = False,
    ):
        self.parquet_file = Path(parquet_file)
        self.id_column = id_column

        # Use title and abstract as default text input because these usually
        # contain the most useful semantic information for literature search.
        self.text_columns = text_columns or ["title", "abstract"]
        self.model_name = model_name
        self.batch_size = batch_size
        # chunk_size controls how many rows are read from Parquet at once.
        # The same chunk_size is also used later during similarity search.
        self.chunk_size = chunk_size

        # If test_mode is True, only the first batch is embedded.
        # This is useful to check whether the model, files, and dependencies work
        # before embedding the whole dataset.
        self.test_mode = test_mode

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU.")

        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={"dtype": "auto"},
            device=self.device,
        )

        output_prefix_path = Path(output_prefix)

        # Create parent folders if they do not exist
        output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = output_prefix_path.with_name(
            f"{output_prefix_path.stem}_embeddings.npy"
        )
        self.metadata_path = output_prefix_path.with_name(
            f"{output_prefix_path.stem}_metadata.meta.json"
        )

        # Path where the USearch vector index is saved.
        # This is not the same as a full database; it stores the searchable vector index
        self.vector_index_path = output_prefix_path.with_name(
            f"{output_prefix_path.stem}_usearch.index"
        )

    def _count_rows(self) -> int:
        """Count rows in the Parquet file."""

        query = f"""
            SELECT COUNT(*)
            FROM read_parquet('{self.parquet_file.as_posix()}')
        """

        return duckdb.sql(query).fetchone()[0]

    def _read_rows(self, limit: int | None = None, offset: int = 0) -> list[tuple]:
        """Read selected rows from the Parquet file using DuckDB."""

        selected_columns = [self.id_column, *self.text_columns]
        columns_sql = ", ".join(f'"{column}"' for column in selected_columns)
        limit_sql = ""

        if limit is not None:
            limit_sql = f"LIMIT {limit}"

        query = f"""
            SELECT {columns_sql}
            FROM read_parquet('{self.parquet_file.as_posix()}')
            {limit_sql}
            OFFSET {offset}
        """

        return duckdb.sql(query).fetchall()

    def _build_ids_and_texts(self, rows: list[tuple]) -> tuple[list[str], list[str]]:
        """Transform raw Parquet rows into IDs and combined text values."""

        selected_columns = [self.id_column, *self.text_columns]
        ids = []
        texts = []

        for row in rows:
            row_values = dict(zip(selected_columns, row, strict=True))

            ids.append(str(row_values[self.id_column]))

            parts = []
            for column in self.text_columns:
                value = row_values.get(column)

                if value is not None:
                    # Remove excessive whitespace and newlines from the text
                    value = " ".join(str(value).split())

                    if value:
                        parts.append(value)

            texts.append(". ".join(parts))

        return ids, texts

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Create normalized embeddings for a list of texts."""

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        print(f"Embedding matrix shape: {embeddings.shape}")

        return np.asarray(embeddings, dtype=np.float32)

    def _save_vector_index(self, embeddings: np.ndarray) -> None:
        """
        Save embeddings into a USearch vector index.

        USearch is used here as a fast vector search index.
        It stores the embedding vectors and allows fast nearest-neighbor search.

        Important:
        - The index key is the row position: 0, 1, 2, ...
        - These keys must stay aligned with metadata["ids"] and metadata["texts"].
        - Example: key 25 means metadata["ids"][25] and metadata["texts"][25].
        """

        print("Building USearch vector index...")

        embedding_dimension = embeddings.shape[1]

        index = Index(
            ndim=embedding_dimension,
            metric="cos",
            dtype="f32",
        )

        keys = np.arange(embeddings.shape[0], dtype=np.uint64)

        index.add(keys, embeddings)

        index.save(str(self.vector_index_path))

        print(f"Saved USearch index to: {self.vector_index_path}\n")

    def _save_embeddings(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
    ) -> None:
        """Save embeddings and metadata."""

        np.save(self.embeddings_path, embeddings)

        metadata = {
            "ids": ids,
            "texts": texts,
            "model": self.model_name,
            "text_columns": self.text_columns,
        }

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)

        print(f"Saved embeddings to: {self.embeddings_path}")
        print(f"Saved metadata to: {self.metadata_path}\n")

    def run(self) -> None:
        """Run the full embedding pipeline."""

        all_ids = []
        all_texts = []
        embedding_chunks = []

        total_rows = self._count_rows()

        if self.test_mode:
            total_rows = min(total_rows, self.batch_size)

        print("Reading Parquet file...")
        limit = self.batch_size if self.test_mode else self.chunk_size
        for offset in range(0, total_rows, self.chunk_size):
            rows = self._read_rows(limit=limit, offset=offset)

            ids, texts = self._build_ids_and_texts(rows)

            print(f"Embedding {len(texts)} texts with {self.model_name}...")
            embeddings = self._embed_texts(texts)

            all_ids.extend(ids)
            all_texts.extend(texts)
            embedding_chunks.append(embeddings)

        combined_embeddings = np.vstack(embedding_chunks)

        self._save_embeddings(
            ids=all_ids,
            texts=all_texts,
            embeddings=combined_embeddings,
        )

        # Save combined embeddings into a USearch index for faster similarity search.
        self._save_vector_index(combined_embeddings)

    def _save_matching_results(
        self,
        results: list[dict],
        output_parquet_path: Path,
        output_csv_path: Path,
    ) -> None:
        """Save full matching rows to Parquet and CSV using DuckDB."""

        con = duckdb.connect()

        try:
            con.execute(
                """
                CREATE TEMP TABLE search_results (
                    id VARCHAR,
                    similarity_score DOUBLE
                )
                """
            )

            con.executemany(
                """
                INSERT INTO search_results VALUES (?, ?)
                """,
                [(result["id"], result["similarity_score"]) for result in results],
            )

            # Add after p.* to remove \n and \r characters from title and abstract
            # columns
            # REPLACE(REPLACE(CAST(title AS VARCHAR),
            # chr(13), ' '), chr(10), ' ') AS title,
            # REPLACE(REPLACE(CAST(abstract AS VARCHAR),
            # chr(13), ' '), chr(10), ' ') AS abstract,
            query = f"""
                SELECT
                    p.*,
                    s.similarity_score
                FROM read_parquet('{self.parquet_file.as_posix()}') AS p
                JOIN search_results AS s
                    ON CAST(p.{self.id_column} AS VARCHAR) = s.id
                ORDER BY s.similarity_score DESC
            """

            con.execute(
                f"COPY ({query}) TO '{output_parquet_path.as_posix()}' (FORMAT PARQUET)"
            )
            con.execute(
                f"""
                COPY (
                    {query}
                )
                TO '{output_csv_path.as_posix()}'
                (
                    FORMAT CSV,
                    HEADER TRUE,
                    DELIMITER ';'
                )
                """
            )
        finally:
            con.close()

        print(f"Saved full matching rows to {output_parquet_path}")
        print(f"Saved full matching rows to {output_csv_path}\n")

    def search_embeddings(
        self,
        query: str,
        output_file: str,
        top_k: int | None = 5,
        similarity_threshold: float | None = None,
    ) -> None:
        """
        Search saved embeddings with a text query using cosine similarity and saves
        the results in a parquet and csv file with the matching parquet rows.
        """
        print(f"Start searching for similar results for query:\n {query}\n")
        rows = self._read_rows()
        ids, _ = self._build_ids_and_texts(rows)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0].astype(np.float32)

        index = Index(
            ndim=len(query_embedding),
            metric="cos",
            dtype="f32",
        )
        index.load(str(self.vector_index_path))

        # USearch needs a fixed number of candidates.
        # If a threshold is used, retrieve more candidates first and filter later.
        search_k = top_k if top_k is not None else len(ids)

        matches = index.search(query_embedding, search_k)

        results = []

        for match in matches:
            index_position = int(match.key)

            # With cosine distance: smaller distance = more similar.
            similarity_score = float(1 - match.distance)

            if (
                similarity_threshold is not None
                and similarity_score < similarity_threshold
            ):
                continue

            results.append(
                {
                    "similarity_score": similarity_score,
                    "id": ids[index_position],
                }
            )

        results = sorted(
            results,
            key=lambda result: result["similarity_score"],
            reverse=True,
        )

        if top_k is not None:
            results = results[:top_k]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_csv_path = output_path.with_suffix(".csv")
        output_parquet_path = output_path.with_suffix(".parquet")
        self._save_matching_results(results, output_parquet_path, output_csv_path)

        print("Finished searching for similar results.\n")
