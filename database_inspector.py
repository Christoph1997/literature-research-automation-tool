"""
Extracts the necessary information from the dataset and saves it in a new file into
parquet format.

Author: Christoph Ruff
"""

import duckdb


class JsonDatabaseInspector:
    """
    Utility class for inspecting the schema of a JSON file
    using DuckDB.
    """

    def __init__(self, json_file: str):
        """
        Initialize the inspector.

        Parameters
        ----------
        json_file : str
            Path to the JSON file that should be analyzed.
        """
        self.json_file = json_file

        # Create an in-memory DuckDB connection
        self.con = duckdb.connect()

    def get_schema(self):
        """
        Read the JSON file and return its schema.

        Returns
        -------
        list
            A list containing column metadata returned by DuckDB.
        """

        query = f"""
        DESCRIBE
        SELECT *
        FROM read_json_auto('{self.json_file}')
        """

        return self.con.execute(query).fetchdf()

    def get_column_names(self):
        """
        Return only the column names.

        Returns
        -------
        list[str]
            List of column names.
        """

        schema = self.get_schema()

        return [column_name for column_name, *_ in schema]

    def get_first_n_rows(self, n: int = 5):
        """
        Return the first n rows of the JSON file.

        Parameters
        ----------
        n : int, optional
            Number of rows to return, by default 5

        Returns
        -------
        list[tuple]
            List of tuples representing the first n rows.
        """

        query = f"""
        SELECT *
        FROM read_json_auto('{self.json_file}')
        LIMIT {n}
        """

        return self.con.execute(query).fetchdf()

    def get_columns_by_name(self, column_names: list[str]):
        """
        Return the specified columns from the JSON file.

        Parameters
        ----------
        column_names : list[str]
            List of column names to retrieve.

        Returns
        -------
        list[tuple]
            List of tuples representing the specified columns.
        """

        columns_str = ", ".join(column_names)

        query = f"""
        SELECT {columns_str}
        FROM read_json_auto('{self.json_file}')
        """

        return self.con.execute(query).fetchdf()

    def export_to_parquet(self, output_file: str, column_names: list[str] = None):
        """
        Export the specified columns to a Parquet file.

        Parameters
        ----------
        output_file : str
            Path to the output Parquet file.
        column_names : list[str], optional
            List of column names to export. If None, all columns are exported.
        """

        if column_names is None:
            print("Exporting all columns to Parquet.")
            query = f"""
            COPY (SELECT * FROM read_json_auto('{self.json_file}'))
            TO '{output_file}' (FORMAT PARQUET)
            """
        else:
            print(f"Exporting specified columns to parquet: {column_names}")
            columns_str = ", ".join(column_names)
            query = f"""
            COPY (SELECT {columns_str} FROM read_json_auto('{self.json_file}'))
            TO '{output_file}' (FORMAT PARQUET)
            """

        self.con.execute(query)

    def close(self):
        """
        Close the DuckDB connection.
        """

        self.con.close()
