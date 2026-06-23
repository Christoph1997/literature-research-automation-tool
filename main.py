"""
Runs the file

Author: Christoph Ruff
"""

from database_inspector import JsonDatabaseInspector


def main():
    # =========================
    # User Configuration
    # =========================
    json_file = "../Arxiv_Dataset/arxiv-metadata-oai-snapshot.json"
    column_names = ["id", "title", "doi", "abstract", "authors_parsed"]
    parquet_file = "../Arxiv_Dataset/arxiv_metadata.parquet"

    print("Start Literature Research Automation Tool!")

    # Create an inspector object for the JSON file
    inspector = JsonDatabaseInspector(json_file)
    inspector.export_to_parquet(output_file=parquet_file, column_names=column_names)

    # columns_name = inspector.get_column_names()
    # columns = inspector.get_columns_by_name(column_names)
    # print("\nColumn names only:")
    # print(columns_name)
    # result = inspector.get_first_n_rows(n=5)
    # print(f"\nFirst {1} rows:\n")
    # print(result)

    # Always close the database connection when finished
    inspector.close()


if __name__ == "__main__":
    main()
