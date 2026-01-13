import pathlib, duckdb
from huggingface_hub import upload_file
from importlib.resources import files

DATASET_REPO_ID = "Carson-Shively/used-car-price"
REVISION        = "main"

EXPORT_REL      = "silver.used_cars"             
LOCAL_FILE      = pathlib.Path("data/silver/silver.parquet")
PATH_IN_REPO    = "data/silver/silver.parquet"

SQL_BRONZE_PKG = "used_car_price.data_layers.bronze"
SQL_SILVER_PKG = "used_car_price.data_layers.silver"

SQL_BRONZE_FILE = "bronze.sql"
SQL_SILVER_FILE = "silver.sql"

def _read_pkg_sql(pkg: str, filename: str) -> str:
    return (files(pkg) / filename).read_text(encoding="utf-8")

def _exec_sql_text(con, sql: str):
    con.execute(sql)

def build_and_write_parquet():
    LOCAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        _exec_sql_text(con, _read_pkg_sql(SQL_BRONZE_PKG, SQL_BRONZE_FILE))
        _exec_sql_text(con, _read_pkg_sql(SQL_SILVER_PKG, SQL_SILVER_FILE))

        con.execute(
            f"COPY (SELECT * FROM {EXPORT_REL}) TO '{LOCAL_FILE.as_posix()}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD);"
        )
    finally:
        con.close()

def upload_parquet():
    info = upload_file(
        path_or_fileobj=str(LOCAL_FILE),
        path_in_repo=PATH_IN_REPO,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        commit_message=f"Upload {PATH_IN_REPO}",
    )
    print("Uploaded", PATH_IN_REPO, getattr(info, "commit_id", ""))

if __name__ == "__main__":
    build_and_write_parquet()
    upload_parquet()