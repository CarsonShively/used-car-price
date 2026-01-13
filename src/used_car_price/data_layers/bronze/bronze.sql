INSTALL httpfs;
LOAD httpfs;

CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE VIEW bronze.used_cars AS
SELECT *
FROM read_parquet(
  'https://huggingface.co/datasets/Carson-Shively/used-car-price/resolve/main/data/bronze/bronze.parquet'
);