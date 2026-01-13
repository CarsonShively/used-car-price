CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE VIEW silver.used_cars AS
WITH typed AS (
  SELECT
    * REPLACE (
      lower(trim(regexp_replace(brand,        '\s+', ' '))) AS brand,
      lower(trim(regexp_replace(model,        '\s+', ' '))) AS model,
      lower(trim(regexp_replace(fuel_type,    '\s+', ' '))) AS fuel_type,
      lower(trim(regexp_replace(engine,       '\s+', ' '))) AS engine,
      lower(trim(regexp_replace(transmission, '\s+', ' '))) AS transmission,
      lower(trim(regexp_replace(ext_col,      '\s+', ' '))) AS ext_col,
      lower(trim(regexp_replace(int_col,      '\s+', ' '))) AS int_col,
      lower(trim(regexp_replace(accident,     '\s+', ' '))) AS accident,
      lower(trim(regexp_replace(clean_title,  '\s+', ' '))) AS clean_title,
      TRY_CAST(model_year AS DOUBLE) AS model_year,
      TRY_CAST(NULLIF(REPLACE(REGEXP_REPLACE(LOWER(TRIM(CAST(milage AS VARCHAR))), '\s*mi\b', ''), ',', ''), '') AS DOUBLE) AS milage,
      TRY_CAST(NULLIF(REPLACE(REPLACE(CAST(price AS VARCHAR), '$', ''), ',', ''), '') AS DOUBLE) AS price
    )
  FROM bronze.used_cars
),
validated AS (
  SELECT
    * REPLACE (
      CASE WHEN price BETWEEN 2000 AND 250000 THEN price ELSE NULL END AS price,
      CASE WHEN model_year BETWEEN 1990 AND 2025 THEN model_year ELSE NULL END AS model_year,
      CASE WHEN milage BETWEEN 0 AND 400000 THEN milage ELSE NULL END AS milage,
      CASE
        WHEN fuel_type IN ('gasoline','hybrid','diesel','e85 flex fuel','plug-in hybrid')
          THEN fuel_type
        ELSE NULL
      END AS fuel_type
    )
  FROM typed
),
deduped AS (
  SELECT *
  FROM validated
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY
      brand, model, model_year, milage, fuel_type,
      engine, transmission, ext_col, int_col, accident, clean_title
    ORDER BY price DESC NULLS LAST
  ) = 1
)
SELECT *
FROM deduped
WHERE price IS NOT NULL;
