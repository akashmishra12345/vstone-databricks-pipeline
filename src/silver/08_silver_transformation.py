# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Silver DLT Pipeline | Streaming Bronze → Silver
# MAGIC
# MAGIC All Silver and Quarantine tables are **Streaming Tables** (`@dlt.table` with `spark.readStream`).
# MAGIC
# MAGIC ### Why streaming tables:
# MAGIC - `dlt.read_stream()` reads from Bronze DLT tables as a stream — only new rows processed each run
# MAGIC - Both Silver and Quarantine use `foreachBatch` pattern via DLT's append mode
# MAGIC - Idempotency guaranteed by DLT checkpoint + MERGE on primary key
# MAGIC
# MAGIC ### Deliverables:
# MAGIC | # | Deliverable | Implemented |
# MAGIC |---|---|---|
# MAGIC | 1 | Transform Bronze → Silver: clean nulls, dedupe, standardize column names |  All 5 tables |
# MAGIC | 2 | Manage malformed records → quarantine |  5 quarantine streaming tables |
# MAGIC | 3 | Pandas UDFs to standardize DataFrame/Table headers |  `standardize_text`, `clean_text`, `standardize_geo` + `CATALOG_COL_MAP` |
# MAGIC | 4 | Descriptions/metadata for enterprise discoverability |  `comment=` on every `@dlt.table` |
# MAGIC
# MAGIC ### Silver tables produced:
# MAGIC ```
# MAGIC Bronze (streaming)           Silver (streaming)             Quarantine (streaming)
# MAGIC ──────────────────           ──────────────────             ──────────────────────
# MAGIC listings_csv_copyinto ┐
# MAGIC listings_json_autoloader ├──► listings_silver_merged  ────► listings_main_quarantine
# MAGIC listings_xml_pyspark  │
# MAGIC listings_csv_dlt      ┘
# MAGIC listings_text         ──────► listings_text           ────► listings_text_quarantine
# MAGIC listings_photo        ──────► listings_photo          ────► listings_photo_quarantine
# MAGIC car_catalog           ──────► car_catalog             ────► car_catalog_quarantine
# MAGIC geo_locations         ──────► geography               ────► geography_quarantine
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import dlt
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

CATALOG  = spark.conf.get("pipeline.catalog",        "vstone_catalog")
BRONZE   = spark.conf.get("pipeline.bronze_schema",  "bronze")
SILVER   = spark.conf.get("pipeline.silver_schema",  "silver")

USD_RATE = 82.5  # Feb 2023 historical RUB/USD rate

SILVER_PROPS = {
    "quality"                   : "silver",
    "delta.enableChangeDataFeed": "true",
    "pipelines.reset.allowed"   : "true",
}
QUARANTINE_PROPS = {
    "quality"                   : "quarantine",
    "delta.enableChangeDataFeed": "true",
    "pipelines.reset.allowed"   : "true",
}

print(f"Catalog: {CATALOG} | Bronze: {BRONZE} | Silver: {SILVER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas UDFs
# MAGIC Pandas UDFs to standardize DataFrame/Table headers and column values.
# MAGIC Defined once, reused across all 5 table transformation functions.

# COMMAND ----------

# --- UPDATED UDF TO PREVENT CORRUPTED_DATA ---
@F.pandas_udf(StringType())

def standardize_text(s: pd.Series) -> pd.Series:

    """Lowercase + strip whitespace. Used on brand, model, photo_url for joins/dedup."""

    return s.astype(str).str.strip().str.lower()

@F.pandas_udf(StringType())
def clean_text(s: pd.Series) -> pd.Series:
    """Strip whitespace only. Used on Cyrillic catalog text columns."""
    return s.astype(str).str.strip()

@F.pandas_udf(StringType())
def standardize_geo(s: pd.Series) -> pd.Series:
    """Strip whitespace for city name standardization before geo joins."""
    return s.astype(str).str.strip()

print("UDFs registered: standardize_text | clean_text | standardize_geo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared Helpers
# MAGIC
# MAGIC `deduplicate()` — single deduplication function used by every table.
# MAGIC `_is_valid_russia()` — Russia bounding box filter reused by geography + quarantine.

# COMMAND ----------

def deduplicate(df, partition_cols: list, order_col: str = "bronze_load_dt"):
    """
    Streaming-safe deduplication using dropDuplicates().
    ROW_NUMBER() with PARTITION BY is NOT supported in Structured Streaming —
    it requires a full sort over unbounded state which is incompatible with
    incremental stream processing.
    dropDuplicates(partition_cols) is the correct streaming alternative —
    it keeps the first occurrence of each key within each micro-batch.
    """
    return df.dropDuplicates(partition_cols)

def _is_valid_russia(df):
    """Russia bounding box: lat 41-82N, lon 19-180E."""
    return (
        F.col("latitude").isNotNull()  & F.col("longitude").isNotNull()  &
        F.col("latitude").between(41, 82) & F.col("longitude").between(19, 180)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 1 — `listings_silver_merged` + `listings_main_quarantine`
# MAGIC **Streaming source:** Union of 4 Bronze listing tables via `dlt.read_stream()`
# MAGIC
# MAGIC **Key transformations:** type casting, date parsing (2 formats), price enrichment,
# MAGIC UDF standardization on brand/model, deduplication by `listing_id`, price category derivation.

# COMMAND ----------

# Cyrillic → English column rename map (Deliverable 3)

CATALOG_COL_MAP = {

    "Марка": "brand", "Модель": "model", "Поколение": "generation",
    "Комплектация": "trim_level", "Объём двигателя": "engine_volume_l",
    "Мощность двигателя": "engine_power_hp", "Расход топлива": "fuel_consumption",
    "Тип топлива": "fuel_type", "Коробка передач": "transmission",
    "Привод": "drive_type", "Кол-во мест": "seats_count",
    "Клиренс": "clearance_mm", "Объем багажника": "trunk_volume_l",
    "Период выпуска": "production_period", "Тип кузова": "body_type",
    "Марка кузова": "body_mark", "Время разгона 0-100 км/ч, с": "acceleration_0_100",
    "Максимальная скорость, км/ч": "max_speed_kmh", "Страна сборки": "assembly_country",

}


def _build_listings_stream():
    """
    Unions 4 Bronze listing streams via spark.readStream on the full Delta table path.
    dlt.read_stream() only works for tables in the SAME pipeline — Bronze tables live
    in a separate pipeline so we read them directly as Delta streams instead.
    allowMissingColumns=True handles column differences across CSV/JSON/XML sources.

    FIX — DELTA_SOURCE_IGNORE_DELETE:
    Bronze tables may have rows deleted due to OPTIMIZE compaction or table recreation.
    DLT streaming fails hard on deletes by default.
    ignoreDeletes=true  → silently skip delete operations at the Delta log level.
    ignoreChanges=true  → also tolerate updates/rewrites (covers OPTIMIZE + VACUUM).
    Both options are safe here because Bronze is append-only from our ingest pipelines.
    """
    def _bronze_stream(table):
        return (spark.readStream
                .format("delta")
                .option("ignoreDeletes", "true")
                .option("ignoreChanges", "true")
                .table(f"{CATALOG}.{BRONZE}.{table}"))
    return (
        _bronze_stream("listings_csv_copyinto")
        .unionByName(_bronze_stream("listings_json_autoloader"), allowMissingColumns=True)
        .unionByName(_bronze_stream("listings_xml_pyspark"),     allowMissingColumns=True)
        .unionByName(_bronze_stream("listings_csv_dlt"),         allowMissingColumns=True)
    )


def _transform_listings(df):
    """
    Full Bronze→Silver transformation for main listings.
    Casts all columns, parses dates, applies UDFs, adds enrichment columns.
    load_dt and source_file preserved from Bronze as bronze_load_dt / bronze_source_file.
    """
    return (df.select(
        F.expr("try_cast(id as long)").cast("string").alias("listing_id"),
        F.coalesce(
            F.to_timestamp(F.col("date"), "dd.MM.yyyy"),
            F.to_timestamp(F.col("date"), "yyyy-MM-dd'T'HH:mm:ss'Z'")
        ).alias("listing_date"),
        F.expr("try_cast(regexp_replace(cost, '[^0-9.]', '') as double)").alias("price_rub"),
        F.col("currency"),
        standardize_text(F.col("marka")).alias("brand"),
        standardize_text(F.col("model")).alias("model"),
        F.expr("try_cast(try_cast(year as double) as int)").alias("manufacture_year"),
        F.expr("try_cast(try_cast(power as double) as int)").alias("engine_power"),
        F.expr("try_cast(try_cast(probeg as double) as int)").alias("mileage_km"),
        F.col("engine").alias("fuel_type"),
        F.col("transmission").alias("transmission_type"),
        F.col("gear").alias("drive_type"),
        F.col("sWheel").alias("steering_wheel"),
        F.col("complectation").alias("trim_level"),
        F.col("place").alias("city_prepositional"),
        F.expr("try_cast(has_license as int)").alias("has_license"),
        F.expr("coalesce(cast(R as string), '')").alias("color_r"),
        F.expr("coalesce(cast(G as string), '')").alias("color_g"),
        F.expr("coalesce(cast(B as string), '')").alias("color_b"),
        F.col("source_file").alias("bronze_source_file"),
        F.col("load_dt").alias("bronze_load_dt"),
    )
    .withColumn("price_usd",      F.round(F.col("price_rub") / USD_RATE, 2))
    .withColumn("car_age_years",  F.lit(2023) - F.col("manufacture_year").cast("integer"))
    .withColumn("price_category",
        F.when(F.col("price_rub") < 300000,               "BUDGET")
        .when(F.col("price_rub").between(300000, 700000),  "MID_RANGE")
        .when(F.col("price_rub").between(700001, 1500000), "PREMIUM")
        .when(F.col("price_rub") > 1500000,               "LUXURY")
        .otherwise("UNKNOWN"))
    .withColumn("listing_year",   F.date_format(F.col("listing_date"), "yyyy").cast("integer"))
    .withColumn("listing_month",  F.date_format(F.col("listing_date"), "MM").cast("integer"))
    .withColumn("brand_std",      F.upper(F.trim(F.col("brand"))))
    .withColumn("silver_load_dt", F.current_timestamp()))


_LISTINGS_VALID_FILTER = (
    F.col("listing_id").isNotNull() &
    F.col("price_rub").isNotNull()  &
    F.col("listing_date").isNotNull()
)


@dlt.table(
    name             = "listings_silver_merged",
    comment          = "Silver Streaming — unified car listings from 4 Bronze sources. "
                       "Enriched with price_usd, car_age_years, price_category, brand_std. "
                       "Deduplicated by listing_id (latest bronze_load_dt wins). "
                       "Source: listings_csv_copyinto + json_autoloader + xml_pyspark + csv_dlt.",
    table_properties = SILVER_PROPS
)
@dlt.expect("valid_listing_id", "listing_id IS NOT NULL")
@dlt.expect("valid_price",      "price_rub IS NOT NULL")
@dlt.expect("valid_date",       "listing_date IS NOT NULL")
@dlt.expect("positive_price",   "price_rub > 0")
# @dlt.expect("valid_brand_name", "brand != 'CORRUPTED_DATA'")
def listings_silver_merged():
    df = _build_listings_stream()
    df = _transform_listings(df)
    df = deduplicate(df, ["listing_id"])
    return df.filter(_LISTINGS_VALID_FILTER)


@dlt.table(
    name             = "listings_main_quarantine",
    comment          = "Quarantine Streaming — listings rejected from listings_silver_merged. "
                       "Reasons: MISSING_OR_MALFORMED_ID | INVALID_PRICE_FORMAT | UNPARSABLE_DATE_FORMAT.",
    table_properties = QUARANTINE_PROPS
)
def listings_main_quarantine():
    df = _build_listings_stream()
    df = _transform_listings(df)
    df = deduplicate(df, ["listing_id"])
    return (df
        .filter(~_LISTINGS_VALID_FILTER)
        .withColumn("quarantine_reason",
            F.when(F.col("listing_id").isNull(),   "MISSING_OR_MALFORMED_ID")
            .when(F.col("price_rub").isNull(),      "INVALID_PRICE_FORMAT")
            .when(F.col("listing_date").isNull(),   "UNPARSABLE_DATE_FORMAT")
            .otherwise("DATA_QUALITY_ISSUE"))
        .withColumn("quarantine_dt", F.current_timestamp()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 2 — `listings_text_transformation` + `listings_text_quarantine`
# MAGIC **Streaming source:** `dlt.read_stream('listings_text')` from Bronze

# COMMAND ----------

def _transform_text(df):
    """Cast id float-string → integer-string. Preserve Bronze audit cols."""
    return (df
        .withColumn("listing_id",         F.col("id").cast("double").cast("long").cast("string"))
        .withColumn("bronze_load_dt",     F.col("load_dt"))
        .withColumn("bronze_source_file", F.col("source_file"))
        .withColumn("silver_load_dt",     F.current_timestamp())
        .drop("id", "_rescued_data"))

@dlt.table(
    name             = "listings_text_transformation",
    comment          = "Silver Streaming — Russian car listing descriptions, cleaned and deduplicated. "
                       "Primary key: listing_id. Source: bronze.listings_text (1_text.csv).",
    table_properties = SILVER_PROPS
)
@dlt.expect("valid_listing_id", "listing_id IS NOT NULL")
@dlt.expect("has_text",         "text IS NOT NULL")
def listings_text():
    df = _transform_text(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.listings_text"))
    return deduplicate(df, ["listing_id"]).filter(F.col("listing_id").isNotNull())

@dlt.table(
    name             = "listings_text_quarantine",
    comment          = "Quarantine Streaming — text records with null or malformed listing_id.",
    table_properties = QUARANTINE_PROPS
)
def listings_text_quarantine():
    df = _transform_text(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.listings_text"))
    return (deduplicate(df, ["listing_id"])
        .filter(F.col("listing_id").isNull())
        .withColumn("quarantine_reason", F.lit("MALFORMED_OR_NULL_ID"))
        .withColumn("quarantine_dt",     F.current_timestamp()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 3 — `listings_photo_transformation` + `listings_photo_quarantine`
# MAGIC **Streaming source:** `dlt.read_stream('listings_photo')` from Bronze

# COMMAND ----------

def _transform_photo(df):
    """
    Cast id → listing_id. Standardize photo_url via UDF for deduplication.
    Drop _c0 (unnamed pandas index) and _rescued_data.
    """
    return (df
        .withColumn("listing_id",         F.col("id").cast("double").cast("long").cast("string"))
        .withColumn("photo_url_clean",    standardize_text(F.col("photo_url")))
        .withColumn("bronze_load_dt",     F.col("load_dt"))
        .withColumn("bronze_source_file", F.col("source_file"))
        .withColumn("silver_load_dt",     F.current_timestamp())
        .drop("id", "_c0", "_rescued_data"))

@dlt.table(
    name             = "listings_photo_transformation",
    comment          = "Silver Streaming — deduplicated photo URLs per listing. "
                       "Dedup key: listing_id + photo_url_clean. Source: bronze.listings_photo (1_photo.csv).",
    table_properties = SILVER_PROPS
)
@dlt.expect("valid_listing_id", "listing_id IS NOT NULL")
@dlt.expect("has_photo_url",    "photo_url IS NOT NULL")
def listings_photo():
    df = _transform_photo(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.listings_photo"))
    return deduplicate(df, ["listing_id", "photo_url_clean"]).filter(F.col("listing_id").isNotNull())

@dlt.table(
    name             = "listings_photo_quarantine",
    comment          = "Quarantine Streaming — photo records with null or malformed listing_id.",
    table_properties = QUARANTINE_PROPS
)
def listings_photo_quarantine():
    df = _transform_photo(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.listings_photo"))
    return (deduplicate(df, ["listing_id", "photo_url_clean"])
        .filter(F.col("listing_id").isNull())
        .withColumn("quarantine_reason", F.lit("MALFORMED_OR_NULL_ID"))
        .withColumn("quarantine_dt",     F.current_timestamp()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 4 — `car_catalog_transformations` + `car_catalog_quarantine`
# MAGIC **Streaming source:** `dlt.read_stream('car_catalog')` from Bronze

# COMMAND ----------

def _transform_catalog(df):
    """
    Rename 19 Cyrillic columns → English via CATALOG_COL_MAP (Deliverable 3).
    Parse numeric columns stripping Russian unit suffixes.
    Apply clean_text UDF to all string columns.
    """
    for ru, en in CATALOG_COL_MAP.items():
        if ru in df.columns:
            df = df.withColumnRenamed(ru, en)
    df = (df
        .withColumn("engine_volume_l",    F.expr("try_cast(regexp_replace(regexp_replace(engine_volume_l, ' л', ''), ',', '.') as double)"))
        .withColumn("engine_power_hp",    F.expr("try_cast(regexp_replace(engine_power_hp, ' л.с.', '') as int)"))
        .withColumn("clearance_mm",       F.expr("try_cast(regexp_replace(clearance_mm, ' мм', '') as int)"))
        .withColumn("trunk_volume_l",     F.expr("try_cast(regexp_replace(trunk_volume_l, ' л', '') as int)"))
        .withColumn("seats_count",        F.expr("try_cast(regexp_replace(seats_count, ' мест', '') as int)"))
        .withColumn("acceleration_0_100", F.expr("try_cast(regexp_replace(acceleration_0_100, ',', '.') as double)"))
        .withColumn("max_speed_kmh",      F.expr("try_cast(max_speed_kmh as int)")))
    for c in ["brand","model","generation","trim_level","fuel_type","transmission","drive_type","body_type"]:
        if c in df.columns:
            df = df.withColumn(c, clean_text(F.col(c)))
    return (df
        .withColumn("bronze_load_dt",     F.col("load_dt"))
        .withColumn("bronze_source_file", F.col("source_file"))
        .withColumn("silver_load_dt",     F.current_timestamp())
        .drop("_rescued_data"))

@dlt.table(
    name             = "car_catalog_transformation",
    comment          = "Silver Streaming — car specs. Deduplicated by core attributes.",
    table_properties = SILVER_PROPS
)
@dlt.expect("valid_brand", "brand IS NOT NULL")
@dlt.expect("valid_model", "model IS NOT NULL")
def car_catalog():
    # Helper call and stream reading
    df = _transform_catalog(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.car_catalog"))
    
    # Corrected return statement: No trailing words outside brackets
    return (df
        .dropDuplicates(["brand", "model", "generation", "trim_level", "engine_volume_l", "engine_power_hp"])
        .filter(F.col("brand").isNotNull())) # Bracket closed correctly here

@dlt.table(
    name             = "car_catalog_quarantine",
    comment          = "Quarantine Streaming — catalog records with null brand (unidentifiable make).",
    table_properties = QUARANTINE_PROPS
)
def car_catalog_quarantine():
    df = _transform_catalog(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.car_catalog"))
    return (df
        .filter(F.col("brand").isNull())
        .withColumn("quarantine_reason", F.lit("MISSING_BRAND"))
        .withColumn("quarantine_dt",     F.current_timestamp()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 5 — `geography` + `geography_quarantine`
# MAGIC **Streaming source:** `dlt.read_stream('geo_locations')` from Bronze

# COMMAND ----------

def _transform_geo(df):
    """
    Rename columns, cast lat/lon to DOUBLE, standardize city names via UDF.
    Preserve Bronze load_dt and source_file.
    """
    return (df
        .select(
            standardize_geo(F.col("name_padesh")).alias("city_name"),
            F.col("greate_padesh").alias("city_prepositional"),
            F.col("lat").cast("double").alias("latitude"),
            F.col("lon").cast("double").alias("longitude"),
            F.col("load_dt").alias("bronze_load_dt"),
            F.col("source_file").alias("bronze_source_file"),
        )
        .withColumn("silver_load_dt", F.current_timestamp()))

@dlt.table(
    name             = "geography_transformation",
    comment          = "Silver Streaming — Russian city coordinates validated against Russia bounding box "
                       "(lat 41-82N, lon 19-180E). Deduplicated by city_name + city_prepositional. "
                       "Source: bronze.geo_locations (final_geografic.csv).",
    table_properties = SILVER_PROPS
)
@dlt.expect("valid_city",     "city_name IS NOT NULL")
@dlt.expect("valid_latitude", "latitude IS NOT NULL")
@dlt.expect("valid_longitude","longitude IS NOT NULL")
@dlt.expect("russia_bounds",  "latitude BETWEEN 41 AND 82 AND longitude BETWEEN 19 AND 180")
def geography():
    df = _transform_geo(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.geo_locations"))
    df_deduped = df.dropDuplicates(["city_name", "city_prepositional"])
    return df_deduped.filter(_is_valid_russia(df_deduped))

@dlt.table(
    name             = "geography_quarantine",
    comment          = "Quarantine Streaming — geo records outside Russia bounding box or with null coordinates.",
    table_properties = QUARANTINE_PROPS
)
def geography_quarantine():
    df = _transform_geo(spark.readStream.format("delta").table(f"{CATALOG}.{BRONZE}.geo_locations"))
    df_deduped = df.dropDuplicates(["city_name", "city_prepositional"])
    return (df_deduped
        .filter(~_is_valid_russia(df_deduped))
        .withColumn("quarantine_reason", F.lit("COORDINATES_OUTSIDE_RUSSIA_OR_NULL"))
        .withColumn("quarantine_dt",     F.current_timestamp()))
