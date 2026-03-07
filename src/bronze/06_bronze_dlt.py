# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Bronze DLT Pipeline | CSV Ingestion (Chunk 2)
# MAGIC
# MAGIC Ingests `1_main_chunk_2.csv` into `vstone_catalog.bronze.listings_csv_dlt`
# MAGIC using **Delta Live Tables + Auto Loader**.
# MAGIC
# MAGIC ### Why column names were broken (`1`, `2000`, `TOYOTA`...):
# MAGIC Auto Loader had stale schema metadata cached in `schemaLocation` from a previous
# MAGIC run where the CSV was read without `header=true`. The cached broken names
# MAGIC (`1`, `2000`, `TOYOTA`) were reused on every subsequent run even after the fix.
# MAGIC
# MAGIC ### Fix applied:
# MAGIC - Explicit `BRONZE_SCHEMA` passed via `.schema()` — Auto Loader never infers names
# MAGIC - `cloudFiles.schemaEvolutionMode = none` — cached schema is ignored, explicit DDL wins
# MAGIC - `cloudFiles.inferColumnTypes = false` — all columns stay STRING in Bronze
# MAGIC
# MAGIC | Design Decision           | Choice                              | Reason |
# MAGIC |---------------------------|-------------------------------------|--------|
# MAGIC | Schema                    | Explicit `StructType`, `inferSchema=false` | Correct column names locked in code |
# MAGIC | Schema evolution          | `schemaEvolutionMode = none`        | Prevents stale cache overriding DDL |
# MAGIC | Idempotency               | Auto Loader checkpoint + DLT state  | Files tracked, never re-ingested |
# MAGIC | Change Data Feed          | `delta.enableChangeDataFeed = true` | Enables efficient CDC for Silver/Gold |
# MAGIC | Audit columns             | `load_dt`, `source_file`            | Mandatory on every row |

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC DLT pipelines use `spark.conf.get()` — `dbutils.widgets` is not available
# MAGIC inside a DLT pipeline execution context.

# COMMAND ----------

CATALOG    = spark.conf.get("pipeline.catalog",        "vstone_catalog")
RAW_SCHEMA = spark.conf.get("pipeline.raw_schema",     "raw")
VOLUME     = spark.conf.get("pipeline.chunks_volume",  "chunks")

FILE_NAME  = "1_main_chunk_2.csv"
INPUT_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{VOLUME}/"

print(f"Input path  : {INPUT_PATH}")
print(f"Source file : {FILE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Schema
# MAGIC
# MAGIC All 19 columns defined explicitly as STRING.
# MAGIC `inferSchema = false` — Bronze is a raw landing zone, no type casting here.
# MAGIC Column names match the CSV header row exactly (case-sensitive).

# COMMAND ----------

BRONZE_SCHEMA = StructType([
    StructField("cost",          StringType(), True),
    StructField("currency",      StringType(), True),
    StructField("marka",         StringType(), True),
    StructField("model",         StringType(), True),
    StructField("year",          StringType(), True),
    StructField("has_license",   StringType(), True),
    StructField("place",         StringType(), True),
    StructField("date",          StringType(), True),
    StructField("id",            StringType(), True),
    StructField("engine",        StringType(), True),
    StructField("power",         StringType(), True),
    StructField("gear",          StringType(), True),
    StructField("probeg",        StringType(), True),
    StructField("sWheel",        StringType(), True),
    StructField("complectation", StringType(), True),
    StructField("transmission",  StringType(), True),
    StructField("R",             StringType(), True),
    StructField("G",             StringType(), True),
    StructField("B",             StringType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Table — `listings_csv_dlt`
# MAGIC
# MAGIC **Auto Loader options:**
# MAGIC - `header = true` — reads column names from row 1 of the CSV
# MAGIC - `schema(BRONZE_SCHEMA)` — locks column names, overrides any cached schema
# MAGIC - `cloudFiles.inferColumnTypes = false` — no type inference, all STRING
# MAGIC - `cloudFiles.schemaEvolutionMode = none` — rejects schema changes, explicit DDL always wins
# MAGIC - `pathGlobFilter` — processes only `1_main_chunk_2.csv` from the chunks folder
# MAGIC
# MAGIC **DLT expectations:**
# MAGIC - `valid_id` — tracks rows where `id IS NULL` in the DLT event log
# MAGIC - `valid_cost` — tracks rows where `cost IS NULL`

# COMMAND ----------

@dlt.table(
    name    = "listings_csv_dlt",
    comment = "Bronze — raw CSV listings from chunk 2 (20% split), ingested via DLT Auto Loader",
    table_properties = {
        "quality"                   : "bronze",
        "delta.enableChangeDataFeed": "true",   # CDF: enables efficient CDC for Silver/Gold
        "pipelines.reset.allowed"   : "true"    # Allow full refresh from DLT UI
    }
)
@dlt.expect("valid_id",   "id IS NOT NULL")
@dlt.expect("valid_cost", "cost IS NOT NULL")
def listings_csv_dlt():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",              "csv")
        .option("cloudFiles.inferColumnTypes",    "false") # All STRING — no inference
        .option("cloudFiles.schemaEvolutionMode", "none")  # Explicit schema always wins — fixes stale cache
        .option("header",                         "true")  # Column names from row 1
        .option("encoding",                       "UTF-8") # Cyrillic in place/marka/model
        .option("pathGlobFilter",                 FILE_NAME)
        .schema(BRONZE_SCHEMA)                             # Explicit DDL — locks column names
        .load(INPUT_PATH)
        .withColumn("load_dt",     F.current_timestamp())  # Mandatory audit: ingestion timestamp
        .withColumn("source_file", F.lit(FILE_NAME))       # Mandatory audit: origin file
    )
