# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Bronze DLT Pipeline | CSV Ingestion (Chunk 2)
# MAGIC
# MAGIC Ingests `1_main_chunk_2.csv` into `vstone_catalog.bronze.listings_csv_dlt`
# MAGIC using Delta Live Tables.

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

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
