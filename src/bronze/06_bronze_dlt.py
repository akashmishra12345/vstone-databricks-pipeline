# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------


import dlt
from pyspark.sql.functions import current_timestamp, lit

CATALOG = spark.conf.get("pipeline.catalog", "vstone_catalog")
RAW_SCHEMA = spark.conf.get("pipeline.raw_schema", "raw")
VOLUME = spark.conf.get("pipeline.chunks_volume", "chunks")

# Dynamic Source Path Construction
INPUT_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{VOLUME}/"
SOURCE_FILE = "1_main_chunk_2.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC #CSV Ingestion via Delta Live Tables (Chunk 2)

# COMMAND ----------

@dlt.table(
    name="listings_csv_dlt",
    comment="Bronze: Idempotent Ingestion of Chunk 2 CSV",
    table_properties={
        "quality": "bronze",
        "delta.enableChangeDataFeed": "true"
    }
)
def listings_csv_dlt():
    return (
        # Auto Loader (Cloud Files) ensures idempotency
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("pathGlobFilter", SOURCE_FILE)
        .option("cloudFiles.inferColumnTypes", "false") # Safety for string ingestion
        .load(INPUT_PATH) 
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit(SOURCE_FILE))
    )
