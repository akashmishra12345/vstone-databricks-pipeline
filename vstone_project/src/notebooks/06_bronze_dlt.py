# Databricks notebook source
# MAGIC %md
# MAGIC #CSV Ingestion via Delta Live Tables (Chunk 2)

# COMMAND ----------

# DLT Pipeline Script: 04_bronze_dlt_IDEMPOTENT
import dlt
from pyspark.sql.functions import current_timestamp, lit

import dlt
from pyspark.sql.functions import current_timestamp, lit

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
        # Use .read_stream().format("cloudFiles") explicitly
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("pathGlobFilter", "1_main_chunk_2.csv")
        .option("cloudFiles.inferColumnTypes", "false")
        .load("/Volumes/vstone_catalog/raw/chunks/") # Ensure this path is correct
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit("1_main_chunk_2.csv"))
    )
