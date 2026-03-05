# Databricks notebook source
# MAGIC %md
# MAGIC # Ingestion using autoloader

# COMMAND ----------

import pyspark.sql.functions as F

# 1. Define Widgets for dynamic CI/CD parameter injection
dbutils.widgets.text("project_catalog", "vstone_catalog")
dbutils.widgets.text("raw_schema", "raw")
dbutils.widgets.text("bronze_schema", "bronze")

# 2. Fetch the values
CATALOG = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
BRONZE = dbutils.widgets.get("bronze_schema")

file_name = "1_main_chunk_3.json"
TARGET_TABLE = f"{CATALOG}.{BRONZE}.listings_json_autoloader"

# Base paths for data in Unity Catalog Volumes
OUTPUT_BASE_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/chunks"

# Using a dedicated sub-folder for this specific table
CHECKPOINT_BASE = f"{OUTPUT_BASE_PATH}/streaming_metadata/listings_json"
schema_location = f"{CHECKPOINT_BASE}/schema"
checkpoint_location = f"{CHECKPOINT_BASE}/chkpt"

# Isolated source path to prevent reading ghost/corrupted metadata from root volume
source_path = f"{OUTPUT_BASE_PATH}/isolated_json_source/"
clean_file_path = f"{source_path}{file_name}"
old_file_path = f"{OUTPUT_BASE_PATH}/{file_name}"

# Ensure metadata and isolated directories exist to avoid permission race conditions
print(" Setting up isolated source and metadata directories...")
dbutils.fs.mkdirs(source_path)
dbutils.fs.mkdirs(schema_location)
dbutils.fs.mkdirs(checkpoint_location)

try:
    # Copy file to isolated directory
    dbutils.fs.cp(old_file_path, clean_file_path)
except Exception:
    pass # File already exists

print("===================================================")
print("     INGESTING LISTINGS (JSON) VIA AUTO LOADER     ")
print("===================================================\n")

print(f" Streaming JSON into {TARGET_TABLE}...")

# Explicit DDL Schema to guarantee bronze layer stability
bronze_schema_ddl = """
    cost STRING, currency STRING, marka STRING, model STRING, year STRING,
    has_license STRING, place STRING, date STRING, id STRING, engine STRING,
    power STRING, gear STRING, probeg STRING, sWheel STRING, complectation STRING,
    transmission STRING, R STRING, G STRING, B STRING
"""

# 1. Read Stream using Auto Loader (cloudFiles)
df_json = (spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", schema_location)
    .schema(bronze_schema_ddl)
    .option("pathGlobFilter", file_name)
    .option("multiLine", "true")
    .option("cloudFiles.useIncrementalListing", "auto") 
    .load(source_path))

# 2. Add Enterprise Audit Columns
df_json_enriched = (df_json
    .withColumn("load_dt", F.current_timestamp())
    .withColumn("source_file", F.col("_metadata.file_path")))

# 3. Write Stream to Bronze Table
query = (df_json_enriched.writeStream
    .format("delta")
    .option("checkpointLocation", checkpoint_location)
    .option("mergeSchema", "true")
    .trigger(availableNow=True) 
    #  THE FIX: Changed from "Complete" to "append"
    .outputMode("append")
    .toTable(TARGET_TABLE)) 

# Block execution until the stream finishes
query.awaitTermination()

# Verification
if spark.catalog.tableExists(TARGET_TABLE):
    total_count = spark.table(TARGET_TABLE).count()
    print(f" [SUCCESS] JSON ingested. Total records in {TARGET_TABLE}: {total_count:,}")
else:
    print(" Table was not created. Check stream logs.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC  select * from vstone_catalog.bronze.listings_json_autoloader limit 5;
