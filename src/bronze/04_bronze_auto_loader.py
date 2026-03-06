# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup & Widgets

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

print(f"INFO: Initializing ingestion for {file_name} into {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Metadata & Source Isolation

# COMMAND ----------

# Using a dedicated sub-folder for this specific table to isolate metadata
CHECKPOINT_BASE = f"{OUTPUT_BASE_PATH}/streaming_metadata/listings_json"
schema_location = f"{CHECKPOINT_BASE}/schema"
checkpoint_location = f"{CHECKPOINT_BASE}/chkpt"

# Isolated source path to prevent reading ghost/corrupted metadata from root volume
source_path = f"{OUTPUT_BASE_PATH}/isolated_json_source/"
clean_file_path = f"{source_path}{file_name}"
old_file_path = f"{OUTPUT_BASE_PATH}/{file_name}"

# Ensure metadata and isolated directories exist
print("INFO: Setting up isolated source and metadata directories...")
dbutils.fs.mkdirs(source_path)
dbutils.fs.mkdirs(schema_location)
dbutils.fs.mkdirs(checkpoint_location)

try:
    # Copy file to isolated directory
    dbutils.fs.cp(old_file_path, clean_file_path)
except Exception:
    pass # File already exists or copy failed

print(f"INFO: Source ready at {source_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Auto Loader Stream Configuration

# COMMAND ----------

# Is cell ko ek baar run karein
dbutils.fs.rm(CHECKPOINT_BASE, True)
print(f"INFO: Metadata cleared at {CHECKPOINT_BASE}")

# COMMAND ----------

# Updated Cell 4: Idempotent Write using MERGE
print(f"INFO: Streaming JSON into {TARGET_TABLE} with Idempotency (MERGE)...")

# 1. Upsert Function (DABs friendly)
def upsert_to_delta(microBatchDF, batchId):
    microBatchDF.createOrReplaceTempView("updates")
    # MERGE logic: ID match hone par insert nahi karega
    microBatchDF._jdf.sparkSession().sql(f"""
        MERGE INTO {TARGET_TABLE} AS target
        USING updates AS source
        ON target.id = source.id
        WHEN NOT MATCHED THEN
          INSERT *
    """)

# 2. Start the Stream
query = (df_json_enriched.writeStream
    .foreachBatch(upsert_to_delta) # Append ki jagah foreachBatch use karein
    .option("checkpointLocation", checkpoint_location)
    .trigger(availableNow=True) 
    .start())

query.awaitTermination()
print("SUCCESS: Idempotent JSON Streaming completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Write Stream & Execution

# COMMAND ----------

# 3. Write Stream to Bronze Table
print(f"INFO: Streaming JSON into {TARGET_TABLE}...")

query = (df_json_enriched.writeStream
    .format("delta")
    .option("checkpointLocation", checkpoint_location)
    .option("mergeSchema", "true")
    .trigger(availableNow=True) 
    .outputMode("append")
    .toTable(TARGET_TABLE)) 

# Block execution until the stream finishes
query.awaitTermination()
print("SUCCESS: JSON Streaming completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Verification & Audit

# COMMAND ----------

# Verification
if spark.catalog.tableExists(TARGET_TABLE):
    total_count = spark.table(TARGET_TABLE).count()
    print(f"\n[SUMMARY]")
    print(f"Table Name     : {TARGET_TABLE}")
    print(f"Total Records  : {total_count:,}")
    print(f"Checkpoint     : {checkpoint_location}")
else:
    print("ERROR: Table was not created. Check stream logs.")

# COMMAND ----------

# MAGIC %md
# MAGIC # data verification

# COMMAND ----------

# MAGIC %sql
# MAGIC  select * from vstone_catalog.bronze.listings_json_autoloader limit 5;
