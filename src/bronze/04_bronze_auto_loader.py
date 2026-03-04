# Databricks notebook source
# MAGIC %md
# MAGIC #  Widgets & Configuration

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 1. Wipe the entire schema and all tables inside it
# MAGIC DROP SCHEMA IF EXISTS vstone_catalog.bronze CASCADE;
# MAGIC
# MAGIC -- 2. Recreate the completely empty schema for your pipeline
# MAGIC CREATE SCHEMA vstone_catalog.bronze;

# COMMAND ----------

import time
from pyspark.sql.functions import current_timestamp, lit

# ======================================================================================
# Section 1: DYNAMIC CONFIGURATION
# ======================================================================================
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Target Catalog Name")
dbutils.widgets.text("raw_schema", "raw", "2. Raw Schema Name")
dbutils.widgets.text("bronze_schema", "bronze", "3. Bronze Schema Name")

CATALOG = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
BRONZE = dbutils.widgets.get("bronze_schema")

file_name = "1_main_chunk_3.json"

# The existing volume path where you DO have permissions
base_volume_path = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/chunks/"
old_file_path = f"{base_volume_path}{file_name}"

# 🚨 THE FIX: Create a SUBDIRECTORY inside your allowed volume!
clean_input_dir = f"{base_volume_path}pristine_json/"
clean_file_path = f"{clean_input_dir}{file_name}"

print("🧹 Setting up an isolated subdirectory inside the permitted Volume...")
# This is allowed because you have access to the parent Volume
dbutils.fs.mkdirs(clean_input_dir)

try:
    # Copy the JSON file into this clean subdirectory
    dbutils.fs.cp(old_file_path, clean_file_path)
    print(f"✅ Successfully copied {file_name} to the clean subdirectory.")
except Exception as e:
    print(f"⚠️ Note: File might already exist. Proceeding...")

# Setup unique checkpoint and target table
unique_run_id = int(time.time())
checkpoint_path = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/checkpoints/al_run_{unique_run_id}"
TARGET_TABLE = f"{CATALOG}.{BRONZE}.listings_json_al_final"

print(f"🎯 Target Table: {TARGET_TABLE}")

# ======================================================================================
# Section 2: AUTO LOADER READ STREAM (From Clean Subdirectory)
# ======================================================================================
print("⏳ Starting Auto Loader Stream...")

bronze_schema_ddl = """
    cost STRING, currency STRING, marka STRING, model STRING, year STRING, 
    has_license STRING, place STRING, date STRING, id STRING, engine STRING, 
    power STRING, gear STRING, probeg STRING, sWheel STRING, complectation STRING, 
    transmission STRING, R STRING, G STRING, B STRING
"""

df_json = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .schema(bronze_schema_ddl) 
    .option("pathGlobFilter", file_name)
    .option("multiLine", "true") 
    .load(clean_input_dir)) # 👈 Pointing EXACTLY to the new, uncontaminated subdirectory

# ======================================================================================
# Section 3: WRITE STREAM 
# ======================================================================================
print(f"📝 Writing to {TARGET_TABLE}...")

query = (df_json
    .withColumn("load_dt", current_timestamp())
    .withColumn("source_file", lit(file_name))
    .writeStream
    .option("checkpointLocation", checkpoint_path) 
    .trigger(availableNow=True) 
    .toTable(TARGET_TABLE))

query.awaitTermination()

# ======================================================================================
# Section 4: VERIFICATION
# ======================================================================================
total_count = spark.table(TARGET_TABLE).count()
print(f"✅ SUCCESS! Auto Loader ingested JSON. Total records in {TARGET_TABLE}: {total_count:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM vstone_catalog.bronze.listings_json_autoloader;

# COMMAND ----------

# MAGIC %md
# MAGIC # JSON Ingestion (Chunk 3) via Auto Loader

# COMMAND ----------

# print(f" Auto Loader Schema Path: {schema_path}")
# print(f" Target Table: {TARGET_TABLE}")

# # ======================================================================================
# # Section 2: AUTO LOADER READ STREAM (Pure String Ingestion)
# # ======================================================================================
# print(" Starting Auto Loader Stream...")

# df_json = (spark.readStream
#     .format("cloudFiles")
#     .option("cloudFiles.format", "json")
#     .option("cloudFiles.schemaLocation", schema_path) # Safe UC Volume Path
#     .option("pathGlobFilter", "1_main_chunk_3.json")
#     .option("multiLine", "true") 
#     .option("cloudFiles.inferColumnTypes", "false") # Read as strings for Bronze
#     .load(input_path))

# # ======================================================================================
# # Section 3: WRITE STREAM (Idempotent & Resilient)
# # ======================================================================================
# query = (df_json
#     .withColumn("load_dt", current_timestamp())
#     .withColumn("source_file", lit("1_main_chunk_3.json"))
#     .writeStream
#     .option("checkpointLocation", checkpoint_path) # Safe UC Volume Path
#     .trigger(availableNow=True) # Processes all available data and then stops
#     .toTable(TARGET_TABLE))

# # Wait for the stream to finish processing the current batch
# query.awaitTermination()

# # Verification
# total_count = spark.table(TARGET_TABLE).count()
# print(f" Auto Loader successfully ingested JSON. Total records in {TARGET_TABLE}: {total_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC  select * from vstone_catalog.bronze.listings_json_autoloader limit 5;
