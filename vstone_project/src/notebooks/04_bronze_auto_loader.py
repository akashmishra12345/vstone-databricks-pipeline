# Databricks notebook source
# MAGIC %md
# MAGIC #JSON Ingestion (Chunk 3) via Auto Loader

# COMMAND ----------

# src/notebooks/04_bronze_autoloader_IDEMPOTENT.py
from pyspark.sql.functions import current_timestamp, lit

# 1. UNIQUE ISOLATED PATHS (Keep these consistent)
input_path = "/Volumes/vstone_catalog/raw/chunks/"
checkpoint_root = "/Volumes/vstone_catalog/raw/checkpoints/autoloader_vFinal_Isolated"
schema_path = f"{checkpoint_root}/schema"
checkpoint_path = f"{checkpoint_root}/checkpoint"

# 2. REMOVED: Metadata Cleanup
# Idempotency ke liye hum checkpoint delete NAHI karenge. 
# Isse Auto Loader ko pata rahega ki '1_main_chunk_3.json' pehle hi load ho chuki hai.

# 3. AUTO LOADER: Pure String Ingestion
df_json = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", schema_path) 
    .option("pathGlobFilter", "1_main_chunk_3.json")
    .option("multiLine", "true") 
    .option("cloudFiles.inferColumnTypes", "false") # All data as string for Silver safety
    .load(input_path))

# 4. WRITE STREAM
# Trigger AvailableNow ensures it processes new files and stops
query = (df_json
    .withColumn("load_dt", current_timestamp())
    .withColumn("source_file", lit("1_main_chunk_3.json"))
    .writeStream
    .option("checkpointLocation", checkpoint_path) 
    .option("mergeSchema", "true")
    .trigger(availableNow=True) # Essential for batch-style idempotency
    .toTable("vstone_catalog.bronze.listings_json_autoloader"))

query.awaitTermination()

# 5. VERIFICATION
count = spark.table("vstone_catalog.bronze.listings_json_autoloader").count()
print(f"✅ JSON Ingestion Complete. Total records in table: {count:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC  select * from vstone_catalog.bronze.listings_json_autoloader limit 5;
