# Databricks notebook source
# MAGIC %md
# MAGIC #  Widgets & Configuration

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit

# Setup widgets for dynamic execution
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Target Catalog Name")
dbutils.widgets.text("raw_schema", "raw", "2. Raw Schema Name")
dbutils.widgets.text("bronze_schema", "bronze", "3. Bronze Schema Name")

# Fetch values into variables
CATALOG = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
BRONZE = dbutils.widgets.get("bronze_schema")

# 1. UNIQUE ISOLATED PATHS (Defining schema_path here)
input_path = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/chunks/"
checkpoint_root = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/checkpoints/autoloader_vFinal_Isolated"
schema_path = f"{checkpoint_root}/schema" # Fixed definition
checkpoint_path = f"{checkpoint_root}/checkpoint"

# Target Table Name based on catalog explorer
TARGET_TABLE = f"{CATALOG}.{BRONZE}.listings_json_autoloader"

# COMMAND ----------

# MAGIC %md
# MAGIC # JSON Ingestion (Chunk 3) via Auto Loader

# COMMAND ----------

# Section 2: AUTO LOADER: Pure String Ingestion Logic

# 3. AUTO LOADER: Pure String Ingestion
# inferColumnTypes=false ensures everything is read as string for Silver layer safety
df_json = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", schema_path) # Uses the variable from Section 1
    .option("pathGlobFilter", "1_main_chunk_3.json")
    .option("multiLine", "true") 
    .option("cloudFiles.inferColumnTypes", "false") 
    .load(input_path))

# 4. WRITE STREAM
# trigger(availableNow=True) handles backlogs and stops
query = (df_json
    .withColumn("load_dt", current_timestamp())
    .withColumn("source_file", lit("1_main_chunk_3.json"))
    .writeStream
    .option("checkpointLocation", checkpoint_path) 
    .option("mergeSchema", "true")
    .trigger(availableNow=True) 
    .toTable(TARGET_TABLE))

query.awaitTermination()

# 5. VERIFICATION
count = spark.table(TARGET_TABLE).count()
print(f" JSON Ingestion Complete. Total records in table: {count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC  select * from vstone_catalog.bronze.listings_json_autoloader limit 5;
