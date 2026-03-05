# Databricks notebook source

# Setup widgets for dynamic execution
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Target Catalog Name")
dbutils.widgets.text("bronze_schema", "bronze", "2. Bronze Schema Name")
dbutils.widgets.text("chunks_volume", "chunks", "3. Source Chunks Volume")

# Fetch values into variables
CATALOG = dbutils.widgets.get("project_catalog")
BRONZE = dbutils.widgets.get("bronze_schema")
VOLUME = dbutils.widgets.get("chunks_volume")

# Dynamic Construction of Paths and Tables
TABLE_NAME = f"{CATALOG}.{BRONZE}.listings_csv_copyinto"
FILE_PATH = f"/Volumes/{CATALOG}/raw/{VOLUME}/1_main_chunk_1.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC # Ingestion Logic (Copy Into)

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit
from pyspark.errors import AnalysisException

print(f" Starting Native PySpark Ingestion for {TABLE_NAME}...")

# ======================================================================================
# 1. READ & ENRICH (In-Memory Processing)
# ======================================================================================
FILE_NAME = "1_main_chunk_1.csv"

df_csv = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .load(FILE_PATH) \
    .withColumn("load_dt", current_timestamp()) \
    .withColumn("source_file", lit(FILE_NAME))

# ======================================================================================
# 2. IDEMPOTENCY CHECK (Serverless-Safe Try/Except Pattern)
# ======================================================================================
is_already_loaded = False

try:
    # Safely attempting to read the table. No blocked catalog API used.
    existing_df = spark.table(TABLE_NAME)
    
    # If table exists, check if our file is already inside
    loaded_count = existing_df.filter(existing_df.source_file == FILE_NAME).count()
    
    if loaded_count > 0:
        is_already_loaded = True
        print(f" Idempotency Check: '{FILE_NAME}' has already been loaded. Skipping to prevent duplicates.")

except AnalysisException:
    # If the table does not exist, Spark throws an AnalysisException. We catch it gracefully.
    print("ℹ Table does not exist yet. It will be created dynamically.")
    pass # is_already_loaded remains False

# ======================================================================================
# 3. WRITE TO DELTA (Auto Schema-Merge & Table Creation)
# ======================================================================================
if not is_already_loaded:
    print(f" Writing data to {TABLE_NAME}...")
    df_csv.write.format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable(TABLE_NAME)
    print(" Data successfully loaded!")

# ======================================================================================
# 4. VERIFICATION CHECK
# ======================================================================================
total_count = spark.table(TABLE_NAME).count()
print(f" Table {TABLE_NAME} state verified. Total records: {total_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Table data 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.bronze.listings_csv_copyinto limit 5;
