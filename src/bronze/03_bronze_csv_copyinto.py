# Databricks notebook source
# MAGIC %md
# MAGIC #Widgets & Configuration

# COMMAND ----------


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

# 1. CREATE TABLE ONLY IF NOT EXISTS (Schema lock)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    cost STRING, currency STRING, marka STRING, model STRING, year STRING,
    has_license STRING, place STRING, date STRING, id STRING, engine STRING,
    power STRING, gear STRING, probeg STRING, sWheel STRING, complectation STRING,
    transmission STRING, R STRING, G STRING, B STRING,
    load_dt TIMESTAMP, 
    source_file STRING
) USING DELTA
""")

# 2. IDEMPOTENT COPY INTO
spark.sql(f"""
COPY INTO {TABLE_NAME}
FROM '{FILE_PATH}'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false') 
COPY_OPTIONS ('mergeSchema' = 'true')
""")

# 3. SMART Audit Columns Update (ONLY FOR NEW DATA)
spark.sql(f"""
UPDATE {TABLE_NAME} 
SET load_dt = current_timestamp(), source_file = '1_main_chunk_1.csv' 
WHERE load_dt IS NULL
""")

# 4. Verification Check
total_count = spark.table(TABLE_NAME).count()
print(f" Table {TABLE_NAME} loaded. Total records: {total_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Table data 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.bronze.listings_csv_copyinto limit 5;
