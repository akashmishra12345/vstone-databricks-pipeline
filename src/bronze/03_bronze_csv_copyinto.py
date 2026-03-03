# Databricks notebook source
# MAGIC %md
# MAGIC #CSV Ingestion (Chunk 1) via COPY INTO

# COMMAND ----------

# Notebook: 03_bronze_copy_into_IDEMPOTENT
CATALOG = "vstone_catalog"
BRONZE = "bronze"
TABLE_NAME = f"{CATALOG}.{BRONZE}.listings_csv_copyinto"
FILE_PATH = f"/Volumes/{CATALOG}/raw/chunks/1_main_chunk_1.csv"

# 1. CREATE TABLE ONLY IF NOT EXISTS (Schema lock)
# Idempotency ke liye hum table drop nahi karenge, bas structure ensure karenge.
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
# COPY INTO automatically tracks which files are already loaded.
# Agar file path/content same hai, toh yeh re-run par 0 rows load karega.
spark.sql(f"""
COPY INTO {TABLE_NAME}
FROM '{FILE_PATH}'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false') 
COPY_OPTIONS ('mergeSchema' = 'true')
""")

# 3. SMART Audit Columns Update (ONLY FOR NEW DATA)
# Hum sirf un rows ko update karenge jahan load_dt null hai (yani jo abhi load hui hain).
spark.sql(f"""
UPDATE {TABLE_NAME} 
SET load_dt = current_timestamp(), source_file = '1_main_chunk_1.csv' 
WHERE load_dt IS NULL
""")

# 4. Verification Check
total_count = spark.table(TABLE_NAME).count()
print(f"✅ Table {TABLE_NAME} loaded. Total records: {total_count:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.bronze.listings_csv_copyinto limit 5;
