# Databricks notebook source
# MAGIC %md
# MAGIC #XML Ingestion via PySpark (Chunk 4)

# COMMAND ----------

# Notebook: 05_bronze_xml_pyspark_IDEMPOTENT
from pyspark.sql.functions import current_timestamp, lit, col # <--- ADDED 'col' HERE

SOURCE_FILE_NAME = "1_main_chunk_4.xml"
FILE_PATH = f"/Volumes/vstone_catalog/raw/chunks/{SOURCE_FILE_NAME}"
TABLE_NAME = "vstone_catalog.bronze.listings_xml_pyspark"

# 1. READ XML AS STRINGS
df_xml = (spark.read
  .format("xml")
  .option("rowTag", "record") 
  .option("inferSchema", "false") 
  .load(FILE_PATH)
  .withColumn("load_dt", current_timestamp())
  .withColumn("source_file", lit(SOURCE_FILE_NAME)))

# 2. IMPLEMENT IDEMPOTENCY: Check and Delete
if spark.catalog.tableExists(TABLE_NAME):
    print(f"🔄 Cleaning up existing records for {SOURCE_FILE_NAME}...")
    # Using spark.sql to avoid any further NameErrors with filter/col
    spark.sql(f"DELETE FROM {TABLE_NAME} WHERE source_file = '{SOURCE_FILE_NAME}'")

# 3. WRITE TO BRONZE TABLE
(df_xml.write
  .mode("append")
  .option("mergeSchema", "true")
  .saveAsTable(TABLE_NAME))

# 4. VERIFICATION
# col is now defined for this check
count = spark.table(TABLE_NAME).filter(col("source_file") == SOURCE_FILE_NAME).count()
print(f"✅ XML Ingestion Complete: {count:,} records from {SOURCE_FILE_NAME} are now in Bronze.")

# COMMAND ----------

# Check if data exists in the XML table
df_check = spark.table("vstone_catalog.bronze.listings_xml_pyspark")
print(f"Total Rows Ingested from XML: {df_check.count():,}")
display(df_check.limit(5))
