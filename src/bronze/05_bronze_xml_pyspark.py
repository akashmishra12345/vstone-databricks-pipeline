# Databricks notebook source
# MAGIC %md
# MAGIC #  Widgets & Configuration 

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit, col

# Setup widgets for dynamic execution
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Target Catalog Name")
dbutils.widgets.text("raw_schema", "raw", "2. Raw Schema Name")
dbutils.widgets.text("bronze_schema", "bronze", "3. Bronze Schema Name")
dbutils.widgets.text("chunks_volume", "chunks", "4. Chunks Volume Name")

# Fetch values into variables
CATALOG = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
BRONZE = dbutils.widgets.get("bronze_schema")
VOLUME = dbutils.widgets.get("chunks_volume")

# Fixed variable definitions to resolve NameError
SOURCE_FILE_NAME = "1_main_chunk_4.xml"
FILE_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{VOLUME}/{SOURCE_FILE_NAME}"
TABLE_NAME = f"{CATALOG}.{BRONZE}.listings_xml_pyspark"

# COMMAND ----------

# MAGIC %md
# MAGIC #XML Ingestion via PySpark (Chunk 4)

# COMMAND ----------

# 1. READ XML AS STRINGS
df_xml = (spark.read
  .format("xml")
  .option("rowTag", "record") 
  .option("inferSchema", "false") 
  .load(FILE_PATH) # Uses the variable from Section 1
  .withColumn("load_dt", current_timestamp())
  .withColumn("source_file", lit(SOURCE_FILE_NAME)))

# 2. IMPLEMENT IDEMPOTENCY: Check and Delete
if spark.catalog.tableExists(TABLE_NAME):
    print(f" Cleaning up existing records for {SOURCE_FILE_NAME} in {TABLE_NAME}...")
    spark.sql(f"DELETE FROM {TABLE_NAME} WHERE source_file = '{SOURCE_FILE_NAME}'")

# 3. WRITE TO BRONZE TABLE
(df_xml.write
  .mode("append")
  .option("mergeSchema", "true")
  .saveAsTable(TABLE_NAME))

# 4. VERIFICATION
count = spark.table(TABLE_NAME).filter(col("source_file") == SOURCE_FILE_NAME).count()
print(f" XML Ingestion Complete: {count:,} records from {SOURCE_FILE_NAME} are now in Bronze.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Verification

# COMMAND ----------

# Check if data exists in the XML table
df_check = spark.table("vstone_catalog.bronze.listings_xml_pyspark")
print(f"Total Rows Ingested from XML: {df_check.count():,}")
display(df_check.limit(5))
