# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Bronze XML Ingestion | PySpark Native
# MAGIC
# MAGIC Ingests `1_main_chunk_4.xml` into `vstone_catalog.bronze.listings_xml_pyspark`
# MAGIC using **PySpark native XML support** 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets & Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Catalog")
dbutils.widgets.text("raw_schema",      "raw",            "2. Raw Schema")
dbutils.widgets.text("bronze_schema",   "bronze",         "3. Bronze Schema")

CATALOG    = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
BRONZE     = dbutils.widgets.get("bronze_schema")

FILE_NAME    = "1_main_chunk_4.xml"
SOURCE_FILE  = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/chunks/{FILE_NAME}"
TARGET_TABLE = f"{CATALOG}.{BRONZE}.listings_xml_pyspark"

print(f"Source file  : {SOURCE_FILE}")
print(f"Target table : {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Schema DDL
# MAGIC
# MAGIC All 19 source columns defined as STRING — `inferSchema = false`.
# MAGIC `rowTag = "record"` matches the XML element written by `csv_to_xml.py`:
# MAGIC

# COMMAND ----------

BRONZE_SCHEMA = StructType([
    StructField("cost",          StringType(), True),
    StructField("currency",      StringType(), True),
    StructField("marka",         StringType(), True),
    StructField("model",         StringType(), True),
    StructField("year",          StringType(), True),
    StructField("has_license",   StringType(), True),
    StructField("place",         StringType(), True),
    StructField("date",          StringType(), True),
    StructField("id",            StringType(), True),
    StructField("engine",        StringType(), True),
    StructField("power",         StringType(), True),
    StructField("gear",          StringType(), True),
    StructField("probeg",        StringType(), True),
    StructField("sWheel",        StringType(), True),
    StructField("complectation", StringType(), True),
    StructField("transmission",  StringType(), True),
    StructField("R",             StringType(), True),
    StructField("G",             StringType(), True),
    StructField("B",             StringType(), True),
])

print(f"Schema defined — {len(BRONZE_SCHEMA.fields)} columns, inferSchema = false (all STRING)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read XML with PySpark

# COMMAND ----------

print(f"Reading XML with PySpark native reader...")

df_xml = (spark.read
    .format("xml")
    .option("rowTag",       "record")   # Each <record> element = one row
    .option("inferSchema",  "false")    # Explicit schema — no type inference
    .schema(BRONZE_SCHEMA)
    .load(SOURCE_FILE)
    .withColumn("load_dt",     F.current_timestamp())  # Mandatory audit: ingestion time
    .withColumn("source_file", F.lit(FILE_NAME))       # Mandatory audit: origin file
)

raw_count = df_xml.count()
print(f"Rows read from XML   : {raw_count:,}")
print(f"Columns              : {df_xml.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Null Guard
# MAGIC
# MAGIC Drop rows where `id` is null — these are malformed records or empty XML elements.

# COMMAND ----------

df_clean = df_xml.filter(F.col("id").isNotNull())
null_dropped = raw_count - df_clean.count()

print(f"Rows after null guard : {df_clean.count():,}")
print(f"Null id rows dropped  : {null_dropped:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Idempotent Write

# COMMAND ----------

if spark.catalog.tableExists(TARGET_TABLE):
    before = spark.table(TARGET_TABLE).count()
    spark.sql(f"DELETE FROM {TARGET_TABLE} WHERE source_file = '{FILE_NAME}'")
    after_delete = spark.table(TARGET_TABLE).count()
    print(f"Existing rows before delete : {before:,}")
    print(f"Rows deleted for {FILE_NAME} : {before - after_delete:,}")
else:
    print(f"Table does not exist yet — will be created on write.")

(df_clean.write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(TARGET_TABLE))

print(f"Write complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification & Null Audit

# COMMAND ----------

if not spark.catalog.tableExists(TARGET_TABLE):
    raise Exception(f"ERROR: Table not found after write — {TARGET_TABLE}")

df_bronze      = spark.table(TARGET_TABLE)
total          = df_bronze.count()
this_file_rows = df_bronze.filter(F.col("source_file") == FILE_NAME).count()
null_id        = df_bronze.filter(F.col("id").isNull()).count()
null_cost      = df_bronze.filter(F.col("cost").isNull()).count()
null_marka     = df_bronze.filter(F.col("marka").isNull()).count()
no_load_dt     = df_bronze.filter(F.col("load_dt").isNull()).count()
no_source_file = df_bronze.filter(F.col("source_file").isNull()).count()

print(f"\n{'='*60}")
print(f"  INGESTION SUMMARY")
print(f"{'='*60}")
print(f"  Table              : {TARGET_TABLE}")
print(f"  Total rows         : {total:,}")
print(f"  Rows from {FILE_NAME} : {this_file_rows:,}")
print(f"{'='*60}")
print(f"  NULL CHECKS")
print(f"  null id            : {null_id:,}      (must be 0)")
print(f"  null cost          : {null_cost:,}")
print(f"  null marka         : {null_marka:,}")
print(f"{'='*60}")
print(f"  AUDIT COLUMNS")
print(f"  missing load_dt    : {no_load_dt:,}    (must be 0)")
print(f"  missing source_file: {no_source_file:,} (must be 0)")
print(f"{'='*60}")
print(f"  Reader             : PySpark native XML (no Pandas)")
print(f"  Schema             : inferSchema = false (explicit DDL)")
print(f"  Idempotency        : DELETE + append on source_file")
print(f"{'='*60}")

all_pass = (null_id == 0 and no_load_dt == 0 and no_source_file == 0 and this_file_rows > 0)
if all_pass:
    print(f"  ALL CHECKS PASSED — {this_file_rows:,} rows ingested cleanly.")
else:
    print(f"  WARNING: One or more checks failed — investigate above.")

# COMMAND ----------

display(spark.table(TARGET_TABLE).limit(10))
