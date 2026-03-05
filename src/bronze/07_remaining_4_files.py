# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

import dlt
from pyspark.sql.functions import current_timestamp, lit

# DLT mein widgets ke bajaye spark.conf.get use hota hai
CATALOG = spark.conf.get("pipeline.catalog", "vstone_catalog")
RAW_SCHEMA = spark.conf.get("pipeline.raw_schema", "raw")
VOLUME = spark.conf.get("pipeline.landing_volume", "landing")

# Dynamic Landing Path Construction based on your catalog explorer
LANDING_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{VOLUME}/"

# Generic Table Properties to handle Russian headers & special characters
standard_props = {
    # Metadata tag for Data Cataloging and Medallion Architecture governance
    "quality": "bronze",
    # Enables metadata-only schema evolution (allows dropping/renaming columns 
    # instantly without rewriting massive underlying Parquet data files)
    "delta.columnMapping.mode": "name", 
    #Enforces strict protocol versions required to safely support Column Mapping
    "delta.minReaderVersion": "2",
    "delta.minWriterVersion": "5"
}

# COMMAND ----------

# MAGIC %md
# MAGIC # DLT Tables Logic

# COMMAND ----------

# Section 2: DLT Table Definitions Logic

# 1. TABLE: listings_text_bronze (Fixed Parsing for Descriptions)
@dlt.table(name="listings_text_bronze", table_properties=standard_props)
def listings_text_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("pathGlobFilter", "1_text.csv")
        # --- CRITICAL PARSING OPTIONS ---
        .option("multiLine", "true")      
        .option("escape", '"')            
        .option("quote", '"')             
        .option("inferSchema", "false")   
        .load(LANDING_PATH)
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit("1_text.csv"))
    )

# 2. TABLE: listings_photo_bronze (String Ingestion)
@dlt.table(name="listings_photo_bronze", table_properties=standard_props)
def listings_photo_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("pathGlobFilter", "1_photo.csv")
        .option("cloudFiles.inferColumnTypes", "false") # Changed to false
        .load(LANDING_PATH)
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit("1_photo.csv"))
    )

# 3. TABLE: car_catalog_bronze (Semicolon Separated)
@dlt.table(name="car_catalog_bronze", table_properties=standard_props)
def car_catalog_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("sep", ";") 
        .option("header", "true")
        .option("pathGlobFilter", "catalogs.csv")
        .option("cloudFiles.inferColumnTypes", "false") # Changed to false
        .load(LANDING_PATH)
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit("catalogs.csv"))
    )

# 4. TABLE: geo_locations_bronze (Fixed for Row Index)
@dlt.table(name="geo_locations_bronze", table_properties=standard_props)
def geo_locations_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true") 
        .option("cloudFiles.inferColumnTypes", "false") # Changed to false for stability
        .option("pathGlobFilter", "final_geografic.csv")
        .load(LANDING_PATH)
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit("final_geografic.csv"))
    )

# COMMAND ----------

# %sql
# -- 1. Text Data
# SELECT * FROM `vstone_catalog`.`bronze`.`listings_text_bronze` LIMIT 10;

# COMMAND ----------

# %sql
# -- 2. Photo Data
# SELECT * FROM `vstone_catalog`.`bronze`.`listings_photo_bronze` LIMIT 10;

# COMMAND ----------

# %sql
# -- 3. Catalog Data (Most critical for backticks due to Russian headers)
# SELECT * FROM `vstone_catalog`.`bronze`.`car_catalog_bronze` LIMIT 10;

# COMMAND ----------

# %sql
# -- 4. Geo Data
# SELECT * FROM `vstone_catalog`.`bronze`.`geo_locations_bronze` LIMIT 10;
