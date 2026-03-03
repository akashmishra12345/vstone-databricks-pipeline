# Databricks notebook source
import dlt
from pyspark.sql.functions import current_timestamp, lit

# Configuration: Path to your landing zone
LANDING_PATH = "/Volumes/vstone_catalog/raw/landing/"

# Generic Table Properties to handle Russian headers & special characters
standard_props = {
    "quality": "bronze",
    "delta.columnMapping.mode": "name", 
    "delta.minReaderVersion": "2",
    "delta.minWriterVersion": "5"
}

# 1. TABLE: listings_text_bronze (Fixed Parsing for Descriptions)
@dlt.table(name="listings_text_bronze", table_properties=standard_props)
def listings_text_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("pathGlobFilter", "1_text.csv")
        
        # --- CRITICAL PARSING OPTIONS ---
        .option("multiLine", "true")      # Taaki description ki newlines handle ho sakein
        .option("escape", '"')            # Taaki description ke andar ke quotes handle hon
        .option("quote", '"')             # Taaki commas (,) text ka part maane jayein
        .option("inferSchema", "false")   # Bronze mein hamesha string format best hai
        
        .load(LANDING_PATH)
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit("1_text.csv"))
    )
# 2. TABLE: listings_photo_bronze (Comma Separated)
@dlt.table(name="listings_photo_bronze", table_properties=standard_props)
def listings_photo_bronze():
    return (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "csv")
      .option("header", "true")
      .option("pathGlobFilter", "1_photo.csv")
      .load(LANDING_PATH)
      .withColumn("load_dt", current_timestamp())
      .withColumn("source_file", lit("1_photo.csv")))

# 3. TABLE: car_catalog_bronze (Semicolon Separated - FIXED)
@dlt.table(name="car_catalog_bronze", table_properties=standard_props)
def car_catalog_bronze():
    return (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "csv")
      .option("sep", ";") # Handles semicolon delimiter
      .option("header", "true")
      .option("pathGlobFilter", "catalogs.csv")
      .load(LANDING_PATH)
      .withColumn("load_dt", current_timestamp())
      .withColumn("source_file", lit("catalogs.csv")))

# 4. TABLE: geo_locations_bronze (FIXED for Row Index & Unnamed first comma)
@dlt.table(name="geo_locations_bronze", table_properties=standard_props)
def geo_locations_bronze():
    return (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "csv")
      .option("header", "true") 
      # Hum inferColumnTypes use karenge taaki Spark automatic '_c0' ko column maan le
      .option("cloudFiles.inferColumnTypes", "true")
      .option("pathGlobFilter", "final_geografic.csv")
      .load(LANDING_PATH)
      .withColumn("load_dt", current_timestamp())
      .withColumn("source_file", lit("final_geografic.csv")))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 1. Text Data
# MAGIC SELECT * FROM `vstone_catalog`.`bronze`.`listings_text_bronze` LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2. Photo Data
# MAGIC SELECT * FROM `vstone_catalog`.`bronze`.`listings_photo_bronze` LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 3. Catalog Data (Most critical for backticks due to Russian headers)
# MAGIC SELECT * FROM `vstone_catalog`.`bronze`.`car_catalog_bronze` LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 4. Geo Data
# MAGIC SELECT * FROM `vstone_catalog`.`bronze`.`geo_locations_bronze` LIMIT 10;
