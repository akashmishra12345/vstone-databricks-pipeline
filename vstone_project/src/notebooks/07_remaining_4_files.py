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
        # Handling descriptions with nested quotes and newlines
        .option("multiLine", "true")      
        .option("escape", '"')            
        .option("quote", '"')             
        .option("inferSchema", "false")   
        .load(LANDING_PATH)
        .withColumn("load_dt", current_timestamp())
        .withColumn("source_file", lit("1_text.csv"))
    )

# 2. TABLE: listings_photo_bronze (PRO-FIX for Rescued Data)
@dlt.table(
    name="listings_photo_bronze", 
    table_properties=standard_props
)
def listings_photo_bronze():
    return (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "csv")
      .option("header", "true")
      # FIX: In unnamed columns (_c0) ko handle karne ke liye evolution mode enable karein
      .option("cloudFiles.inferColumnTypes", "true") 
      .option("cloudFiles.schemaEvolutionMode", "addNewColumns") 
      .option("pathGlobFilter", "1_photo.csv")
      .option("escape", '"')      
      .option("quote", '"')       
      .option("multiLine", "true") 
      .load(LANDING_PATH)
      .withColumn("load_dt", current_timestamp())
      .withColumn("source_file", lit("1_photo.csv")))

# 3. TABLE: car_catalog_bronze (Semicolon Separated - FIXED)
@dlt.table(name="car_catalog_bronze", table_properties=standard_props)
def car_catalog_bronze():
    return (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "csv")
      .option("sep", ";") # Semicolon handling for Russian catalogs
      .option("header", "true")
      .option("pathGlobFilter", "catalogs.csv")
      .load(LANDING_PATH)
      .withColumn("load_dt", current_timestamp())
      .withColumn("source_file", lit("catalogs.csv")))

# 4. TABLE: geo_locations_bronze (FIXED for CI/CD Consistency)
@dlt.table(name="geo_locations_bronze", table_properties=standard_props)
def geo_locations_bronze():
    return (spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "csv")
      .option("header", "true") 
      # FIX: Added whitespacing and sampling ratio to resolve 1,523 records failure
      .option("ignoreLeadingWhiteSpace", "true")
      .option("ignoreTrailingWhiteSpace", "true")
      .option("cloudFiles.inferColumnTypes", "true")
      .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
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
