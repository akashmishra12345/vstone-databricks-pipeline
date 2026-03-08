# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Bronze DLT | Remaining 4 Landing Files
# MAGIC
# MAGIC | Table | File | Columns |
# MAGIC |---|---|---|
# MAGIC | `listings_text` | `1_text.csv` | `id`, `text` |
# MAGIC | `listings_photo` | `1_photo.csv` | `_c0`, `photo_url`, `id` |
# MAGIC | `car_catalog` | `catalogs.csv` | 19 Cyrillic columns, `;` delimiter |
# MAGIC | `geo_locations` | `final_geografic.csv` | `_c0`, `name_padesh`, `greate_padesh`, `lat`, `lon` |
# MAGIC
# MAGIC **Schema corrections vs previous version:**
# MAGIC - `listings_text`: 2 cols (`id`, `text`)
# MAGIC - `listings_photo`:  `_c0`
# MAGIC - `car_catalog`: 19 Cyrillic cols
# MAGIC - `geo_locations`: `_c0`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

CATALOG = spark.conf.get("pipeline.catalog",        "vstone_catalog")
RAW_SCH = spark.conf.get("pipeline.raw_schema",     "raw")
VOLUME  = spark.conf.get("pipeline.landing_volume", "landing")

LANDING_PATH = f"/Volumes/{CATALOG}/{RAW_SCH}/{VOLUME}/"

BRONZE_PROPS = {
    "quality"                   : "bronze",
    "delta.enableChangeDataFeed": "true",
    "delta.columnMapping.mode"  : "name",   # required for Cyrillic column names
    "delta.minReaderVersion"    : "2",
    "delta.minWriterVersion"    : "5",
    "pipelines.reset.allowed"   : "true",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schemas
# MAGIC Column names verified by running `spark.read.csv().columns` against each actual file.

# COMMAND ----------

# ── listings_text: exactly 2 columns
SCHEMA_TEXT = StructType([
    StructField("id",   StringType(), True),
    StructField("text", StringType(), True),   # Russian car descriptions, multiLine
])

# ── listings_photo: 3 columns
# Spark names the unnamed pandas index column '_c0' when header=true
SCHEMA_PHOTO = StructType([
    StructField("_c0",       StringType(), True),   # unnamed pandas index, Spark calls it _c0
    StructField("photo_url", StringType(), True),
    StructField("id",        StringType(), True),
])

# ── car_catalog: 19 Cyrillic columns, semicolon-delimited

SCHEMA_CATALOG = StructType([
    StructField("Марка",                       StringType(), True),
    StructField("Модель",                      StringType(), True),
    StructField("Поколение",                   StringType(), True),
    StructField("Комплектация",                StringType(), True),
    StructField("Объём двигателя",             StringType(), True),
    StructField("Мощность двигателя",          StringType(), True),
    StructField("Расход топлива",              StringType(), True),
    StructField("Тип топлива",                 StringType(), True),
    StructField("Коробка передач",             StringType(), True),
    StructField("Привод",                      StringType(), True),
    StructField("Кол-во мест",                 StringType(), True),
    StructField("Клиренс",                     StringType(), True),
    StructField("Объем багажника",             StringType(), True),
    StructField("Период выпуска",              StringType(), True),
    StructField("Тип кузова",                  StringType(), True),
    StructField("Марка кузова",                StringType(), True),
    StructField("Время разгона 0-100 км/ч, с", StringType(), True),
    StructField("Максимальная скорость, км/ч", StringType(), True),
    StructField("Страна сборки",               StringType(), True),
])

# ── geo_locations: 5 columns
# Spark names the unnamed pandas index column '_c0' when header=true
SCHEMA_GEO = StructType([
    StructField("_c0",          StringType(), True),   # unnamed pandas index
    StructField("name_padesh",  StringType(), True),
    StructField("greate_padesh",StringType(), True),
    StructField("lat",          StringType(), True),   # cast to DOUBLE in Silver
    StructField("lon",          StringType(), True),   # cast to DOUBLE in Silver
])

print("All 4 schemas defined — column names verified against actual file headers.")
print(f"  listings_text   : {[f.name for f in SCHEMA_TEXT.fields]}")
print(f"  listings_photo  : {[f.name for f in SCHEMA_PHOTO.fields]}")
print(f"  car_catalog     : {len(SCHEMA_CATALOG.fields)} columns (Cyrillic)")
print(f"  geo_locations   : {[f.name for f in SCHEMA_GEO.fields]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Table — `listings_text`

# COMMAND ----------

@dlt.table(
    name             = "listings_text",
    comment          = "Bronze — Russian car descriptions from 1_text.csv",
    table_properties = BRONZE_PROPS
)
@dlt.expect("valid_id", "id IS NOT NULL")
def listings_text():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",              "csv")
        .option("cloudFiles.schemaEvolutionMode", "none")
        .option("cloudFiles.inferColumnTypes",    "false")
        .option("header",                         "true")
        .option("multiLine",                      "true")   # descriptions span multiple lines
        .option("escape",                         '"')
        .option("quote",                          '"')
        .option("encoding",                       "UTF-8")
        .option("pathGlobFilter",                 "1_text.csv")
        .schema(SCHEMA_TEXT)
        .load(LANDING_PATH)
        .withColumn("load_dt",     F.current_timestamp())
        .withColumn("source_file", F.lit("1_text.csv"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Table — `listings_photo`
# MAGIC
# MAGIC `_c0` is the correct Spark name for the unnamed pandas index column.

# COMMAND ----------

@dlt.table(
    name             = "listings_photo",
    comment          = "Bronze — photo URLs from 1_photo.csv",
    table_properties = BRONZE_PROPS
)
@dlt.expect("valid_id", "id IS NOT NULL")
def listings_photo():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",              "csv")
        .option("cloudFiles.schemaEvolutionMode", "none")
        .option("cloudFiles.inferColumnTypes",    "false")
        .option("header",                         "true")
        .option("encoding",                       "UTF-8")
        .option("pathGlobFilter",                 "1_photo.csv")
        .schema(SCHEMA_PHOTO)
        .load(LANDING_PATH)
        .withColumn("load_dt",     F.current_timestamp())
        .withColumn("source_file", F.lit("1_photo.csv"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Table — `car_catalog`
# MAGIC
# MAGIC 19 Cyrillic column headers. `delta.columnMapping.mode=name` in table properties handles these safely.

# COMMAND ----------

@dlt.table(
    name             = "car_catalog",
    comment          = "Bronze — car make/model catalog from catalogs.csv (Cyrillic headers, semicolon-delimited)",
    table_properties = BRONZE_PROPS
)
def car_catalog():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",              "csv")
        .option("cloudFiles.schemaEvolutionMode", "none")
        .option("cloudFiles.inferColumnTypes",    "false")
        .option("header",                         "true")
        .option("sep",                            ";")
        .option("encoding",                       "UTF-8")
        .option("pathGlobFilter",                 "catalogs.csv")
        .schema(SCHEMA_CATALOG)
        .load(LANDING_PATH)
        .withColumn("load_dt",     F.current_timestamp())
        .withColumn("source_file", F.lit("catalogs.csv"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Table — `geo_locations`
# MAGIC
# MAGIC `lat` and `lon` are kept as STRING in Bronze — cast to DOUBLE in Silver.

# COMMAND ----------

@dlt.table(
    name             = "geo_locations",
    comment          = "Bronze — city coordinates from final_geografic.csv",
    table_properties = BRONZE_PROPS
)
def geo_locations():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format",              "csv")
        .option("cloudFiles.schemaEvolutionMode", "none")
        .option("cloudFiles.inferColumnTypes",    "false")
        .option("header",                         "true")
        .option("encoding",                       "UTF-8")
        .option("pathGlobFilter",                 "final_geografic.csv")
        .schema(SCHEMA_GEO)
        .load(LANDING_PATH)
        .withColumn("load_dt",     F.current_timestamp())
        .withColumn("source_file", F.lit("final_geografic.csv"))
    )
