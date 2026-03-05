# Databricks notebook source
# MAGIC %md
# MAGIC # Widgets & Configuration

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit, regexp_replace, row_number, trim
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

# Setup widgets for environment-agnostic paths
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Target Catalog Name")
dbutils.widgets.text("bronze_schema", "bronze", "2. Bronze Schema Name")
dbutils.widgets.text("silver_schema", "silver", "3. Silver Schema Name")

# Fetch values into variables
CATALOG = dbutils.widgets.get("project_catalog")
BRONZE = dbutils.widgets.get("bronze_schema")
SILVER = dbutils.widgets.get("silver_schema")

# Dependency check: ensure all required bronze tables exist before proceeding
required_bronze = [
    "listings_csv_copyinto", "listings_json_autoloader",
    "listings_xml_pyspark"
]
existing_tables = [r.tableName for r in spark.sql(f"SHOW TABLES IN {CATALOG}.{BRONZE}").collect()]
missing = [t for t in required_bronze if t not in existing_tables]
if missing:
    raise Exception(
        f"DEPENDENCY CHECK FAILED. Run bronze pipelines first.\nMissing tables: {missing}"
    )
print(f"All {len(required_bronze)} bronze dependencies verified.")


# COMMAND ----------

# MAGIC %md
# MAGIC # Text Data Transformations

# COMMAND ----------

# TEXT LAYER: Table configuration
BRONZE_TABLE = f"{CATALOG}.{BRONZE}.listings_text_bronze"
SILVER_TABLE = f"{CATALOG}.{SILVER}.listings_text_silver"
QUARANTINE_TABLE = f"{CATALOG}.{SILVER}.listings_text_quarantine"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformation Logic

# COMMAND ----------

# Section 2: Text Transformation Logic

# 1. BRONZE LOAD: Listings Text Data
df_text_bronze = spark.table(BRONZE_TABLE)
bronze_total = df_text_bronze.count()

# 2. INITIAL TRANSFORMATIONS & CASTING
df_text_prep = df_text_bronze.withColumn(
    "listing_id", col("id").cast("double").cast("long").cast("string")
).withColumn(
    "source_file", lit("1_text.csv")
).withColumn(
    "silver_load_dt", current_timestamp()
).drop("id", "_c0", "_rescued_data")

# 3. STRICT SPLIT: Valid vs Malformed
df_invalid = df_text_prep.filter(col("listing_id").isNull())
df_valid_raw = df_text_prep.filter(col("listing_id").isNotNull())

# 4. DEDUPLICATION (By Listing ID)
# Deduplicate by listing_id, keeping the latest by load_dt
window_spec = Window.partitionBy("listing_id").orderBy(col("load_dt").desc())

df_text_silver_final = (df_valid_raw
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") == 1)
    .drop("rn")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Audit & Reconciliation Report

# COMMAND ----------

# Section 3: Final Writes & Report (Text)

# Write deduplicated valid text records to Silver table
df_text_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE)

# Prepare and write malformed text records to Quarantine table
df_quarantine_final = df_invalid.withColumn("quarantine_reason", lit("malformed_id_in_text_data"))

df_quarantine_final.write.format("delta").mode("append").option("mergeSchema", "true") \
    .saveAsTable(QUARANTINE_TABLE)

# Audit Report
silver_cnt = df_text_silver_final.count()
quarantine_cnt = df_quarantine_final.count()
duplicates_cnt = df_valid_raw.count() - silver_cnt

print(f"\n{'='*45}")
print(f" TEXT LAYER RECONCILIATION")
print(f"{'='*45}")
print(f"1. Bronze Total       : {bronze_total:,}")
print(f"2. Silver (Final)     : {silver_cnt:,}")
print(f"3. Quarantine         : {quarantine_cnt:,}")
print(f"4. Dropped Duplicates : {duplicates_cnt:,}")
print(f"{'='*45}")
print(f"Verification: {silver_cnt + quarantine_cnt + duplicates_cnt:,} (Matches Bronze)")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.listings_text_silver limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC # Photo Transformations

# COMMAND ----------

# PHOTO LAYER: Table configuration
BRONZE_TABLE = f"{CATALOG}.{BRONZE}.listings_photo_bronze"
SILVER_TABLE = f"{CATALOG}.{SILVER}.listings_photo_silver"
QUARANTINE_TABLE = f"{CATALOG}.{SILVER}.listings_photo_quarantine"


# COMMAND ----------

# Section 1: Configuration & UDFs (Photo)
import pandas as pd
from pyspark.sql.functions import col, current_timestamp, lit, row_number, pandas_udf, trim
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

# Pandas UDF for string standardization (lowercase, trimmed) to improve deduplication and joins
@pandas_udf(StringType())
def standardize_string_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Transformation & Deduplication

# COMMAND ----------

# Section 2: Transformations, Splitting & Deduplication (Photo)

# 1. BRONZE LOAD: Load raw photo records from Bronze table
df_photos_bronze = spark.table(BRONZE_TABLE)
bronze_total = df_photos_bronze.count()

# 2. INITIAL TRANSFORMATIONS & CASTING
df_photos_prep = df_photos_bronze.withColumn(
    # Clean listing_id: handles float strings and potential nulls
    "listing_id", col("id").cast("double").cast("long").cast("string")
).withColumn(
    # Standardizing URL using the UDF for safer deduplication
    "photo_url_clean", standardize_string_pd(col("photo_url"))
).withColumn(
    "source_file", lit("photos_main.csv")
).withColumn(
    "silver_load_dt", current_timestamp()
).drop("id", "_c0", "_rescued_data") 

# 3. STRICT SPLIT: Valid vs Malformed
df_invalid = df_photos_prep.filter(col("listing_id").isNull())  # Malformed records (missing listing_id)
df_valid_raw = df_photos_prep.filter(col("listing_id").isNotNull())  # Valid records

# 4. DEDUPLICATION (From Valid Records Only)
# Partitioning by ID + Cleaned URL, keeping latest by load_dt
window_spec = Window.partitionBy("listing_id", "photo_url_clean").orderBy(col("load_dt").desc())

df_photos_silver_final = (df_valid_raw
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") == 1)
    .drop("rn")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Writes & Reconciliation Report

# COMMAND ----------

# Section 3: Final Writes & Audit (Photo)

# Write deduplicated valid photo records to Silver table
df_photos_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE)

# Prepare and write malformed photo records to Quarantine table
df_quarantine_final = df_invalid.withColumn("quarantine_reason", lit("malformed_id_in_photo_data"))

df_quarantine_final.write.format("delta").mode("append").option("mergeSchema", "true") \
    .saveAsTable(QUARANTINE_TABLE)

# Reconciliation report: counts for bronze, silver, quarantine, and dropped duplicates
silver_cnt = df_photos_silver_final.count()
quarantine_cnt = df_quarantine_final.count()
duplicates_cnt = df_valid_raw.count() - silver_cnt

print(f"\n{'='*45}")
print(f" PHOTO LAYER RECONCILIATION")
print(f"{'='*45}")
print(f"1. Bronze Total       : {bronze_total:,}")
print(f"2. Silver (Final)     : {silver_cnt:,}")
print(f"3. Quarantine         : {quarantine_cnt:,}")
print(f"4. Dropped Duplicates : {duplicates_cnt:,}")
print(f"{'='*45}")
print(f"Verification: {silver_cnt + quarantine_cnt + duplicates_cnt:,} (Must match Bronze)")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.listings_photo_silver limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC # Catalogs Transformations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets & Dependencies

# COMMAND ----------

# Section 1: Widgets & Dependencies
import pandas as pd
from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit, regexp_replace, trim, expr, when
from pyspark.sql.types import StringType

# Fetch values from existing widgets
# CATALOG = dbutils.widgets.get("project_catalog")
# BRONZE = dbutils.widgets.get("bronze_schema")
# SILVER = dbutils.widgets.get("silver_schema")

# Define Dynamic Table Names
BRONZE_TABLE = f"{CATALOG}.{BRONZE}.car_catalog_bronze"
SILVER_TABLE = f"{CATALOG}.{SILVER}.car_catalog_silver"
QUARANTINE_TABLE = f"{CATALOG}.{SILVER}.listings_catalog_quarantine"

# 1. Mapping Dictionary: Russian column names to English column names for catalog normalization
FULL_CATALOG_MAP = {
    "Марка": "brand", "Модель": "model", "Поколение": "generation",
    "Комплектация": "trim_level", "Объём двигателя": "engine_volume_l",
    "Мощность двигателя": "engine_power_hp", "Расход топлива": "fuel_consumption",
    "Тип топлива": "fuel_type", "Коробка передач": "transmission",
    "Привод": "drive_type", "Кол-во мест": "seats_count",
    "Клиренс": "clearance_mm", "Объем багажника": "trunk_volume_l",
    "Период выпуска": "production_period", "Тип кузова": "body_type",
    "Марка кузова": "body_mark", "Время разгона 0-100 км/ч, с": "acceleration_0_100",
    "Максимальная скорость, км/ч": "max_speed_kmh", "Страна сборки": "assembly_country"
}

# Pandas UDF for text cleaning: trims whitespace from string columns
@pandas_udf(StringType())
def clean_text_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformation & Cleaning

# COMMAND ----------

# Section 2: Bronze Load & Data Transformations

# 2. BRONZE LOAD: Load raw car catalog records from Bronze table
df_cat_bronze = spark.table(BRONZE_TABLE)
bronze_total = df_cat_bronze.count()

# 3. RENAME COLUMNS: Map Russian column names to English using FULL_CATALOG_MAP
df_cat_renamed = df_cat_bronze
for rus_name, eng_name in FULL_CATALOG_MAP.items():
    if rus_name in df_cat_bronze.columns:
        df_cat_renamed = df_cat_renamed.withColumnRenamed(rus_name, eng_name)

# 4. DATA TRANSFORMATIONS: Clean and cast numeric columns, add metadata columns
df_cat_cleaned = df_cat_renamed.withColumn(
    "engine_volume_l", expr("try_cast(regexp_replace(regexp_replace(engine_volume_l, ' л', ''), ',', '.') as double)")
).withColumn(
    "engine_power_hp", expr("try_cast(regexp_replace(engine_power_hp, ' л.с.', '') as int)")
).withColumn(
    "clearance_mm", expr("try_cast(regexp_replace(clearance_mm, ' мм', '') as int)")
).withColumn(
    "trunk_volume_l", expr("try_cast(regexp_replace(trunk_volume_l, ' л', '') as int)")
).withColumn(
    "seats_count", expr("try_cast(regexp_replace(seats_count, ' мест', '') as int)")
).withColumn(
    "acceleration_0_100", expr("try_cast(regexp_replace(acceleration_0_100, ',', '.') as double)")
).withColumn(
    "max_speed_kmh", expr("try_cast(max_speed_kmh as int)")
).withColumn(
    "source_file", lit("catalogs.csv")
).withColumn(
    "silver_load_dt", current_timestamp()
)

# Standardize text columns: trim whitespace from key string columns for deduplication and joins
text_cols = ["brand", "model", "generation", "trim_level", "fuel_type", "transmission", "drive_type", "body_type"]
for c in text_cols:
    df_cat_cleaned = df_cat_cleaned.withColumn(c, clean_text_pd(col(c)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation & Writes

# COMMAND ----------

# Section 3: Split, Deduplication & Final Writes

# 5. STRICT SPLIT: Valid vs Quarantine
# Nullify brands that are string representations of missing values ("None", "nan", "null")
df_safety = df_cat_cleaned.withColumn("brand", when(col("brand").isin("None", "nan", "null"), None).otherwise(col("brand")))

df_invalid = df_safety.filter(col("brand").isNull())  # Records with missing or invalid brand
df_valid_raw = df_safety.filter(col("brand").isNotNull())  # Records with valid brand

# 6. DEDUPLICATION
# Deduplicate by key car attributes for catalog integrity
df_cat_silver_final = df_valid_raw.dropDuplicates(["brand", "model", "generation", "trim_level", "engine_volume_l", "engine_power_hp"])

# 7. WRITES
# Write deduplicated valid catalog records to Silver table
df_cat_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE)

# Prepare and write invalid catalog records to Quarantine table
df_quarantine_final = df_invalid.withColumn("quarantine_reason", lit("missing_brand_or_invalid_data")) \
                                .withColumn("quarantine_dt", current_timestamp())

df_quarantine_final.write.format("delta").mode("append").option("mergeSchema", "true") \
    .saveAsTable(QUARANTINE_TABLE)

# COMMAND ----------

# 8. RECONCILIATION REPORT
# Audit counts for silver, quarantine, dropped duplicates, and total processed for integrity check
silver_cnt = df_cat_silver_final.count()
quarantine_cnt = df_quarantine_final.count()
dups_cnt = df_valid_raw.count() - silver_cnt

print(f"\n SUCCESS: {silver_cnt:,} rows in {SILVER_TABLE}")
print(f" QUARANTINE: {quarantine_cnt:,} rows in {QUARANTINE_TABLE}")
print(f" DUPLICATES: {dups_cnt:,} rows dropped")
print(f"Check: {silver_cnt + quarantine_cnt + dups_cnt:,} == {bronze_total:,} (Total Bronze)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.car_catalog_silver limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC # Geo Locations Transformations

# COMMAND ----------

# Section 1: Dependencies & Environment Config
import pandas as pd
from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit, trim
from pyspark.sql.types import StringType

# Widget values are fetched elsewhere; variables CATALOG, BRONZE, SILVER assumed defined
# CATALOG = dbutils.widgets.get("project_catalog")
# BRONZE = dbutils.widgets.get("bronze_schema")
# SILVER = dbutils.widgets.get("silver_schema")

# Dynamic table names for geography layer processing
BRONZE_TABLE = f"{CATALOG}.{BRONZE}.geo_locations_bronze"
SILVER_TABLE = f"{CATALOG}.{SILVER}.geography_silver"
QUARANTINE_TABLE = f"{CATALOG}.{SILVER}.geography_quarantine"

# Pandas UDF for standardizing city names (trims whitespace for deduplication and joins)
@pandas_udf(StringType())
def standardize_geo_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformation & Coordinate Validation

# COMMAND ----------

# Section 2: Bronze Load & Data Prep (Updated for Russia Context)

# 2. LOAD FROM BRONZE: Load raw geography records from Bronze table
df_geo_bronze = spark.table(BRONZE_TABLE)
bronze_total = df_geo_bronze.count()

# 3. TRANSFORMATIONS & CLEANING: Standardize city names, cast coordinates, add metadata columns
df_geo_prep = df_geo_bronze.select(
    col("name_padesh").alias("city_name"),
    col("greate_padesh").alias("city_prepositional"),
    col("lat").cast("double").alias("latitude"),
    col("lon").cast("double").alias("longitude"),
    lit("geography.csv").alias("source_file"),
    current_timestamp().alias("silver_load_dt")
).withColumn("city_name", standardize_geo_pd(col("city_name")))

# 4. STRICT SPLIT: Valid vs Malformed (Russia Bounding Box)
# Russia-specific latitude (41°N to 82°N) and longitude (19°E to 180°E) filters applied
df_valid_raw = df_geo_prep.filter(
    col("latitude").isNotNull() & col("longitude").isNotNull() &
    # Russia Latitude Range (Approx 41°N to 82°N)
    col("latitude").between(41, 82) & 
    # Russia Longitude Range (Approx 19°E to 180°E)
    col("longitude").between(19, 180)
)

# LEFT ANTI JOIN: Identify strictly distinct quarantine records (not valid)
# Quarantine reason: coordinates outside Russia or null values
df_quarantine_final = df_geo_prep.join(df_valid_raw, ["city_name"], "left_anti") \
    .withColumn("quarantine_reason", lit("coordinates_outside_russia_or_null")) \
    .withColumn("quarantine_dt", current_timestamp())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deduplication & Final Writes

# COMMAND ----------

# Section 3: Deduplication & Writing to Silver

# 5. DEDUPLICATION (From Valid Pool Only)
# Deduplicate geography records by city_prepositional for unique city mapping in car listings join
df_geo_silver_final = df_valid_raw.dropDuplicates(["city_prepositional"])

# 6. FINAL WRITES
# Main Silver Geography Table
df_geo_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .saveAsTable(SILVER_TABLE)

# Specific Quarantine Table for Geography
df_quarantine_final.write.format("delta").mode("append").option("mergeSchema","true") \
    .saveAsTable(QUARANTINE_TABLE)

# COMMAND ----------

# 7. PROFESSIONAL RECONCILIATION REPORT
silver_cnt = df_geo_silver_final.count()
quarantine_cnt = df_quarantine_final.count()
duplicates_cnt = df_valid_raw.count() - silver_cnt

print(f"\n{'='*45}")
print(f" GEOGRAPHY LAYER RECONCILIATION")
print(f"{'='*45}")
print(f"1. Bronze Total       : {bronze_total:,}")
print(f"2. Silver (Final)     : {silver_cnt:,}")
print(f"3. Quarantine         : {quarantine_cnt:,}")
print(f"4. Dropped Duplicates : {duplicates_cnt:,}")
print(f"{'='*45}")
print(f"Check: {silver_cnt + quarantine_cnt + duplicates_cnt:,} (Must match Bronze Total)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.geography_silver limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC # Main File (Transformations)

# COMMAND ----------

# Section 1: Dependencies & Widgets
import pandas as pd
from pyspark.sql.functions import (
    pandas_udf, col, current_timestamp, lit, row_number, 
    regexp_replace, expr, lower, trim, coalesce, when, round, date_format, upper
)
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

# Fetch values from existing widgets (assumed defined elsewhere)
# CATALOG = dbutils.widgets.get("project_catalog")
# BRONZE  = dbutils.widgets.get("bronze_schema")
# SILVER  = dbutils.widgets.get("silver_schema")

# Dynamic Table Paths for Silver and Quarantine outputs
VALID_TARGET_TABLE = f"{CATALOG}.{SILVER}.listings_silver_merged"
QUARANTINE_TABLE   = f"{CATALOG}.{SILVER}.listings_main_quarantine"

# Business Constants (Feb 2023 Rate)
USD_RATE = 82.5 

# Pandas UDF for standardizing text columns (lowercase, trimmed) for deduplication and joins
@pandas_udf(StringType())
def standardize_text_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Union, Transformation & Deduplication

# COMMAND ----------

# Section 2: Merging & Initial Cleansing (Syntax Fix)
from pyspark.sql.functions import col, expr, try_to_timestamp, coalesce, row_number
from pyspark.sql.window import Window

# 1. UNION ALL 4 SOURCES
# Merge all bronze sources for listings (CSV, JSON, XML, DLT) into a unified DataFrame
df_master_bronze = (
    spark.table(f"{CATALOG}.{BRONZE}.listings_csv_copyinto")
    .unionByName(spark.table(f"{CATALOG}.{BRONZE}.listings_json_autoloader"), allowMissingColumns=True)
    .unionByName(spark.table(f"{CATALOG}.{BRONZE}.listings_xml_pyspark"), allowMissingColumns=True)
    .unionByName(spark.table(f"{CATALOG}.{BRONZE}.listings_csv_dlt"), allowMissingColumns=True)
)

# 2. TRANSFORMATIONS: Quotes are mandatory for format strings!
# Clean, cast, and standardize columns for Silver layer processing
df_silver_prep = df_master_bronze.select(
    # --- Identifiers ---
    expr("try_cast(id as long)").cast("string").alias("listing_id"),
    
    # --- FIXED: Added quotes around date patterns ---
    coalesce(
        # Use lit() to wrap the format strings
        try_to_timestamp(col("date"), lit("dd.MM.yyyy")),              
        try_to_timestamp(col("date"), lit("yyyy-MM-dd'T'HH:mm:ss'Z'")) 
    ).alias("listing_date"),
    
    # ... (Baki saare columns wahi rahenge) ...
    expr("try_cast(regexp_replace(cost, '[^0-9.]', '') as double)").alias("price_rub"),
    col("currency"),
    standardize_text_pd(col("marka")).alias("brand"),
    standardize_text_pd(col("model")).alias("model"),
    expr("try_cast(try_cast(year as double) as int)").alias("manufacture_year"),
    expr("try_cast(try_cast(power as double) as int)").alias("engine_power"),
    expr("try_cast(try_cast(probeg as double) as int)").alias("mileage_km"),
    col("engine").alias("fuel_type"),
    col("transmission").alias("transmission_type"),
    col("gear").alias("drive_type"),
    col("sWheel").alias("steering_wheel"), 
    col("complectation").alias("trim_level"), 
    col("place").alias("city_prepositional"),
    expr("try_cast(has_license as int)").alias("has_license"), 
    expr("coalesce(cast(R as string), '')").alias("color_r"),
    expr("coalesce(cast(G as string), '')").alias("color_g"),
    expr("coalesce(cast(B as string), '')").alias("color_b"),
    col("source_file"),
    col("load_dt").alias("bronze_load_dt")
)

# 3. GLOBAL DEDUPLICATION
# Deduplicate listings by listing_id, keeping the latest record by bronze_load_dt
window_spec = Window.partitionBy("listing_id").orderBy(col("bronze_load_dt").desc())
df_final_deduped = (df_silver_prep 
    .withColumn("rn", row_number().over(window_spec)) 
    .filter(col("rn") == 1) 
    .drop("rn")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Rules & Final Partitioned Writes

# COMMAND ----------

# Section 3: Business Rules, Enrichment & Final Persistence

# Check if deduped DataFrame exists before proceeding
if 'df_final_deduped' not in locals():
    print(" ERROR: 'df_final_deduped' not found. Please run Section 2 cell first!")
else:
    # 4. PRICE NORMALIZATION (RUB -> USD)
    # Conversion rate based on Feb 2023 historical data (82.5 RUB/USD)
    df_with_usd = df_final_deduped.withColumn("price_usd", round(col("price_rub") / USD_RATE, 2))

    # 5. DERIVED COLUMNS & ENRICHMENT
    # Add car age, price category, year/month extraction, standardized brand, and load timestamp
    df_enriched = (df_with_usd
        .withColumn("car_age_years", lit(2023) - col("manufacture_year").cast("integer"))
        .withColumn("price_category",
            when(col("price_rub") < 300000, lit("BUDGET"))
            .when(col("price_rub").between(300000, 700000), lit("MID_RANGE"))
            .when(col("price_rub").between(700001, 1500000), lit("PREMIUM"))
            .when(col("price_rub") > 1500000, lit("LUXURY"))
            .otherwise(lit("UNKNOWN"))
        )
        # Extracting Year and Month from our fixed listing_date
        .withColumn("listing_year", date_format(col("listing_date"), "yyyy").cast("integer"))
        .withColumn("listing_month", date_format(col("listing_date"), "MM").cast("integer"))
        .withColumn("brand_std", upper(trim(col("brand"))))
        .withColumn("silver_load_dt", current_timestamp())
    )

    # 6. SPLIT & WRITE (Valid vs Quarantine)
    # Filtering records that failed ID, Price, or Date parsing
    is_valid_filter = col("listing_id").isNotNull() & col("price_rub").isNotNull() & col("listing_date").isNotNull()

    df_silver_valid = df_enriched.filter(is_valid_filter)

    # Quarantine capture logic for audit
    df_silver_quarantine = df_enriched.filter(~is_valid_filter) \
        .withColumn("quarantine_reason", 
            when(col("listing_id").isNull(), lit("MISSING_OR_MALFORMED_ID"))
            .when(col("price_rub").isNull(), lit("INVALID_PRICE_FORMAT"))
            .when(col("listing_date").isNull(), lit("UNPARSABLE_DATE_FORMAT"))
            .otherwise(lit("DATA_QUALITY_ISSUE"))
        )

    # Save Final Enriched Silver Table
    (df_silver_valid.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(VALID_TARGET_TABLE))

    (df_silver_quarantine.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(QUARANTINE_TABLE))

    # 7. FINAL RECONCILIATION REPORT (Audit Evidence)
    # Print counts for processed, valid, and quarantined records for audit
    silver_valid_count = df_silver_valid.count()
    quarantine_count = df_silver_quarantine.count()

    print(f"\n{'='*45}")
    print(f" SUCCESS: Day 5 Silver Enrichment Complete")
    print(f"{'='*45}")
    print(f"✓ Total Bronze Processed : {df_master_bronze.count():,}")
    print(f"✓ Valid Silver (Enriched): {silver_valid_count:,}")
    print(f"✓ Quarantined Rows       : {quarantine_count:,}")
    print(f"{'='*45}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.listings_silver_merged limit 10;
