# Databricks notebook source
# MAGIC %md
# MAGIC # Text transformations

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit, regexp_replace, row_number, trim
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

# 1. PANDAS UDF: Standardization (Optimized with trim)
@pandas_udf(StringType())
def standardize_description_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# 2. BRONZE LOAD
df_text_bronze = spark.table("vstone_catalog.bronze.listings_text_bronze")
bronze_total = df_text_bronze.count()

# 3. TRANSFORMATIONS & DATA TYPING
# Professional practice: Handle nulls and casts early
df_prep = df_text_bronze.withColumn(
    "listing_id", col("id").cast("double").cast("long").cast("string")
).withColumn(
    "description_clean", standardize_description_pd(regexp_replace(col("text"), "<br/>", " "))
).withColumn(
    "source_file", lit("1_text.csv")
).withColumn(
    "silver_load_dt", current_timestamp()
)

# 4. SPLIT: VALID vs QUARANTINE (The "Filter First" Rule)
# Isse records dono tables mein repeat nahi honge
df_invalid = df_prep.filter(col("listing_id").isNull() | (trim(col("id")) == ""))
df_valid_raw = df_prep.filter(col("listing_id").isNotNull())

# 5. DEDUPLICATION (From Valid Records only)
# Keeping the latest record per listing_id
window_spec = Window.partitionBy("listing_id").orderBy(col("load_dt").desc())

df_silver_final = (df_valid_raw
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") == 1)
    .drop("rn", "id", "text") # Dropping raw columns only at the end
)

# 6. FINAL WRITES
# Main Silver Data
df_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("vstone_catalog.silver.listings_text_silver")

# Quarantine Data
df_quarantine_final = df_invalid.withColumn("quarantine_reason", lit("malformed_id_or_empty")) \
    .select("id", "listing_id", "source_file", "quarantine_reason", "silver_load_dt")

df_quarantine_final.write.format("delta").mode("append").option("mergeSchema", "true") \
    .saveAsTable("vstone_catalog.silver.listings_text_quarantine")

# 7. PROFESSIONAL RECONCILIATION REPORT (No more negative counts!)
silver_count = df_silver_final.count()
quarantine_count = df_quarantine_final.count()
# Total Duplicates = (Total Valid) - (Unique Silver)
duplicates_count = df_valid_raw.count() - silver_count

print(f"\n{'='*40}")
print(f"RECONCILIATION REPORT: TEXT DATA")
print(f"{'='*40}")
print(f"1. Bronze Total       : {bronze_total:,}")
print(f"2. Silver (Final)     : {silver_count:,}")
print(f"3. Quarantine         : {quarantine_count:,}")
print(f"4. Dropped Duplicates : {duplicates_count:,}")
print(f"{'='*40}")
print(f"Check: {silver_count + quarantine_count + duplicates_count:,} (Should match Bronze Total)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.listings_text_silver limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC # photo transformations

# COMMAND ----------

# src/notebooks/06_silver_cleansing.py
from pyspark.sql.functions import col, current_timestamp, lit, row_number
from pyspark.sql.window import Window

# 1. BRONZE LOAD: Listings Photo Data
df_photos_bronze = spark.table("vstone_catalog.bronze.listings_photo_bronze")
bronze_total = df_photos_bronze.count()

# 2. INITIAL TRANSFORMATIONS & CASTING
df_photos_prep = df_photos_bronze.withColumn(
    # Clean listing_id: handles float strings and potential nulls
    "listing_id", col("id").cast("double").cast("long").cast("string")
).withColumn(
    "source_file", lit("photos_main.csv")
).withColumn(
    "silver_load_dt", current_timestamp()
).drop("id", "_c0", "_rescued_data") # CLEANUP: Audit alerts fixed here

# 3. STRICT SPLIT: Valid vs Malformed (The "Anti-Overlap" Rule)
# Isse math hamesha positive aayega (Bronze = Silver + Quarantine + Dups)
df_invalid = df_photos_prep.filter(col("listing_id").isNull())
df_valid_raw = df_photos_prep.filter(col("listing_id").isNotNull())

# 4. DEDUPLICATION (From Valid Records Only)
# Partitioning by ID + URL ensures we keep all unique photos per car
window_spec = Window.partitionBy("listing_id", "photo_url").orderBy(col("load_dt").desc())

df_photos_silver_final = (df_valid_raw
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") == 1)
    .drop("rn")
)

# 5. FINAL WRITES
# Main Silver Photo Table
df_photos_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("vstone_catalog.silver.listings_photo_silver")

# Specific Quarantine Table for Photos
df_quarantine_final = df_invalid.withColumn("quarantine_reason", lit("malformed_id_in_photo_data"))

df_quarantine_final.write.format("delta").mode("append").option("mergeSchema", "true") \
    .saveAsTable("vstone_catalog.silver.listings_photo_quarantine")

# 6. PROFESSIONAL RECONCILIATION REPORT (The "No-Negative" Math)
silver_cnt = df_photos_silver_final.count()
quarantine_cnt = df_quarantine_final.count()
# Duplicates calculation based on valid pool only
duplicates_cnt = df_valid_raw.count() - silver_cnt

print(f"\n{'='*45}")
print(f"📊 PHOTO LAYER RECONCILIATION")
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
# MAGIC # catalogs

# COMMAND ----------

# src/notebooks/06_silver_cleansing.py
import pandas as pd
from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit, regexp_replace, trim, expr, when
from pyspark.sql.types import StringType

# 1. Mapping
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

@pandas_udf(StringType())
def clean_text_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

# 2. BRONZE LOAD
df_cat_bronze = spark.table("vstone_catalog.bronze.car_catalog_bronze")
bronze_total = df_cat_bronze.count()

# 3. RENAME COLUMNS
df_cat_renamed = df_cat_bronze
for rus_name, eng_name in FULL_CATALOG_MAP.items():
    if rus_name in df_cat_bronze.columns:
        df_cat_renamed = df_cat_renamed.withColumnRenamed(rus_name, eng_name)

# 4. DATA TRANSFORMATIONS (Sari original logic kept)
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

# Standardize text
text_cols = ["brand", "model", "generation", "trim_level", "fuel_type", "transmission", "drive_type", "body_type"]
for c in text_cols:
    df_cat_cleaned = df_cat_cleaned.withColumn(c, clean_text_pd(col(c)))

# 5. STRICT SPLIT: Valid vs Quarantine
# Sabse pehle "None" string ko actual NULL mein badalte hain for safety
df_safety = df_cat_cleaned.withColumn("brand", when(col("brand").isin("None", "nan", "null"), None).otherwise(col("brand")))

df_invalid = df_safety.filter(col("brand").isNull())
df_valid_raw = df_safety.filter(col("brand").isNotNull())

# 6. DEDUPLICATION (Using your specific keys)
df_cat_silver_final = df_valid_raw.dropDuplicates(["brand", "model", "generation", "trim_level", "engine_volume_l", "engine_power_hp"])

# 7. WRITES
df_cat_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("vstone_catalog.silver.car_catalog_silver")

df_quarantine_final = df_invalid.withColumn("quarantine_reason", lit("missing_brand_or_invalid_data")) \
                                .withColumn("quarantine_dt", current_timestamp())

df_quarantine_final.write.format("delta").mode("append").option("mergeSchema", "true") \
    .saveAsTable("vstone_catalog.silver.listings_catalog_quarantine")

# 8. RECONCILIATION REPORT (The Pro Way)
silver_cnt = df_cat_silver_final.count()
quarantine_cnt = df_quarantine_final.count()
dups_cnt = df_valid_raw.count() - silver_cnt

print(f"\n✅ SUCCESS: {silver_cnt:,} rows in car_catalog_silver")
print(f"🛡️ QUARANTINE: {quarantine_cnt:,} rows")
print(f"🗑️ DUPLICATES: {dups_cnt:,} rows")
print(f"Check: {silver_cnt + quarantine_cnt + dups_cnt} == {bronze_total} (Total Bronze)")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.car_catalog_silver limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC # Geo locations 

# COMMAND ----------

# src/notebooks/06_silver_cleansing_geo.py
import pandas as pd
from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit, trim
from pyspark.sql.types import StringType

# Setup Variables
CATALOG = "vstone_catalog"
BRONZE = "bronze"
SILVER = "silver"

# 1. PANDAS UDF: Standardize Location Names (Clean whitespace & Case)
@pandas_udf(StringType())
def standardize_geo_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

# 2. LOAD FROM BRONZE
df_geo_bronze = spark.table(f"{CATALOG}.{BRONZE}.geo_locations_bronze")
bronze_total = df_geo_bronze.count()

# 3. TRANSFORMATIONS & CLEANING
df_geo_prep = df_geo_bronze.select(
    col("name_padesh").alias("city_name"),
    col("greate_padesh").alias("city_prepositional"),
    col("lat").cast("double").alias("latitude"),
    col("lon").cast("double").alias("longitude"),
    lit("geography.csv").alias("source_file"),
    current_timestamp().alias("silver_load_dt")
).withColumn("city_name", standardize_geo_pd(col("city_name")))

# 4. STRICT SPLIT: Valid vs Malformed (No Overlap)
# Coordinates range check is professional standard for Geo-data
df_valid_raw = df_geo_prep.filter(
    col("latitude").isNotNull() & col("longitude").isNotNull() &
    col("latitude").between(-90, 90) &
    col("longitude").between(-180, 180)
)

# LEFT ANTI JOIN ensure strictly distinct quarantine records
df_quarantine_final = df_geo_prep.join(df_valid_raw, ["city_name"], "left_anti") \
    .withColumn("quarantine_reason", lit("invalid_coordinates_range_or_null")) \
    .withColumn("quarantine_dt", current_timestamp())

# 5. DEDUPLICATION (From Valid Pool Only)
# Car listings join ke liye prepositional name unique hona chahiye
df_geo_silver_final = df_valid_raw.dropDuplicates(["city_prepositional"])

# 6. FINAL WRITES
# Main Silver Geography Table
df_geo_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema","true") \
    .saveAsTable(f"{CATALOG}.{SILVER}.geography_silver")

# Specific Quarantine Table for Geography
df_quarantine_final.write.format("delta").mode("append").option("mergeSchema","true") \
    .saveAsTable(f"{CATALOG}.{SILVER}.geography_quarantine")

# 7. PROFESSIONAL RECONCILIATION REPORT (The Math Fix)
silver_cnt = df_geo_silver_final.count()
quarantine_cnt = df_quarantine_final.count()
# Total Duplicates = (Total Valid) - (Unique Silver)
duplicates_cnt = df_valid_raw.count() - silver_cnt

print(f"\n{'='*45}")
print(f"🌍 GEOGRAPHY LAYER RECONCILIATION")
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
# MAGIC # main file 

# COMMAND ----------

# src/notebooks/06_silver_cleansing.py
import pandas as pd
from pyspark.sql.functions import (
    pandas_udf, col, current_timestamp, lit, row_number, 
    regexp_replace, expr, lower, trim, coalesce, when
)
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

# =============================================================
# 1. UNION ALL 4 SOURCES (Merging Main Chunks)
# =============================================================
df_master_bronze = (
    spark.table("vstone_catalog.bronze.listings_csv_copyinto")
    .unionByName(spark.table("vstone_catalog.bronze.listings_json_autoloader"), allowMissingColumns=True)
    .unionByName(spark.table("vstone_catalog.bronze.listings_xml_pyspark"), allowMissingColumns=True)
    .unionByName(spark.table("vstone_catalog.bronze.listings_csv_dlt"), allowMissingColumns=True)
)

# =============================================================
# 2. PANDAS UDF for Text Standardization
# =============================================================
@pandas_udf(StringType())
def standardize_text_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# =============================================================
# 3. TRANSFORMATIONS: Preserving Original Date Format
# =============================================================
df_silver_prep = df_master_bronze.select(
    # --- Identifiers ---
    expr("try_cast(id as long)").cast("string").alias("listing_id"),
    
    # --- Date Parsing (Preserved as String to avoid format changes) ---
    col("date").cast("string").alias("listing_date"),
    
    # --- Financials ---
    expr("try_cast(regexp_replace(cost, '[^0-9.]', '') as double)").alias("price_rub"),
    col("currency"),
    
    # --- Core Attributes (Standardized via UDF) ---
    standardize_text_pd(col("marka")).alias("brand"),
    standardize_text_pd(col("model")).alias("model"),
    expr("try_cast(try_cast(year as double) as int)").alias("year"),
    
    # --- Technical Specs ---
    expr("try_cast(try_cast(power as double) as int)").alias("engine_power"),
    expr("try_cast(try_cast(probeg as double) as int)").alias("mileage_km"),
    col("engine").alias("fuel_type"),
    col("transmission").alias("transmission_type"),
    col("gear").alias("drive_type"),
    col("sWheel").alias("steering_wheel"), 
    col("complectation").alias("trim_level"), 
    
    # --- Location & Flags ---
    col("place").alias("city_prepositional"),
    expr("try_cast(has_license as int)").alias("has_license"), 
    
    # --- Visuals (RGB Colors) ---
    expr("coalesce(cast(R as string), '')").alias("color_r"),
    expr("coalesce(cast(G as string), '')").alias("color_g"),
    expr("coalesce(cast(B as string), '')").alias("color_b"),
    
    # --- Metadata ---
    current_timestamp().alias("silver_load_dt"),
    col("source_file")
)

# =============================================================
# 4. GLOBAL DEDUPLICATION (By listing_id)
# =============================================================
window_spec = Window.partitionBy("listing_id").orderBy(col("silver_load_dt").desc())

df_final_deduped = df_silver_prep \
    .withColumn("rn", row_number().over(window_spec)) \
    .filter(col("rn") == 1) \
    .drop("rn")

# =============================================================
# 5. SPLIT: Valid vs Quarantine (Audit Checks)
# =============================================================
is_valid_filter = col("listing_id").isNotNull() & col("price_rub").isNotNull()

df_silver_valid = df_final_deduped.filter(is_valid_filter)

df_silver_quarantine = df_final_deduped.filter(~is_valid_filter) \
    .withColumn("quarantine_reason", 
        when(col("listing_id").isNull(), lit("MISSING_ID"))
        .when(col("price_rub").isNull(), lit("INVALID_PRICE"))
        .otherwise(lit("DATA_QUALITY_ISSUE"))
    )

# =============================================================
# 6. WRITE TO DELTA (Final Tables)
# =============================================================
# Writing Valid Data to Silver
df_silver_valid.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("vstone_catalog.silver.listings_silver_merged")

# Writing Bad Data to Quarantine for Day 5 Audit
df_silver_quarantine.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("vstone_catalog.silver.listings_main_quarantine")

# =============================================================
# 7. RECONCILIATION REPORT (Evidence for Day 5)
# =============================================================
print(f"✅ Success: Day 5 Silver Processing Complete.")
print(f"Rows in Silver: {df_silver_valid.count():,}")
print(f"Rows in Quarantine: {df_silver_quarantine.count():,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.listings_silver_merged limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Final Audit with Explicit Duplicate Tracking
# MAGIC WITH bronze_union AS (
# MAGIC     SELECT id FROM vstone_catalog.bronze.listings_csv_copyinto
# MAGIC     UNION ALL 
# MAGIC     SELECT id FROM vstone_catalog.bronze.listings_json_autoloader
# MAGIC     UNION ALL 
# MAGIC     SELECT id FROM vstone_catalog.bronze.listings_xml_pyspark
# MAGIC     UNION ALL 
# MAGIC     SELECT id FROM vstone_catalog.bronze.listings_csv_dlt
# MAGIC ),
# MAGIC stats AS (
# MAGIC     SELECT 
# MAGIC         (SELECT COUNT(*) FROM bronze_union) as bronze_total,
# MAGIC         (SELECT COUNT(*) FROM vstone_catalog.silver.listings_silver_merged) as silver_total,
# MAGIC         (SELECT COUNT(*) FROM vstone_catalog.silver.listings_main_quarantine) as quarantine_total
# MAGIC )
# MAGIC SELECT 'Bronze Master (Raw)' as category, bronze_total as count FROM stats
# MAGIC UNION ALL
# MAGIC SELECT 'Silver Final (Unique)' as category, silver_total as count FROM stats
# MAGIC UNION ALL
# MAGIC SELECT 'Quarantine (Malformed)' as category, quarantine_total as count FROM stats
# MAGIC UNION ALL
# MAGIC -- Yeh line aapko batayegi ki kitne records duplicate hone ki wajah se drop huye
# MAGIC SELECT 'Dropped Duplicates' as category, 
# MAGIC        (bronze_total - silver_total - quarantine_total) as count 
# MAGIC FROM stats;
