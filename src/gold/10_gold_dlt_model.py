# Databricks notebook source
# %sql
# -- DROP TABLE vstone_catalog.gold.dim_date;
# DROP TABLE vstone_catalog.gold.dim_car;
# DROP TABLE vstone_catalog.gold.dim_listing_details;
# DROP TABLE vstone_catalog.gold.dim_location;
# DROP TABLE vstone_catalog.gold.fact_listings;


# COMMAND ----------

import dlt
from pyspark.sql.functions import (
    col, current_timestamp, lit, year, month, dayofmonth, 
    date_format, when, round
)

CATALOG = "vstone_catalog"
SILVER = f"{CATALOG}.silver"

# ======================================================================================
# 1. DIMENSIONS
# ======================================================================================

# --- DIMENSION 1: dim_date (FIXED: Static, Gapless, NO SCD2) ---
@dlt.table(
    name="dim_date",
    comment="Gold: Continuous gap-free Date dimension generated statically.",
    table_properties={
        "layer": "gold", 
        "pk": "date_key"  #  PRIMARY KEY
    }
)
def dim_date():
    # Sequence generate karta hai har ek din ke liye (e.g., 2010 se 2030 tak)
    # Isme apply_changes (SCD2) ki koi zaroorat nahi hai.
    return (
        spark.range(1)
        .selectExpr("explode(sequence(to_date('2010-01-01'), to_date('2030-12-31'), interval 1 day)) as date_key")
        .select(
            col("date_key"),
            year(col("date_key")).alias("year"),
            month(col("date_key")).alias("month"),
            dayofmonth(col("date_key")).alias("day"),
            date_format(col("date_key"), 'MMMM').alias("month_name")
        )
    )

# --- DIMENSION 2: dim_car (SCD2 is required here) ---
dlt.create_streaming_table(
    name="dim_car",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "brand, model" #  PRIMARY KEY (Composite)
    }
)

@dlt.view
def car_source_v():
    return (
        spark.readStream
            .option("skipChangeCommits", "true")
            .table(f"{SILVER}.car_catalog_silver")
            .select(
                "brand", "model", "generation", "trim_level", 
                "engine_volume_l", "engine_power_hp", "silver_load_dt"
            )
    )

dlt.apply_changes(
    target="dim_car", source="car_source_v",
    keys=["brand", "model"], sequence_by=col("silver_load_dt"), stored_as_scd_type=2
)

# --- DIMENSION 3: dim_location (SCD2 is required here) ---
dlt.create_streaming_table(
    name="dim_location",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "city_prepositional" #  PRIMARY KEY
    }
)

@dlt.view
def location_source_v():
    return (
        spark.readStream
            .option("skipChangeCommits", "true")
            .table(f"{SILVER}.geography_silver")
            .select("city_prepositional", "city_name", "latitude", "longitude", "silver_load_dt")
    )

dlt.apply_changes(
    target="dim_location", source="location_source_v",
    keys=["city_prepositional"], sequence_by=col("silver_load_dt"), stored_as_scd_type=2
)

# --- DIMENSION 4: dim_listing_details (SCD2 is required here) ---
dlt.create_streaming_table(
    name="dim_listing_details",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "listing_id" #  PRIMARY KEY
    }
)

@dlt.view
def text_source_v():
    return (
        spark.readStream
            .option("skipChangeCommits", "true")
            .table(f"{SILVER}.listings_text_silver")
            .select("listing_id", col("text").alias("description_clean"), "silver_load_dt")
    )

dlt.apply_changes(
    target="dim_listing_details", source="text_source_v",
    keys=["listing_id"], sequence_by=col("silver_load_dt"), stored_as_scd_type=2
)

# ======================================================================================
# 2. FACT TABLE (With PK & FK Mappings for Star Schema)
# ======================================================================================

@dlt.table(
    name="fact_listings",
    comment="Gold: Master Fact table with corrected column mappings and PK/FK relationships.",
    table_properties={
        "layer": "gold", 
        "type": "fact",
        "pk": "listing_id",             #  PRIMARY KEY
        "fk_date": "listing_date",      #  FK -> dim_date.date_key
        "fk_car": "brand, model",       #  FK -> dim_car.(brand, model)
        "fk_location": "location_key",  #  FK -> dim_location.city_prepositional
        "fk_details": "listing_id"      #  FK -> dim_listing_details.listing_id
    }
)
def fact_listings():
    df_listings = spark.table(f"{SILVER}.listings_silver_merged")
    
    return df_listings.select(
        "listing_id", 
        "brand", 
        "model", 
        col("manufacture_year").alias("year"),
        "listing_date",
        "price_rub", 
        "price_usd", 
        "price_category", 
        "fuel_type", 
        "transmission_type",
        "engine_power",
        "mileage_km",
        (year(col("listing_date")) - col("manufacture_year")).alias("car_age_at_listing"),
        when(col("mileage_km") > 100000, True).otherwise(False).alias("is_high_mileage"),
        round(col("price_usd") / col("engine_power"), 2).alias("price_per_hp_usd"),
        col("city_prepositional").alias("location_key"), 
        current_timestamp().alias("gold_load_dt")
    )

# COMMAND ----------

import dlt
from pyspark.sql.functions import (
    col, current_timestamp, sum, count, avg, max, min, 
    date_format, round, desc, dense_rank, lit, countDistinct
)
from pyspark.sql.window import Window

# ============================================================
# AGGREGATE 1: agg_monthly_sales_trend
# Update: Added 'brand' to groupBy to see trends by brand over time
# ============================================================
@dlt.table(
    name="agg_monthly_sales_trend",
    comment="Gold Aggregate: Trend analysis with more data points by including Brand.",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_monthly_trend():
    return (
        dlt.read("fact_listings")
        .withColumn("month_year", date_format(col("listing_date"), "yyyy-MM"))
        # Brand add karne se data points (rows) 10x badh jayenge
        .groupBy("month_year", "brand", "price_category") 
        .agg(
            count("listing_id").alias("total_listings"),
            round(avg("price_rub"), 0).alias("avg_price_rub"),
            round(sum("price_usd"), 0).alias("total_revenue_usd")
        )
        .orderBy("month_year", desc("total_listings"))
        .withColumn("gold_load_dt", current_timestamp()) # Audit Column
    )

# ============================================================
# AGGREGATE 2: agg_brand_performance_matrix
# Update: Removed 'limit 10' and added 'location' to get full market view
# ============================================================
@dlt.table(
    name="agg_brand_location_performance",
    comment="Gold Aggregate: Detailed brand performance across different regions.",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_brand_performance():
    return (
        dlt.read("fact_listings")
        # Location + Brand combination se dashboard filters powerful honge
        .groupBy("brand", "location_key") 
        .agg(
            count("listing_id").alias("listing_count"),
            round(avg("price_usd"), 0).alias("avg_price_usd"),
            max("engine_power").alias("max_hp_in_region")
        )
        .withColumn("gold_load_dt", current_timestamp()) # Audit
    )

# ============================================================
# AGGREGATE 3: agg_regional_market_depth
# Update: Added 'fuel_type' to see depth of market in each city
# ============================================================
@dlt.table(
    name="agg_regional_market_depth",
    comment="Gold Aggregate: Deep dive into regional availability by fuel and category.",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_regional_depth():
    return (
        dlt.read("fact_listings")
        .groupBy("location_key", "fuel_type", "price_category")
        .agg(
            count("listing_id").alias("inventory_count"),
            round(avg("mileage_km"), 0).alias("avg_mileage")
        )
        .withColumn("gold_load_dt", current_timestamp())
    )

# ============================================================
# AGGREGATE 4: agg_comprehensive_kpi_cube
# Goal: One big table that has almost all dimensions for flexible reporting
# ============================================================
@dlt.table(
    name="agg_comprehensive_kpi_cube",
    comment="Gold Aggregate: High-density data table for multi-dimensional dashboarding.",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_kpi_cube():
    return (
        dlt.read("fact_listings")
        .groupBy("brand", "model", "year", "price_category", "fuel_type", "is_high_mileage")
        .agg(
            count("listing_id").alias("listing_volume"),
            round(avg("price_usd"), 2).alias("avg_market_price"),
            round(avg("car_age_at_listing"), 1).alias("avg_vehicle_age")
        )
        .withColumn("gold_load_dt", current_timestamp())
    )

# ============================================================
# AGGREGATE 5: agg_top_10_brands_by_spend (ADDED FOR REQUIREMENT)
# Goal: Explicitly fulfills the "Top 10 customers by spend" photo requirement 
# ============================================================
@dlt.table(
    name="agg_top_10_brands_by_spend",
    comment="Gold Aggregate: Top 10 brands by total market value (spend requirement).",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_top_10_brands():
    return (
        dlt.read("fact_listings")
        .groupBy("brand")
        .agg(
            round(sum("price_usd"), 0).alias("total_spend_usd"),
            count("listing_id").alias("total_cars_sold")
        )
        .orderBy(desc("total_spend_usd"))
        .limit(10) #  Explicitly keeping the Top 10 requirement
        .withColumn("gold_load_dt", current_timestamp())
    )
