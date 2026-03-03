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
    date_format, lead, desc, isnull, when, round
)
from pyspark.sql.window import Window

CATALOG = "vstone_catalog"
SILVER = f"{CATALOG}.silver"

# ======================================================================================
# 1. SCD TYPE 2 DIMENSIONS (With Explicit Primary Keys & Expectations)
# ======================================================================================

# --- DIMENSION 1: dim_date ---
dlt.create_streaming_table(
    name="dim_date",
    comment="Gold: Date dimension with SCD2. Key links to fact_listings.listing_date.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "date_key",
        "pipelines.autoOptimize.zOrderCols": "year,month"
    }
)

@dlt.view
@dlt.expect_or_drop("valid_date_format", "date_key IS NOT NULL")
def date_source_v():
    # FIX: Added 'skipChangeCommits' to resolve DELTA_SOURCE_TABLE_IGNORE_CHANGES error
    return (
        spark.readStream
            .option("skipChangeCommits", "true") 
            .table(f"{SILVER}.listings_silver_merged")
            .select(
                col("listing_date").alias("date_key"),
                year(col("listing_date")).alias("year"),
                month(col("listing_date")).alias("month"),
                dayofmonth(col("listing_date")).alias("day"),
                date_format(col("listing_date"), 'MMMM').alias("month_name"),
                current_timestamp().alias("load_dt")
            ).distinct()
    )

dlt.apply_changes(
    target="dim_date",
    source="date_source_v",
    keys=["date_key"],
    sequence_by=col("load_dt"),
    stored_as_scd_type=2
)

# --- DIMENSION 2: dim_car ---
dlt.create_streaming_table(
    name="dim_car",
    comment="Gold: Car specs with SCD2. Composite Key links to brand/model in Fact.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "brand, model"
    }
)

@dlt.view
@dlt.expect_or_drop("valid_car_identity", "brand IS NOT NULL AND model IS NOT NULL")
def car_source_v():
    # FIX: Added 'skipChangeCommits' for stability
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
    target="dim_car",
    source="car_source_v",
    keys=["brand", "model"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=2
)

# --- DIMENSION 3: dim_location ---
dlt.create_streaming_table(
    name="dim_location",
    comment="Gold: Geography dimension with SCD2. Key links to fact_listings.location_key.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "city_prepositional"
    }
)

@dlt.view
@dlt.expect("valid_coordinates", "latitude IS NOT NULL AND longitude IS NOT NULL")
def location_source_v():
    # FIX: Added 'skipChangeCommits' to handle geography_silver updates
    return (
        spark.readStream
            .option("skipChangeCommits", "true")
            .table(f"{SILVER}.geography_silver")
            .select(
                "city_prepositional", "city_name", "latitude", "longitude", "silver_load_dt"
            )
    )

dlt.apply_changes(
    target="dim_location",
    source="location_source_v",
    keys=["city_prepositional"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=2
)

# --- DIMENSION 4: dim_listing_details ---
dlt.create_streaming_table(
    name="dim_listing_details",
    comment="Gold: Text descriptions with SCD2. Key links to fact_listings.listing_id.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "listing_id"
    }
)

@dlt.view
@dlt.expect_or_drop("meaningful_description", "LENGTH(description_clean) > 5")
def text_source_v():
    # FIX: Added 'skipChangeCommits' for stability
    return (
        spark.readStream
            .option("skipChangeCommits", "true")
            .table(f"{SILVER}.listings_text_silver")
            .select(
                "listing_id", "description_clean", "silver_load_dt"
            )
    )

dlt.apply_changes(
    target="dim_listing_details",
    source="text_source_v",
    keys=["listing_id"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=2
)

# ======================================================================================
# 2. FACT TABLE (Detailed Metrics, PK-FK & High-Level Expectations)
# ======================================================================================

@dlt.table(
    name="fact_listings",
    comment="Gold: Master Fact table with advanced metrics and PK/FK relationships.",
    table_properties={
        "layer": "gold", 
        "type": "fact",
        "pk": "listing_id",
        "fk_location": "location_key",
        "fk_car": "brand, model",
        "fk_date": "listing_date",
        "fk_details": "listing_id"
    }
)
@dlt.expect_or_fail("critical_id_check", "listing_id IS NOT NULL")
@dlt.expect_or_drop("positive_price_check", "price_rub > 0")
@dlt.expect("reasonable_mileage", "mileage_km BETWEEN 0 AND 1000000")
@dlt.expect("valid_manufacture_year", "year BETWEEN 1900 AND 2025")
def fact_listings():
    # Final Fact table joining logic remains intact as per your requirement
    return spark.table(f"{SILVER}.listings_silver_merged").select(
        "listing_id", 
        "brand", 
        "model", 
        "year", 
        "listing_date",
        "price_rub", 
        "price_usd", 
        "price_category", 
        "fuel_type", 
        "transmission_type",
        "engine_power",
        "mileage_km",
        (year(col("listing_date")) - col("year")).alias("car_age_at_listing"),
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
