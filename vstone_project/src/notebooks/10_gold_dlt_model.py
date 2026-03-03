# Databricks notebook source
# %sql
# -- DROP TABLE vstone_catalog.gold.dim_date;
# DROP TABLE vstone_catalog.gold.dim_car;
# DROP TABLE vstone_catalog.gold.dim_listing_details;
# DROP TABLE vstone_catalog.gold.dim_location;
# DROP TABLE vstone_catalog.gold.fact_listings;


# COMMAND ----------

# src/notebooks/10_gold_dlt_model.py
import dlt
from pyspark.sql.functions import (
    col, current_timestamp, lit, year, month, dayofmonth, 
    date_format, lead, desc, isnull, when, round
)
from pyspark.sql.window import Window

CATALOG = "vstone_catalog"
SILVER = f"{CATALOG}.silver"

# ======================================================================================
# 1. SCD TYPE 2 DIMENSIONS (Clean - No Expectations)
# ======================================================================================

# --- DIMENSION 1: dim_date ---
dlt.create_streaming_table(
    name="dim_date",
    comment="Gold: Date dimension with SCD2.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "date_key",
        "pipelines.autoOptimize.zOrderCols": "year,month"
    }
)

@dlt.view
def date_source_v():
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
    comment="Gold: Car specs with SCD2.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "brand, model"
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
    target="dim_car",
    source="car_source_v",
    keys=["brand", "model"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=2
)

# --- DIMENSION 3: dim_location ---
dlt.create_streaming_table(
    name="dim_location",
    comment="Gold: Geography dimension with SCD2.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "city_prepositional"
    }
)

@dlt.view
def location_source_v():
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
    comment="Gold: Text descriptions with SCD2.",
    table_properties={
        "layer": "gold", 
        "scd_type": "2", 
        "pk": "listing_id"
    }
)

@dlt.view
def text_source_v():
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
# 2. FACT TABLE (Detailed Metrics & Joins)
# ======================================================================================

@dlt.table(
    name="fact_listings",
    comment="Gold: Master Fact table with calculated metrics (USD conversion & Price categorization).",
    table_properties={
        "layer": "gold", 
        "type": "fact",
        "pk": "listing_id"
    }
)
def fact_listings():
    # Fixed: Calculating price_usd and price_category on-the-fly
    return spark.table(f"{SILVER}.listings_silver_merged").select(
        "listing_id", 
        "brand", 
        "model", 
        "year", 
        "listing_date",
        "price_rub", 
        # # 1. Calculate USD (Assuming fixed rate for demo/logic)
        # round(col("price_rub") / 90.0, 2).alias("price_usd"),
        # # 2. Derive Price Category
        # when(col("price_rub") > 5000000, "Luxury")
        #     .when(col("price_rub") > 2000000, "Premium")
        #     .otherwise("Standard").alias("price_category"),
        "fuel_type", 
        "transmission_type",
        "engine_power",
        "mileage_km",
        (year(col("listing_date")) - col("year")).alias("car_age_at_listing"),
        when(col("mileage_km") > 100000, True).otherwise(False).alias("is_high_mileage"),
        # 3. Use the calculated USD column for HP metric
        round((col("price_rub") / 90.0) / col("engine_power"), 2).alias("price_per_hp_usd"),
        col("city_prepositional").alias("location_key"), 
        current_timestamp().alias("gold_load_dt")
    )

# COMMAND ----------

import dlt
from pyspark.sql.functions import (
    col, current_timestamp, sum, count, avg, max, min, 
    date_format, round, desc, dense_rank, lit, countDistinct, when
)

# ============================================================
# AGGREGATE 1: agg_monthly_sales_trend
# Logic: Price Category recalculated to avoid Unresolved Column error
# ============================================================
@dlt.table(
    name="agg_monthly_sales_trend",
    comment="Gold Aggregate: Trend analysis with Brand-level granularity.",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_monthly_trend():
    return (
        dlt.read("fact_listings")
        .withColumn("month_year", date_format(col("listing_date"), "yyyy-MM"))
        .groupBy("month_year", "brand", "price_category") 
        .agg(
            count("listing_id").alias("total_listings"),
            round(avg("price_rub"), 0).alias("avg_price_rub"),
            # Corrected: Using price_rub conversion directly to avoid resolution error
            round(sum(col("price_rub") / 90.0), 0).alias("total_revenue_usd")
        )
        .orderBy("month_year", desc("total_listings"))
        .withColumn("gold_load_dt", current_timestamp())
    )

# ============================================================
# AGGREGATE 2: agg_brand_location_performance
# Logic: average price calculated from base price_rub
# ============================================================
@dlt.table(
    name="agg_brand_location_performance",
    comment="Gold Aggregate: Detailed brand performance across regions.",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_brand_performance():
    return (
        dlt.read("fact_listings")
        .groupBy("brand", "location_key") 
        .agg(
            count("listing_id").alias("listing_count"),
            # Corrected: price_usd resolution fix
            round(avg(col("price_rub") / 90.0), 0).alias("avg_price_usd"),
            max("engine_power").alias("max_hp_in_region")
        )
        .withColumn("gold_load_dt", current_timestamp())
    )

# ============================================================
# AGGREGATE 3: agg_regional_market_depth
# Logic remains same (uses existing physical columns)
# ============================================================
@dlt.table(
    name="agg_regional_market_depth",
    comment="Gold Aggregate: Regional availability by fuel and category.",
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
# Logic: Full market view with vehicle age analysis
# ============================================================
@dlt.table(
    name="agg_comprehensive_kpi_cube",
    comment="Gold Aggregate: High-density data for multi-dimensional dashboards.",
    table_properties={"layer": "gold", "type": "aggregate"}
)
def agg_kpi_cube():
    return (
        dlt.read("fact_listings")
        .groupBy("brand", "model", "year", "price_category", "fuel_type", "is_high_mileage")
        .agg(
            count("listing_id").alias("listing_volume"),
            # Corrected: price_usd resolution fix
            round(avg(col("price_rub") / 90.0), 2).alias("avg_market_price"),
            round(avg("car_age_at_listing"), 1).alias("avg_vehicle_age")
        )
        .withColumn("gold_load_dt", current_timestamp())
    )
