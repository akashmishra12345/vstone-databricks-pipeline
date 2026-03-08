# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Gold Layer — Star Schema DLT Pipeline
# MAGIC
# MAGIC **Architecture:** Kimball Star Schema | All tables fully materialized
# MAGIC
# MAGIC | Table | DLT API | SCD | Materialized | Audit column |
# MAGIC |---|---|---|---|---|
# MAGIC | `dim_date` | `@dlt.table` | None | ✅ Materialized table | `gold_load_dt` |
# MAGIC | `dim_car` | `create_streaming_table` + `apply_changes` | SCD2 | ✅ Materialized table | `silver_load_dt` + `__START_AT/__END_AT` |
# MAGIC | `dim_location` | `create_streaming_table` + `apply_changes` | SCD2 | ✅ Materialized table | `silver_load_dt` + `__START_AT/__END_AT` |
# MAGIC | `dim_listing_details` | `create_streaming_table` + `apply_changes` | SCD2 | ✅ Materialized table | `silver_load_dt` + `__START_AT/__END_AT` |
# MAGIC | `dim_listing_photos` | `create_streaming_table` + `apply_changes` | SCD2 | ✅ Materialized table | `silver_load_dt` + `__START_AT/__END_AT` |
# MAGIC | `fact_listings` | `@dlt.table` | — | ✅ Materialized table | `gold_load_dt` |
# MAGIC | `agg_monthly_sales_trend` | `@dlt.table` | — | ✅ Materialized table | `gold_load_dt` |
# MAGIC | `agg_brand_location_performance` | `@dlt.table` | — | ✅ Materialized table | `gold_load_dt` |
# MAGIC | `agg_regional_market_depth` | `@dlt.table` | — | ✅ Materialized table | `gold_load_dt` |
# MAGIC | `agg_comprehensive_kpi_cube` | `@dlt.table` | — | ✅ Materialized table | `gold_load_dt` |
# MAGIC | `agg_top_10_brands_by_spend` | `@dlt.table` | — | ✅ Materialized table | `gold_load_dt` |
# MAGIC
# MAGIC > **Note on SCD2 dims:** `create_streaming_table` + `apply_changes` is the **only** DLT API that supports SCD2. These tables are physically identical to `@dlt.table` — fully materialized Delta tables. The UI label differs ('Streaming table' vs 'Materialized view') but both are materialized on disk.
# MAGIC
# MAGIC > **Note on dim_car 2.2K rows:** This is correct. SCD2 key = unique `brand+model`. 2.2K unique car models from 117K Silver rows. Reconciliation: `SELECT COUNT(DISTINCT brand, model) FROM silver.car_catalog_transformation` = `SELECT COUNT(*) FROM gold.dim_car WHERE __CURRENT = true`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Imports & Configuration

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "vstone_catalog"
SILVER  = f"{CATALOG}.silver"

GOLD_PROPS = {
    "quality"                   : "gold",
    "delta.enableChangeDataFeed": "true",
    "pipelines.reset.allowed"   : "true",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — `dim_date` — `@dlt.table` | Materialized table

# COMMAND ----------

# ── DIM_DATE ─ Materialized table | Static calendar | No SCD2 ────────────────
# @dlt.table = fully materialized Delta table (shows as "Materialized table" in UI)
# PK: date_key | Range: 2010-01-01 → 2030-12-31
# Audit: gold_load_dt (explicit current_timestamp)

@dlt.table(
    name             = "dim_date",
    comment          = "Gold: Gap-free Date dimension 2010-2030. "
                       "Static calendar — no SCD2. Materialized table. "
                       "PK: date_key. Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "pk": "date_key"},
)
def dim_date():
    return (
        spark.range(1)
        .selectExpr(
            "explode(sequence(to_date('2010-01-01'), to_date('2030-12-31'), interval 1 day)) as date_key"
        )
        .select(
            F.col("date_key"),
            F.year("date_key").alias("year"),
            F.quarter("date_key").alias("quarter"),
            F.month("date_key").alias("month"),
            F.date_format("date_key", "MMMM").alias("month_name"),
            F.weekofyear("date_key").alias("week_of_year"),
            F.dayofmonth("date_key").alias("day"),
            F.date_format("date_key", "EEEE").alias("day_name"),
            F.when(F.dayofweek("date_key").isin(1, 7), True)
             .otherwise(False).alias("is_weekend"),
            F.current_timestamp().alias("gold_load_dt"),
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — `dim_car` — SCD Type 2 | Materialized table

# COMMAND ----------

# ── DIM_CAR ─ Materialized table | SCD Type 2 ────────────────────────────────
# create_streaming_table + apply_changes = only DLT API for SCD2.
# Produces a fully materialized Delta table (same storage as @dlt.table).
# PK: brand + model | Source: silver.car_catalog_transformation
# Audit: silver_load_dt (pass-through) + __START_AT / __END_AT (auto-added by SCD2)
#
# dim_car row count = unique brand+model combinations (~2.2K).
# This is CORRECT — it is a dimension, not a fact.
# Reconciliation: COUNT(DISTINCT brand, model) in Silver
#               = COUNT(*) WHERE __CURRENT = true in dim_car

dlt.create_streaming_table(
    name             = "dim_car",
    comment          = "Gold SCD2: Car specifications. Tracks engine/generation/trim changes. "
                       "Materialized table. PK: brand+model. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "brand,model"},
)

dlt.apply_changes(
    target             = "dim_car",
    source             = f"{SILVER}.car_catalog_transformation",
    keys               = ["brand", "model"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — `dim_location` — SCD Type 2 | Materialized table

# COMMAND ----------

# ── DIM_LOCATION ─ Materialized table | SCD Type 2 ───────────────────────────
# PK: city_prepositional | Source: silver.geography_transformation
# Schema: city_prepositional, city_name, latitude, longitude,
#         bronze_load_dt, bronze_source_file, silver_load_dt
# Audit: silver_load_dt + __START_AT / __END_AT

dlt.create_streaming_table(
    name             = "dim_location",
    comment          = "Gold SCD2: City/region with lat/lon. Tracks coordinate/name changes. "
                       "Materialized table. PK: city_prepositional. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "city_prepositional"},
)

dlt.apply_changes(
    target             = "dim_location",
    source             = f"{SILVER}.geography_transformation",
    keys               = ["city_prepositional"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — `dim_listing_details` — SCD Type 2 | Materialized table

# COMMAND ----------

# ── DIM_LISTING_DETAILS ─ Materialized table | SCD Type 2 ────────────────────
# PK: listing_id | Source: silver.listings_text_transformation
# Schema: text, load_dt, source_file, listing_id,
#         bronze_load_dt, bronze_source_file, silver_load_dt
# Audit: silver_load_dt + __START_AT / __END_AT

dlt.create_streaming_table(
    name             = "dim_listing_details",
    comment          = "Gold SCD2: Listing text description. Tracks description edits. "
                       "Materialized table. PK: listing_id. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "listing_id"},
)

dlt.apply_changes(
    target             = "dim_listing_details",
    source             = f"{SILVER}.listings_text_transformation",
    keys               = ["listing_id"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — `dim_listing_photos` — SCD Type 2 | Materialized table

# COMMAND ----------

# ── DIM_LISTING_PHOTOS ─ Materialized table | SCD Type 2 ─────────────────────
# PK: listing_id + photo_url_clean (composite — one listing → many photos)
# Source: silver.listings_photo_transformation (~7.9M rows)
# Schema: listing_id, photo_url, photo_url_clean,
#         bronze_load_dt, bronze_source_file, silver_load_dt
# Audit: silver_load_dt + __START_AT / __END_AT

dlt.create_streaming_table(
    name             = "dim_listing_photos",
    comment          = "Gold SCD2: Photo URLs per listing. ~7.9M rows, one-to-many. "
                       "Tracks photo URL changes. Materialized table. "
                       "PK: listing_id+photo_url_clean. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "listing_id,photo_url_clean"},
)

dlt.apply_changes(
    target             = "dim_listing_photos",
    source             = f"{SILVER}.listings_photo_transformation",
    keys               = ["listing_id", "photo_url_clean"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — `fact_listings` — `@dlt.table` | Materialized table

# COMMAND ----------

# ── FACT_LISTINGS ─ Materialized table | Batch | @dlt.table ──────────────────
# Source: silver.listings_silver_merged (1,083,237 rows — already deduped)
# Silver → Gold is 1:1 — no additional dedup or row drops.
#
# FK relationships (all 4 dimensions):
#   listing_id    → dim_listing_details.listing_id
#   listing_date  → dim_date.date_key
#   brand + model → dim_car.(brand, model)
#   location_key  → dim_location.city_prepositional
#   photo_count   → denormalized from dim_listing_photos
#
# Gold-level derived columns:
#   car_age_at_listing = YEAR(listing_date) - manufacture_year
#   is_high_mileage    = mileage_km > 100,000
#   price_per_hp_usd   = price_usd / engine_power
# Audit: gold_load_dt = current_timestamp() at Gold load time

@dlt.table(
    name             = "fact_listings",
    comment          = "Gold Fact: Car listings — full star schema 4 dim FKs. "
                       "1:1 with Silver (no dedup). PK: listing_id. "
                       "Derived: car_age_at_listing, is_high_mileage, price_per_hp_usd. "
                       "Audit: gold_load_dt.",
    table_properties = {
        **GOLD_PROPS,
        "type"        : "fact",
        "pk"          : "listing_id",
        "fk_details"  : "listing_id -> dim_listing_details.listing_id",
        "fk_date"     : "listing_date -> dim_date.date_key",
        "fk_car"      : "brand+model -> dim_car.(brand,model)",
        "fk_location" : "location_key -> dim_location.city_prepositional",
        "fk_photos"   : "listing_id -> dim_listing_photos (photo_count denorm)",
    },
)
def fact_listings():
    df = spark.table(f"{SILVER}.listings_silver_merged")

    photo_counts = (
        spark.table(f"{SILVER}.listings_photo_transformation")
        .groupBy("listing_id")
        .agg(F.count("photo_url_clean").alias("photo_count"))
    )

    return (
        df
        .join(photo_counts, on="listing_id", how="left")
        .withColumn("photo_count", F.coalesce(F.col("photo_count"), F.lit(0)))
        .select(
            # ── PK / FK → dim_listing_details & dim_listing_photos ─────────────
            "listing_id",

            # ── FK → dim_date.date_key ─────────────────────────────────────────
            F.col("listing_date").cast("date").alias("listing_date"),

            # ── FK → dim_car.(brand, model) ────────────────────────────────────
            "brand",
            "model",

            # ── FK → dim_location.city_prepositional ───────────────────────────
            F.col("city_prepositional").alias("location_key"),

            # ── Listing attributes ─────────────────────────────────────────────
            "manufacture_year",
            "engine_power",
            "mileage_km",
            "fuel_type",
            "transmission_type",
            "drive_type",
            "steering_wheel",
            "trim_level",
            "has_license",
            "color_r",
            "color_g",
            "color_b",

            # ── Silver-computed financials (1:1, no recalculation) ─────────────
            "price_rub",
            "price_usd",
            "price_category",
            "car_age_years",
            "listing_year",
            "listing_month",

            # ── Gold-level derived metrics ─────────────────────────────────────
            (F.year(F.col("listing_date")) - F.col("manufacture_year"))
                .alias("car_age_at_listing"),
            F.when(F.col("mileage_km") > 100000, True)
                .otherwise(False).alias("is_high_mileage"),
            F.round(
                F.col("price_usd") / F.nullif(F.col("engine_power"), F.lit(0)), 2
            ).alias("price_per_hp_usd"),

            # ── Denormalized photo count ────────────────────────────────────────
            "photo_count",

            # ── Audit ──────────────────────────────────────────────────────────
            "bronze_load_dt",
            "bronze_source_file",
            "silver_load_dt",
            F.current_timestamp().alias("gold_load_dt"),
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Aggregates (5 tables) — `@dlt.table` | Materialized tables

# COMMAND ----------

# ── AGGREGATES ─ All @dlt.table | Materialized tables | gold_load_dt on each ──
# All 5 aggregates read from fact_listings via dlt.read() (same DLT pipeline).

# ── AGG 1: Monthly Sales Trend ───────────────────────────────────────────────
@dlt.table(
    name             = "agg_monthly_sales_trend",
    comment          = "Gold Agg: Listing volume and revenue by month, brand, price segment. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_monthly_trend():
    return (
        dlt.read("fact_listings")
        .withColumn("month_year", F.date_format("listing_date", "yyyy-MM"))
        .groupBy("month_year", "brand", "price_category")
        .agg(
            F.count("listing_id").alias("total_listings"),
            F.round(F.avg("price_rub"), 0).alias("avg_price_rub"),
            F.round(F.avg("price_usd"), 0).alias("avg_price_usd"),
            F.round(F.sum("price_usd"), 0).alias("total_revenue_usd"),
        )
        .orderBy("month_year", F.desc("total_listings"))
        .withColumn("gold_load_dt", F.current_timestamp())
    )

# ── AGG 2: Brand + Location Performance ──────────────────────────────────────
@dlt.table(
    name             = "agg_brand_location_performance",
    comment          = "Gold Agg: Brand performance by region. Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_brand_performance():
    return (
        dlt.read("fact_listings")
        .groupBy("brand", "location_key")
        .agg(
            F.count("listing_id").alias("listing_count"),
            F.round(F.avg("price_usd"), 0).alias("avg_price_usd"),
            F.round(F.avg("mileage_km"), 0).alias("avg_mileage_km"),
            F.max("engine_power").alias("max_hp_in_region"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )

# ── AGG 3: Regional Market Depth ─────────────────────────────────────────────
@dlt.table(
    name             = "agg_regional_market_depth",
    comment          = "Gold Agg: Inventory depth by city, fuel type, price segment. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_regional_depth():
    return (
        dlt.read("fact_listings")
        .groupBy("location_key", "fuel_type", "price_category")
        .agg(
            F.count("listing_id").alias("inventory_count"),
            F.round(F.avg("mileage_km"), 0).alias("avg_mileage"),
            F.round(F.avg("price_usd"), 0).alias("avg_price_usd"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )

# ── AGG 4: Comprehensive KPI Cube ─────────────────────────────────────────────
@dlt.table(
    name             = "agg_comprehensive_kpi_cube",
    comment          = "Gold Agg: Multi-dim KPI cube — brand, model, year, segment, fuel, mileage. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_kpi_cube():
    return (
        dlt.read("fact_listings")
        .groupBy(
            "brand", "model", "manufacture_year",
            "price_category", "fuel_type", "is_high_mileage",
        )
        .agg(
            F.count("listing_id").alias("listing_volume"),
            F.round(F.avg("price_usd"), 2).alias("avg_market_price_usd"),
            F.round(F.avg("car_age_at_listing"), 1).alias("avg_age_at_listing"),
            F.round(F.avg("mileage_km"), 0).alias("avg_mileage_km"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )

# ── AGG 5: Top 10 Brands by Total Market Value ────────────────────────────────
@dlt.table(
    name             = "agg_top_10_brands_by_spend",
    comment          = "Gold Agg: Top 10 brands by cumulative USD market value. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_top_10_brands():
    return (
        dlt.read("fact_listings")
        .groupBy("brand")
        .agg(
            F.round(F.sum("price_usd"), 0).alias("total_market_value_usd"),
            F.count("listing_id").alias("total_listings"),
            F.round(F.avg("price_usd"), 0).alias("avg_price_usd"),
        )
        .orderBy(F.desc("total_market_value_usd"))
        .limit(10)
        .withColumn("gold_load_dt", F.current_timestamp())
    )
