# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Gold Layer — Star Schema DLT Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & Configuration

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
# MAGIC ## `dim_date` — Static Calendar | No SCD2
# MAGIC Required columns: date_key, month, quarter, year, week.

# COMMAND ----------

@dlt.table(
    name             = "dim_date",
    comment          = "Gold: Gap-free Date dimension 2010-2030. "
                       "Static calendar — no SCD2. "
                       "PK: date_key. Columns: date, month, quarter, year, week_of_year. "
                       "Audit: gold_load_dt.",
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
# MAGIC ## Lookup Dimensions — Static | No SCD2

# COMMAND ----------

@dlt.table(
    name             = "dim_price_category",
    comment          = "Gold Lookup: Price category codes. Static — derived from Silver CASE bands. "
                       "FK: price_category_key (INT) → fact_listings.price_category_key. "
                       "Values: BUDGET / MID_RANGE / PREMIUM / LUXURY / UNKNOWN.",
    table_properties = {**GOLD_PROPS, "type": "lookup"},
)
def dim_price_category():
    data = [
        (1, "BUDGET",    "< 300,000 RUB"),
        (2, "MID_RANGE", "300,000 – 700,000 RUB"),
        (3, "PREMIUM",   "700,001 – 1,500,000 RUB"),
        (4, "LUXURY",    "> 1,500,000 RUB"),
        (5, "UNKNOWN",   "Unclassified"),
    ]
    schema = "price_category_key INT, price_category STRING, price_range_desc STRING"
    return (
        spark.createDataFrame(data, schema)
        .withColumn("gold_load_dt", F.current_timestamp())
    )


@dlt.table(
    name             = "dim_steering",
    comment          = "Gold Lookup: Steering wheel side codes. "
                       "Source: listings_silver_merged.steering_wheel "
                       "(Bronze column sWheel, renamed in Silver _transform_listings). "
                       "NOT present in car catalog — separate dim is correct. "
                       "FK: steering_key (INT) → fact_listings.steering_key.",
    table_properties = {**GOLD_PROPS, "type": "lookup"},
)
def dim_steering():
    return (
        spark.table(f"{SILVER}.listings_silver_merged")
        .select(F.col("steering_wheel"))
        .filter(F.col("steering_wheel").isNotNull())
        .distinct()
        .withColumn(
            "steering_key",
            F.dense_rank().over(Window.orderBy("steering_wheel")).cast("int")
        )
        .select("steering_key", "steering_wheel")
        .withColumn("gold_load_dt", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## `dim_car` — SCD Type 2
# MAGIC
# MAGIC Source: `silver.car_catalog_transformation`
# MAGIC Natural PK: `brand + model`

# COMMAND ----------

dlt.create_streaming_table(
    name             = "dim_car",
    comment          = "Gold SCD2: Car specifications from car_catalog_transformation. "
                       "Contains fuel_type, transmission, drive_type — no separate lookup dims needed. "
                       "Natural PK: brand + model. "
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
# MAGIC ## `dim_location` — SCD Type 2
# MAGIC
# MAGIC Source: `silver.geography_transformation`
# MAGIC Natural PK: `city_prepositional`

# COMMAND ----------

dlt.create_streaming_table(
    name             = "dim_location",
    comment          = "Gold SCD2: City/region with lat/lon from geography_transformation. "
                       "Natural PK: city_prepositional. "
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
# MAGIC ## `dim_listing_details` — SCD Type 2
# MAGIC
# MAGIC Source: `silver.listings_text_transformation`
# MAGIC Natural PK: `listing_id`

# COMMAND ----------

dlt.create_streaming_table(
    name             = "dim_listing_details",
    comment          = "Gold SCD2: Russian listing text descriptions from listings_text_transformation. "
                       "Tracks description edits over time. "
                       "Natural PK: listing_id. "
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
# MAGIC ## `dim_listing_photos` — SCD Type 2
# MAGIC
# MAGIC Source: `silver.listings_photo_transformation`
# MAGIC Natural PK: `listing_id + photo_url_clean`

# COMMAND ----------

dlt.create_streaming_table(
    name             = "dim_listing_photos",
    comment          = "Gold SCD2: Photo URLs per listing from listings_photo_transformation. "
                       "~7.9M rows, one-to-many. "
                       "Natural PK: listing_id + photo_url_clean. "
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
# MAGIC ## `fact_listings` — Fact Table

# COMMAND ----------

@dlt.table(
    name             = "fact_listings",
    comment          = "Gold Fact: Car listings — star schema, no raw strings. "
                       "listing_id = degenerate dimension (grain key). "
                       "fuel_type/transmission/drive_type resolved via dim_car at query time. "
                       "Derived: car_age_at_listing, is_high_mileage, price_per_hp_usd. "
                       "Audit: bronze_load_dt → silver_load_dt → gold_load_dt.",
    table_properties = {
        **GOLD_PROPS,
        "type"              : "fact",
        "grain"             : "listing_id",
        "fk_date"           : "listing_date -> dim_date.date_key",
        "fk_car"            : "brand+model -> dim_car.(brand,model)",
        "fk_location"       : "location_key -> dim_location.city_prepositional",
        "fk_details"        : "listing_id -> dim_listing_details.listing_id",
        "fk_photos"         : "listing_id -> dim_listing_photos (photo_count denorm)",
        "fk_price_category" : "price_category_key -> dim_price_category.price_category_key",
        "fk_steering"       : "steering_key -> dim_steering.steering_key",
    },
)
def fact_listings():
    df = spark.table(f"{SILVER}.listings_silver_merged")

    # ── Photo count — denormalized from Silver photo table ────────────────────
    photo_counts = (
        spark.table(f"{SILVER}.listings_photo_transformation")
        .groupBy("listing_id")
        .agg(F.count("photo_url_clean").alias("photo_count"))
    )

    # ── Integer FK key resolution from lookup dims ────────────────────────────
    price_cat_map = (
        dlt.read("dim_price_category")
        .select("price_category", "price_category_key")
    )
    steer_map = (
        dlt.read("dim_steering")
        .select("steering_wheel", "steering_key")
    )

    return (
        df
        .join(photo_counts,  on="listing_id",    how="left")
        .join(price_cat_map, on="price_category", how="left")
        .join(steer_map,     on="steering_wheel", how="left")
        .withColumn("photo_count",  F.coalesce(F.col("photo_count"), F.lit(0)))
        .withColumn("listing_date", F.col("listing_date").cast("date"))
        .select(
            # ── Degenerate dimension (grain key) ───────────────────────────────
            "listing_id",

            # ── FK → dim_date.date_key ─────────────────────────────────────────
            "listing_date",

            # ── FK → dim_car.(brand, model) ────────────────────────────────────
            # fuel_type / transmission / drive_type live in dim_car — not stored here
            "brand",
            "model",

            # ── FK → dim_location.city_prepositional ───────────────────────────
            F.col("city_prepositional").alias("location_key"),

            # ── FK → dim_price_category (INT) ─────────────────────────────────
            F.col("price_category_key"),

            # ── FK → dim_steering (INT) ────────────────────────────────────────
            # steering_wheel = Bronze sWheel column, renamed in Silver
            F.col("steering_key"),

            # ── Numeric listing attributes (no strings) ────────────────────────
            "manufacture_year",      # INT
            "engine_power",          # INT — HP from Bronze power column
            "mileage_km",            # INT — probeg column
            "has_license",           # INT (0/1)
            "listing_year",          # INT
            "listing_month",         # INT
            "car_age_years",         # INT — 2023 - manufacture_year (Silver-computed)

            # ── Financial measures ─────────────────────────────────────────────
            "price_rub",             # DOUBLE
            "price_usd",             # DOUBLE — price_rub / 82.5

            # ── Gold-level derived measures ────────────────────────────────────
            # listing_date already cast to DATE above — safe to reference by name
            (F.year("listing_date") - F.col("manufacture_year"))
                .alias("car_age_at_listing"),                  # INT
            F.when(F.col("mileage_km") > 100000, True)
                .otherwise(False).alias("is_high_mileage"),    # BOOLEAN
            F.round(
                F.col("price_usd") / F.nullif(F.col("engine_power"), F.lit(0)), 2
            ).alias("price_per_hp_usd"),                       # DOUBLE

            # ── Denormalized photo count ───────────────────────────────────────
            F.col("photo_count").cast("int"),                  # INT

            # ── RGB color codes — cast STRING → INT ───────────────────────────
            # Silver stores R/G/B as empty string fallback: coalesce(cast(R as string), '')
            F.col("color_r").cast("int").alias("color_r"),
            F.col("color_g").cast("int").alias("color_g"),
            F.col("color_b").cast("int").alias("color_b"),

            # ── Full audit chain ───────────────────────────────────────────────
            "bronze_load_dt",
            "bronze_source_file",
            "silver_load_dt",
            F.current_timestamp().alias("gold_load_dt"),
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregates (5 tables)
# MAGIC
# MAGIC All aggregates read from `fact_listings` via `dlt.read()`.
# MAGIC Where category labels are needed for reporting, the relevant lookup dim is joined back.

# COMMAND ----------

# ── AGG 1: Monthly Sales Trend ───────────────────────────────────────────────
@dlt.table(
    name             = "agg_monthly_sales_trend",
    comment          = "Gold Agg: Listing volume and revenue by month, brand, price segment. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_monthly_trend():
    fact      = dlt.read("fact_listings")
    price_cat = dlt.read("dim_price_category").select("price_category_key", "price_category")
    return (
        fact
        .join(price_cat, on="price_category_key", how="left")
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
    fact      = dlt.read("fact_listings")
    price_dim = dlt.read("dim_price_category").select("price_category_key", "price_category")
    # fuel_type resolved by joining to dim_car on brand + model
    dim_car   = dlt.read("dim_car").filter(F.col("__END_AT").isNull()).select("brand", "model", "fuel_type")
    return (
        fact
        .join(dim_car,   on=["brand", "model"],   how="left")
        .join(price_dim, on="price_category_key", how="left")
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
    fact      = dlt.read("fact_listings")
    price_dim = dlt.read("dim_price_category").select("price_category_key", "price_category")
    dim_car   = dlt.read("dim_car").filter(F.col("__END_AT").isNull()).select("brand", "model", "fuel_type")
    return (
        fact
        .join(dim_car,   on=["brand", "model"],   how="left")
        .join(price_dim, on="price_category_key", how="left")
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
