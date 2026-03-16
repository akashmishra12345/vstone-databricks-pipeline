# Databricks notebook source
# MAGIC %md
# MAGIC # 10 -- Gold Layer -- Star Schema DLT Pipeline

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

# ── Stable surrogate key helpers ──────────────────────────────────────────────
def _car_sk(brand_col="brand", model_col="model"):
    return F.crc32(
        F.concat_ws("|",
            F.lower(F.trim(F.col(brand_col))),
            F.lower(F.trim(F.col(model_col)))
        )
    ).cast("int")

def _location_sk(city_col="city_prepositional"):
    return F.crc32(F.lower(F.trim(F.col(city_col)))).cast("int")

# ── listing_id cast helper ────────────────────────────────────────────────────
def _listing_id_bigint(col="listing_id"):
    return F.col(col).cast("bigint")


# COMMAND ----------

# MAGIC %md
# MAGIC ## `dim_date` -- Static Calendar Dimension | No SCD2

# COMMAND ----------

@dlt.table(
    name             = "dim_date",
    comment          = "Gold: Gap-free date dimension 2010-2030. "
                       "Static calendar -- no SCD2. "
                       "PK: date_key (DATE). Columns: year, quarter, month, month_name, "
                       "week_of_year, day, day_name, is_weekend. Audit: gold_load_dt.",
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
# MAGIC ## Lookup Dimensions -- Static | No SCD2
# MAGIC
# MAGIC Small code-list tables with integer primary keys.
# MAGIC These are the FK targets for price_category_key and steering_key in fact_listings.

# COMMAND ----------

@dlt.table(
    name             = "dim_price_category",
    comment          = "Gold Lookup: Price category codes. Static -- derived from Silver CASE bands. "
                       "PK: price_category_key (INT). "
                       "Values: BUDGET / MID_RANGE / PREMIUM / LUXURY / UNKNOWN. "
                       "FK: fact_listings.price_category_key -> dim_price_category.price_category_key.",
    table_properties = {**GOLD_PROPS, "type": "lookup"},
)
def dim_price_category():
    data = [
        (1, "BUDGET",    "< 300,000 RUB"),
        (2, "MID_RANGE", "300,000 - 700,000 RUB"),
        (3, "PREMIUM",   "700,001 - 1,500,000 RUB"),
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
                       "PK: steering_key (INT). "
                       "FK: fact_listings.steering_key -> dim_steering.steering_key.",
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
# MAGIC ## `dim_car` -- SCD Type 2

# COMMAND ----------

# ── Pre-processing view: add car_sk surrogate key before apply_changes ────────

@dlt.view(name="dim_car_source")
def dim_car_source():
    """
    Streaming view: enriches silver.car_catalog_transformation with car_sk.
    car_sk = crc32(lower(brand) || '|' || lower(model)) cast to INT.
    Stable and reproducible -- same brand+model always yields the same car_sk.
    Uses readStream so apply_changes() can consume it as a streaming source.
    """
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.car_catalog_transformation")
        .withColumn("car_sk", _car_sk("brand", "model"))
    )


dlt.create_streaming_table(
    name             = "dim_car",
    comment          = "Gold SCD2: Car specifications from car_catalog_transformation. "
                       "Natural PK: brand + model (composite string). "
                       "Surrogate PK: car_sk (INT) = crc32(lower(brand)|lower(model)). "
                       "FK target: fact_listings.car_sk -> dim_car.car_sk. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "car_sk", "natural_key": "brand,model"},
)

dlt.apply_changes(
    target             = "dim_car",
    source             = "dim_car_source",
    keys               = ["brand", "model"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## `dim_location` -- SCD Type 2

# COMMAND ----------

# ── Pre-processing view: add location_sk surrogate key before apply_changes ───
# MUST use spark.readStream so this is a streaming view consumable by apply_changes().

@dlt.view(name="dim_location_source")
def dim_location_source():
    """
    Streaming view: enriches silver.geography_transformation with location_sk.
    location_sk = crc32(lower(city_prepositional)) cast to INT.
    Uses readStream so apply_changes() can consume it as a streaming source.
    """
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.geography_transformation")
        .withColumn("location_sk", _location_sk("city_prepositional"))
    )


dlt.create_streaming_table(
    name             = "dim_location",
    comment          = "Gold SCD2: Russian city/region from geography_transformation. "
                       "Natural PK: city_prepositional (string). "
                       "Surrogate PK: location_sk (INT) = crc32(lower(city_prepositional)). "
                       "FK target: fact_listings.location_sk -> dim_location.location_sk. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "location_sk", "natural_key": "city_prepositional"},
)

dlt.apply_changes(
    target             = "dim_location",
    source             = "dim_location_source",
    keys               = ["city_prepositional"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## `dim_listing_details` -- SCD Type 2

# COMMAND ----------

# ── Pre-processing view: add text_hash + cast listing_id to BIGINT ──────────

@dlt.view(name="dim_listing_details_source")
def dim_listing_details_source():
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.listings_text_transformation")
        .withColumn("listing_id", _listing_id_bigint("listing_id"))
        .withColumn("text_hash",  F.md5(F.coalesce(F.col("text"), F.lit(""))))
    )


dlt.create_streaming_table(
    name             = "dim_listing_details",
    comment          = "Gold SCD2: Russian listing text descriptions from listings_text_transformation. "
                       "Natural PK: listing_id (BIGINT -- cast from Silver string at Gold boundary). "
                       "Change-detection: text_hash (MD5 of text) -- new SCD2 version when text changes. "
                       "word_count removed -- metric belongs in fact_listings. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "listing_id"},
)

dlt.apply_changes(
    target             = "dim_listing_details",
    source             = "dim_listing_details_source",
    keys               = ["listing_id"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## `dim_listing_photos` -- SCD Type 2
# MAGIC
# MAGIC

# COMMAND ----------

# ── Pre-processing view: cast listing_id to BIGINT before apply_changes ─────

@dlt.view(name="dim_listing_photos_source")
def dim_listing_photos_source():
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.listings_photo_transformation")
        .withColumn("listing_id", _listing_id_bigint("listing_id"))
    )


dlt.create_streaming_table(
    name             = "dim_listing_photos",
    comment          = "Gold SCD2: Photo URLs per listing from listings_photo_transformation. "
                       "Natural PK: listing_id (BIGINT) + photo_url_clean (composite). "
                       "listing_id cast STRING->BIGINT at Gold boundary. "
                       "One-to-many: one listing can have many photos. "
                       "photo_count denormalized into fact_listings. "
                       "Audit: silver_load_dt + __START_AT/__END_AT.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "listing_id,photo_url_clean"},
)

dlt.apply_changes(
    target             = "dim_listing_photos",
    source             = "dim_listing_photos_source",
    keys               = ["listing_id", "photo_url_clean"],
    sequence_by        = "silver_load_dt",
    stored_as_scd_type = 2,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## `fact_listings` -- Fact Table (Kimball Star Schema)
# MAGIC

# COMMAND ----------

@dlt.table(
    name             = "fact_listings",
    comment          = "Gold Fact: Car listings -- Kimball star schema compliant. "
                       "GRAIN: one row per listing_id (BIGINT -- degenerate dimension). "
                       "INTEGER FKs ONLY: car_sk->dim_car, location_sk->dim_location, "
                       "price_category_key->dim_price_category, steering_key->dim_steering. "
                       "DATE FK: listing_date->dim_date.date_key. "
                       "listing_id cast STRING->BIGINT at Gold boundary (same as color_r/g/b->INT). "
                       "MEASURES: price_rub, price_usd, mileage_km, engine_power, photo_count, "
                       "car_age_at_listing, is_high_mileage, price_per_hp_usd, word_count. "
                       "AUDIT CHAIN: bronze_load_dt -> silver_load_dt -> gold_load_dt.",
    table_properties = {
        **GOLD_PROPS,
        "type"              : "fact",
        "grain"             : "listing_id",
        "fk_date"           : "listing_date -> dim_date.date_key",
        "fk_car"            : "car_sk (INT) -> dim_car.car_sk",
        "fk_location"       : "location_sk (INT) -> dim_location.location_sk",
        "fk_details"        : "listing_id (BIGINT) -> dim_listing_details.listing_id",
        "fk_photos"         : "listing_id (BIGINT) -> dim_listing_photos (photo_count denorm)",
        "fk_price_category" : "price_category_key (INT) -> dim_price_category.price_category_key",
        "fk_steering"       : "steering_key (INT) -> dim_steering.steering_key",
    },
)
def fact_listings():
    df = spark.table(f"{SILVER}.listings_silver_merged")

    # ── Photo count -- denormalized from Silver photo table ───────────────────
    photo_counts = (
        spark.table(f"{SILVER}.listings_photo_transformation")
        .withColumn("listing_id", _listing_id_bigint("listing_id"))
        .groupBy("listing_id")
        .agg(F.count("photo_url_clean").alias("photo_count"))
    )


    word_counts = (
        spark.table(f"{SILVER}.listings_text_transformation")
        .filter(F.col("listing_id").isNotNull())
        .withColumn("listing_id", _listing_id_bigint("listing_id"))
        .withColumn(
            "word_count",
            F.size(F.split(F.trim(F.coalesce(F.col("text"), F.lit(""))), r"\s+"))
        )
        .select("listing_id", "word_count")
    )

    # ── Integer FK resolution from lookup dims ────────────────────────────────
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
        
        .withColumn("listing_id",  _listing_id_bigint("listing_id"))
        .withColumn("car_sk",      _car_sk("brand", "model"))
        .withColumn("location_sk", _location_sk("city_prepositional"))
        .join(photo_counts,  on="listing_id",    how="left")
        .join(word_counts,   on="listing_id",    how="left")
        .join(price_cat_map, on="price_category", how="left")
        .join(steer_map,     on="steering_wheel", how="left")
        .withColumn("photo_count",  F.coalesce(F.col("photo_count"), F.lit(0)))
        .withColumn("word_count",   F.coalesce(F.col("word_count"),  F.lit(0)))
        .withColumn("listing_date", F.col("listing_date").cast("date"))
        .select(
            # ── Degenerate dimension (grain key -- string is correct here) ─────
          
            F.col("listing_id").cast("bigint"),     # BIGINT (degenerate dim -- grain)

            # ── FK -> dim_date.date_key ───────────────────────────────────────
            "listing_date",                        # DATE

            # ── FK -> dim_car.car_sk (INTEGER surrogate key) ─────────────────
           
            F.col("car_sk"),                       # INT

            # ── FK -> dim_location.location_sk (INTEGER surrogate key) ────────
            
            F.col("location_sk"),                  # INT

            # ── FK -> dim_price_category.price_category_key (INT) ─────────────
            F.col("price_category_key"),           # INT

            # ── FK -> dim_steering.steering_key (INT) ─────────────────────────
            F.col("steering_key"),                 # INT

            # ── Numeric listing attributes ────────────────────────────────────
            "manufacture_year",                    # INT
            "engine_power",                        # INT (HP)
            "mileage_km",                          # INT
            "has_license",                         # INT (0/1)
            "listing_year",                        # INT
            "listing_month",                       # INT
            "car_age_years",                       # INT (2023 - manufacture_year, Silver-computed)

            # ── Financial measures ────────────────────────────────────────────
            "price_rub",                           # DOUBLE
            "price_usd",                           # DOUBLE (price_rub / 82.5)

            # ── Gold-derived measures ─────────────────────────────────────────
            (F.year("listing_date") - F.col("manufacture_year"))
                .alias("car_age_at_listing"),      # INT

            F.when(F.col("mileage_km") > 100000, True)
                .otherwise(False)
                .alias("is_high_mileage"),         # BOOLEAN

            F.round(
                F.col("price_usd") / F.nullif(F.col("engine_power"), F.lit(0)), 2
            ).alias("price_per_hp_usd"),           # DOUBLE

            # ── Denormalized measures ─────────────────────────────────────────
            F.col("photo_count").cast("int"),      # INT (from photo table join)

            # word_count moved here from dim_listing_details (it is a metric,
            # not a dimension attribute -- per evaluator Vasu Bajaj)
            F.col("word_count").cast("int"),       # INT

            # ── RGB colour codes ──────────────────────────────────────────────
            F.col("color_r").cast("int").alias("color_r"),  # INT
            F.col("color_g").cast("int").alias("color_g"),  # INT
            F.col("color_b").cast("int").alias("color_b"),  # INT

            # ── Full audit chain: Bronze -> Silver -> Gold ────────────────────
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
# MAGIC
# MAGIC

# COMMAND ----------

# ── AGG 1: Monthly Sales Trend ────────────────────────────────────────────────

@dlt.table(
    name             = "agg_monthly_sales_trend",
    comment          = "Gold Agg: Listing volume and revenue by month, brand, price segment. "
                       "brand label resolved via dim_car join on car_sk. "
                       "price_category label resolved via dim_price_category join on price_category_key. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_monthly_trend():
    fact      = dlt.read("fact_listings")
    # Join dim_car (active rows only) to get the brand label from the integer car_sk FK
    dim_car   = (
        dlt.read("dim_car")
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "brand")
        .distinct()
    )
    price_cat = dlt.read("dim_price_category").select("price_category_key", "price_category")
    return (
        fact
        .join(dim_car,   on="car_sk",             how="left")
        .join(price_cat, on="price_category_key",  how="left")
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


# ── AGG 2: Brand + Location Performance ───────────────────────────────────────
@dlt.table(
    name             = "agg_brand_location_performance",
    comment          = "Gold Agg: Brand performance by region. "
                       "brand label from dim_car (car_sk FK). "
                       "city_name label from dim_location (location_sk FK). "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_brand_performance():
    fact     = dlt.read("fact_listings")
    dim_car  = (
        dlt.read("dim_car")
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "brand")
        .distinct()
    )
    dim_loc  = (
        dlt.read("dim_location")
        .filter(F.col("__END_AT").isNull())
        .select("location_sk", "city_name")
        .distinct()
    )
    return (
        fact
        .join(dim_car, on="car_sk",      how="left")
        .join(dim_loc, on="location_sk", how="left")
        .groupBy("brand", "city_name")
        .agg(
            F.count("listing_id").alias("listing_count"),
            F.round(F.avg("price_usd"),   0).alias("avg_price_usd"),
            F.round(F.avg("mileage_km"),  0).alias("avg_mileage_km"),
            F.max("engine_power").alias("max_hp_in_region"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )


# ── AGG 3: Regional Market Depth ──────────────────────────────────────────────

@dlt.table(
    name             = "agg_regional_market_depth",
    comment          = "Gold Agg: Inventory depth by city, fuel type, price segment. "
                       "city_name from dim_location (location_sk FK). "
                       "fuel_type from dim_car (car_sk FK, active rows). "
                       "price_category from dim_price_category. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_regional_depth():
    fact      = dlt.read("fact_listings")
    price_dim = dlt.read("dim_price_category").select("price_category_key", "price_category")
    # Resolve fuel_type via integer car_sk FK -- no string join on brand/model
    dim_car   = (
        dlt.read("dim_car")
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "fuel_type")
        .distinct()
    )
    dim_loc   = (
        dlt.read("dim_location")
        .filter(F.col("__END_AT").isNull())
        .select("location_sk", "city_name")
        .distinct()
    )
    return (
        fact
        .join(dim_car,   on="car_sk",            how="left")
        .join(dim_loc,   on="location_sk",        how="left")
        .join(price_dim, on="price_category_key", how="left")
        .groupBy("city_name", "fuel_type", "price_category")
        .agg(
            F.count("listing_id").alias("inventory_count"),
            F.round(F.avg("mileage_km"), 0).alias("avg_mileage"),
            F.round(F.avg("price_usd"),  0).alias("avg_price_usd"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )


# ── AGG 4: Comprehensive KPI Cube ─────────────────────────────────────────────

@dlt.table(
    name             = "agg_comprehensive_kpi_cube",
    comment          = "Gold Agg: Multi-dim KPI cube -- brand, model, year, segment, fuel, mileage. "
                       "brand/model/fuel_type resolved via dim_car (car_sk FK, active rows). "
                       "price_category resolved via dim_price_category. "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_kpi_cube():
    fact      = dlt.read("fact_listings")
    price_dim = dlt.read("dim_price_category").select("price_category_key", "price_category")
    dim_car   = (
        dlt.read("dim_car")
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "brand", "model", "fuel_type")
        .distinct()
    )
    return (
        fact
        .join(dim_car,   on="car_sk",            how="left")
        .join(price_dim, on="price_category_key", how="left")
        .groupBy(
            "brand", "model", "manufacture_year",
            "price_category", "fuel_type", "is_high_mileage",
        )
        .agg(
            F.count("listing_id").alias("listing_volume"),
            F.round(F.avg("price_usd"),          2).alias("avg_market_price_usd"),
            F.round(F.avg("car_age_at_listing"),  1).alias("avg_age_at_listing"),
            F.round(F.avg("mileage_km"),          0).alias("avg_mileage_km"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )


# ── AGG 5: Top 10 Brands by Total Market Value ────────────────────────────────

@dlt.table(
    name             = "agg_top_10_brands_by_spend",
    comment          = "Gold Agg: Top 10 brands by cumulative USD market value. "
                       "brand label resolved via dim_car (car_sk FK, active rows). "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_top_10_brands():
    fact    = dlt.read("fact_listings")
    dim_car = (
        dlt.read("dim_car")
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "brand")
        .distinct()
    )
    return (
        fact
        .join(dim_car, on="car_sk", how="left")
        .groupBy("brand")
        .agg(
            F.round(F.sum("price_usd"),  0).alias("total_market_value_usd"),
            F.count("listing_id").alias("total_listings"),
            F.round(F.avg("price_usd"),  0).alias("avg_price_usd"),
        )
        .orderBy(F.desc("total_market_value_usd"))
        .limit(10)
        .withColumn("gold_load_dt", F.current_timestamp())
    )
