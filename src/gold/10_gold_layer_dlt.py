# Databricks notebook source
# MAGIC %md
# MAGIC # 10 -- Gold Layer DLT Pipeline
# MAGIC star schema built from Silver tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & Configuration

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = spark.conf.get("pipeline.catalog",       "vstone_catalog")
SILVER  = f"{CATALOG}.silver"

GOLD_PROPS = {
    "quality"                   : "gold",
    "delta.enableChangeDataFeed": "true",
    "pipelines.reset.allowed"   : "true",
}

# ── Stable surrogate key helpers ──────────────────────────────────────────────

def _car_sk(brand_col="brand", model_col="model"):
    """Stable INT surrogate key for dim_car. crc32(lower(brand)|lower(model))."""
    return F.crc32(
        F.concat_ws("|",
            F.lower(F.trim(F.col(brand_col))),
            F.lower(F.trim(F.col(model_col)))
        )
    ).cast("int")


def _location_sk(city_col="city_prepositional"):
    """Stable INT surrogate key for dim_location. crc32(lower(city_prepositional))."""
    return F.crc32(F.lower(F.trim(F.col(city_col)))).cast("int")


print(f"Catalog: {CATALOG} | Silver: {SILVER}")
print("Surrogate key helpers: _car_sk | _location_sk")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `dim_date` -- Static Calendar Dimension (2010-2030)

# COMMAND ----------

@dlt.table(
    name             = "dim_date",
    comment          = "Gold Static: Gap-free calendar dimension 2010-2030. "
                       "No SCD -- calendar never changes. " 
                       "PK: date_key (DATE). "
                       "FK target: fact_listings.listing_date -> dim_date.date_key.",
    table_properties = {**GOLD_PROPS, "type": "static", "pk": "date_key"},
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
# MAGIC ## `dim_price_category` + `dim_steering` -- Static Lookup Dimensions

# COMMAND ----------

@dlt.table(
    name             = "dim_price_category",
    comment          = "Gold Static Lookup: Price band codes. "
                       "5 rows -- derived from Silver _transform_listings price_category CASE bands. "
                       "PK: price_category_key INT. "
                       "FK target: fact_listings.price_category_key -> dim_price_category.price_category_key.",
    table_properties = {**GOLD_PROPS, "type": "lookup"},
)
def dim_price_category():
    # Bands match Silver _transform_listings exactly:
    # < 300,000 RUB = BUDGET | 300k-700k = MID_RANGE | 700k-1.5M = PREMIUM | >1.5M = LUXURY
    data = [
        (1, "BUDGET",    "< 300,000 RUB"),
        (2, "MID_RANGE", "300,000 - 700,000 RUB"),
        (3, "PREMIUM",   "700,001 - 1,500,000 RUB"),
        (4, "LUXURY",    "> 1,500,000 RUB"),
        (5, "UNKNOWN",   "Unclassified (null or gap price)"),
    ]
    return (
        spark.createDataFrame(data, "price_category_key INT, price_category STRING, price_range_desc STRING")
        .withColumn("gold_load_dt", F.current_timestamp())
    )


@dlt.table(
    name             = "dim_steering",
    comment          = "Gold Static Lookup: Steering wheel side codes. "
                       "Source: listings_silver_merged.steering_wheel "
                       "(Bronze sWheel -> Silver standardize_text -> lower+stripped string). "
                       "PK: steering_key INT (dense_rank over steering_wheel). "
                       "FK target: fact_listings.steering_key -> dim_steering.steering_key.",
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
# MAGIC
# MAGIC Source: `car_catalog_transformation`
# MAGIC
# MAGIC Surrogate key `car_sk` = `crc32(lower(brand)|lower(model))` cast to INT.
# MAGIC Stable and reproducible -- same brand+model always yields the same INT.

# COMMAND ----------

@dlt.view(name="dim_car_source")
def dim_car_source():
    """
    Streaming view: enriches car_catalog_transformation with car_sk.
    MUST use spark.readStream -- apply_changes() requires a streaming source.
    car_sk = crc32(lower(brand)|lower(model)) INT.
    brand and model from Silver are already lower+stripped (clean_text UDF).
    """
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.car_catalog_transformation")
        .withColumn("car_sk", _car_sk("brand", "model"))
    )


dlt.create_streaming_table(
    name             = "dim_car",
    comment          = "Gold SCD2: Car specifications from car_catalog_transformation. "
                       "Natural key: brand + model (composite). "
                       "Surrogate PK: car_sk INT = crc32(lower(brand)|lower(model)). "
                       "FK target: fact_listings.car_sk -> dim_car.car_sk. "
                       "SCD2: __START_AT / __END_AT track historical changes. "
                       "Active rows: __END_AT IS NULL.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "car_sk"},
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
# MAGIC
# MAGIC Source: `geography_transformation`

# COMMAND ----------

@dlt.view(name="dim_location_source")
def dim_location_source():
    """
    Streaming view: enriches geography_transformation with location_sk.
    MUST use spark.readStream -- apply_changes() requires a streaming source.
    location_sk = crc32(lower(city_prepositional)) INT.
    city_prepositional from Silver = greate_padesh (pass-through from Bronze).
    """
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.geography_transformation")
        .withColumn("location_sk", _location_sk("city_prepositional"))
    )


dlt.create_streaming_table(
    name             = "dim_location",
    comment          = "Gold SCD2: Russian city/region from geography_transformation. "
                       "Natural key: city_prepositional (string). "
                       "Surrogate PK: location_sk INT = crc32(lower(city_prepositional)). "
                       "FK target: fact_listings.location_sk -> dim_location.location_sk. "
                       "SCD2: __START_AT / __END_AT. Active rows: __END_AT IS NULL.",
    table_properties = {**GOLD_PROPS, "scd_type": "2", "pk": "location_sk"},
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
# MAGIC
# MAGIC Source: `listings_text_transformation`

# COMMAND ----------

@dlt.view(name="dim_listing_details_source")
def dim_listing_details_source():
    """
    Streaming view: adds text_hash for SCD2 change detection.
    MUST use spark.readStream -- apply_changes() requires streaming source.
    text_hash = md5(text) -- : SCD2 needs a change-detection column.
    If text changes for the same listing_id, text_hash changes -> new SCD2 version.
    """
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.listings_text_transformation")
        .withColumn("text_hash", F.md5(F.coalesce(F.col("text"), F.lit(""))))
    )


dlt.create_streaming_table(
    name             = "dim_listing_details",
    comment          = "Gold SCD2: Russian listing text descriptions from listings_text_transformation. "
                       "Natural PK: listing_id"
                       "Change-detection: text_hash (MD5 of text) -- Vasu feedback: SCD2 requires "
                       "a change column; text_hash triggers a new version when description changes. "
                       "SCD2: __START_AT / __END_AT. Active rows: __END_AT IS NULL.",
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
# MAGIC Source: `listings_photo_transformation`

# COMMAND ----------

@dlt.view(name="dim_listing_photos_source")
def dim_listing_photos_source():
    """
    Streaming view for listings_photo_transformation.
    MUST use spark.readStream -- apply_changes() requires streaming source.
    Composite key: listing_id + photo_url_clean (one listing -> many photos).
    """
    return (
        spark.readStream.format("delta")
        .table(f"{SILVER}.listings_photo_transformation")
    )


dlt.create_streaming_table(
    name             = "dim_listing_photos",
    comment          = "Gold SCD2: Photo URLs per listing from listings_photo_transformation. "
                       "Composite PK: listing_id  + photo_url_clean STRING. "
                       "listing_id "
                       "One-to-many: one listing can have many photos. "
                       "photo_count denormalized into fact_listings via groupBy join. "
                       "SCD2: __START_AT / __END_AT. Active rows: __END_AT IS NULL.",
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
# MAGIC ## `fact_listings` --  Fact Table
# MAGIC

# COMMAND ----------

@dlt.table(
    name             = "fact_listings",
    comment          = "Gold Fact Table: Car listings -- Kimball star schema. "
                       "GRAIN: one row per listing_id (STRING degenerate dimension). "
                       "NO STRINGS except listing_id per Vasu Bajaj design rules. "
                       "INTEGER FKs: car_sk->dim_car, location_sk->dim_location, "
                       "price_category_key->dim_price_category, steering_key->dim_steering. "
                       "DATE FK: listing_date->dim_date.date_key. "
                       "NO TYPE CASTING IN GOLD: Silver produces correct types directly. "
                       "color_r/g/b: Silver try_cast(R as int) -> INT. No cast in Gold. "
                       "MEASURES: price_rub, price_usd, mileage_km, engine_power, "
                       "car_age_at_listing, is_high_mileage, price_per_hp_usd, "
                       "photo_count, word_count (metric moved from dim per Vasu). "
                       "AUDIT: bronze_load_dt -> silver_load_dt -> gold_load_dt.",
    table_properties = {
        **GOLD_PROPS,
        "type"              : "fact",
        "grain"             : "listing_id ",
        "fk_date"           : "listing_date -> dim_date.date_key",
        "fk_car"            : "car_sk (INT) -> dim_car.car_sk",
        "fk_location"       : "location_sk (INT) -> dim_location.location_sk",
        "fk_details"        : "listing_id  -> dim_listing_details.listing_id",
        "fk_photos"         : "listing_id  -> dim_listing_photos (photo_count denorm)",
        "fk_price_category" : "price_category_key (INT) -> dim_price_category",
        "fk_steering"       : "steering_key (INT) -> dim_steering",
    },
)
def fact_listings():
    df = spark.table(f"{SILVER}.listings_silver_merged")

    # ── photo_count: count of photos per listing (from Silver photo table) ────
    
    photo_counts = (
        spark.table(f"{SILVER}.listings_photo_transformation")
        .filter(F.col("listing_id").isNotNull())
        .groupBy("listing_id")
        .agg(F.count("photo_url_clean").alias("photo_count"))
    )

    # ── word_count: metric from Silver text table 
   
    word_counts = (
        spark.table(f"{SILVER}.listings_text_transformation")
        .filter(F.col("listing_id").isNotNull())
        .withColumn(
            "word_count",
            F.size(F.split(F.trim(F.coalesce(F.col("text"), F.lit(""))), r"\s+"))
        )
        .select("listing_id", "word_count")
    )

    # ── Integer FK maps: resolve STRING labels -> INT keys ────────────────────
    
    price_cat_map = (
        dlt.read("dim_price_category")
        .select("price_category", "price_category_key")
    )
    # steering_key: Silver produces STRING steering_wheel (lower+stripped).
    
    steer_map = (
        dlt.read("dim_steering")
        .select("steering_wheel", "steering_key")
    )

    return (
        df
        # ── Compute integer surrogate FKs from Silver string columns ──────────
        
        .withColumn("car_sk",      _car_sk("brand", "model"))
        .withColumn("location_sk", _location_sk("city_prepositional"))

        # ── Join photo and word count metrics ─────────────────────────────────
        .join(photo_counts,  on="listing_id",     how="left")
        .join(word_counts,   on="listing_id",     how="left")

        # ── Join to get integer FKs for price_category and steering ──────────
        .join(price_cat_map, on="price_category",  how="left")
        .join(steer_map,     on="steering_wheel",  how="left")

        # ── Default nulls for denormalized counts ─────────────────────────────
        .withColumn("photo_count", F.coalesce(F.col("photo_count"), F.lit(0)))
        .withColumn("word_count",  F.coalesce(F.col("word_count"),  F.lit(0)))

        # ── Cast listing_date TIMESTAMP -> DATE ───────────────────────────────
        # Silver produces TIMESTAMP -- fact date FK should be DATE.
        .withColumn("listing_date", F.col("listing_date").cast("date"))

        .select(
            # ── Grain key (degenerate dimension) ──────────────────────────────
            
            "listing_id",                           

            # ── FK -> dim_date.date_key ───────────────────────────────────────
            "listing_date",                         # DATE

            # ── FK -> dim_car.car_sk (INT surrogate key) ─────────────────────
            
            F.col("car_sk"),                        # INT

            # ── FK -> dim_location.location_sk (INT surrogate key) ────────────
    
            F.col("location_sk"),                   # INT

            # ── FK -> dim_price_category.price_category_key (INT) ─────────────
            F.col("price_category_key"),            # INT

            # ── FK -> dim_steering.steering_key (INT) ─────────────────────────
            F.col("steering_key"),                  # INT

            # ── Numeric listing attributes  ───
            "manufacture_year",                     # INT 
            "engine_power",                         # INT
            "mileage_km",                           # INT
            "has_license",                          # INT
            "listing_year",                         # INT
            "listing_month",                        # INT
            "car_age_years",                        # INT 

            # ── Financial measures ────────────────────────────────────────────
            "price_rub",                            # DOUBLE 
            "price_usd",                            # DOUBLE 

            # ── Gold-derived measures ─────────────────────────────────────────
            (F.year("listing_date") - F.col("manufacture_year"))
                .alias("car_age_at_listing"),       # INT (listing year - manufacture year)

            F.when(F.col("mileage_km") > 100000, True)
             .otherwise(False)
             .alias("is_high_mileage"),             # BOOLEAN

            F.round(
                F.col("price_usd") / F.nullif(F.col("engine_power"), F.lit(0)), 2
            ).alias("price_per_hp_usd"),            # DOUBLE

            # ── Denormalized count metrics ─────────────────────────────────────
            F.col("photo_count").cast("int"),       # INT (joined from photo table)
            F.col("word_count").cast("int"),        # INT 

            # ── RGB colour codes ────────────
         
            "color_r",                              # INT
            "color_g",                              # INT
            "color_b",                              # INT

            # ── Full audit chain: Bronze -> Silver -> Gold ────────────────────
            "bronze_load_dt",                       # TIMESTAMP
            "bronze_source_file",                   # STRING
            "silver_load_dt",                       # TIMESTAMP
            F.current_timestamp().alias("gold_load_dt"),  # TIMESTAMP
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Tables
# MAGIC
# MAGIC All aggregates join back to dim tables via INTEGER FKs -- no string columns
# MAGIC in fact used for grouping. Brand, fuel_type, city_name are resolved by joining
# MAGIC dims on integer keys at query time.

# COMMAND ----------

# ── AGG 1: Monthly Sales Trend ────────────────────────────────────────────────
@dlt.table(
    name             = "agg_monthly_sales_trend",
    comment          = "Gold Agg: Listing volume and revenue by month, brand, price segment. "
                       "brand resolved via dim_car join on car_sk (INT). "
                       "price_category resolved via dim_price_category join on price_category_key (INT). "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_monthly_trend():
    fact      = dlt.read("fact_listings")
    dim_car   = (
        dlt.read("dim_car")
        .filter(F.col("__END_AT").isNull())        # active SCD2 rows only
        .select("car_sk", "brand")
        .distinct()
    )
    price_cat = dlt.read("dim_price_category").select("price_category_key", "price_category")
    return (
        fact
        .join(dim_car,   on="car_sk",            how="left")
        .join(price_cat, on="price_category_key", how="left")
        .filter(F.col("brand").isNotNull())       
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
                       "brand from dim_car (car_sk INT FK). "
                       "city_name from dim_location (location_sk INT FK). "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_brand_performance():
    fact    = dlt.read("fact_listings")
    dim_car = (
        dlt.read("dim_car")
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "brand")
        .distinct()
    )
    dim_loc = (
        dlt.read("dim_location")
        .filter(F.col("__END_AT").isNull())
        .select("location_sk", "city_name")
        .distinct()
    )
    return (
        fact
        .join(dim_car, on="car_sk",      how="left")
        .join(dim_loc, on="location_sk", how="left")
        .filter(F.col("brand").isNotNull())      
        .groupBy("brand", "city_name")
        .agg(
            F.count("listing_id").alias("listing_count"),
            F.round(F.avg("price_usd"),  0).alias("avg_price_usd"),
            F.round(F.avg("mileage_km"), 0).alias("avg_mileage_km"),
            F.max("engine_power").alias("max_hp_in_region"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )


# ── AGG 3: Regional Market Depth ──────────────────────────────────────────────
@dlt.table(
    name             = "agg_regional_market_depth",
    comment          = "Gold Agg: Inventory depth by city, fuel type, price segment. "
                       "city_name from dim_location (location_sk INT FK). "
                       "fuel_type from dim_car (car_sk INT FK, active rows). "
                       "price_category from dim_price_category (price_category_key INT FK). "
                       "Audit: gold_load_dt.",
    table_properties = {**GOLD_PROPS, "type": "aggregate"},
)
def agg_regional_depth():
    fact      = dlt.read("fact_listings")
    price_dim = dlt.read("dim_price_category").select("price_category_key", "price_category")
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
    comment          = "Gold Agg: Multi-dimensional KPI cube. "
                       "brand/model/fuel_type from dim_car (car_sk INT FK, active rows). "
                       "price_category from dim_price_category (price_category_key INT FK). "
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
        .filter(F.col("brand").isNotNull())       
        .groupBy("brand", "model", "manufacture_year", "price_category", "fuel_type", "is_high_mileage")
        .agg(
            F.count("listing_id").alias("listing_volume"),
            F.round(F.avg("price_usd"),         2).alias("avg_market_price_usd"),
            F.round(F.avg("car_age_at_listing"), 1).alias("avg_age_at_listing"),
            F.round(F.avg("mileage_km"),         0).alias("avg_mileage_km"),
        )
        .withColumn("gold_load_dt", F.current_timestamp())
    )


# ── AGG 5: Top 10 Brands by Total Market Value ────────────────────────────────
@dlt.table(
    name             = "agg_top_10_brands_by_spend",
    comment          = "Gold Agg: Top 10 brands by total USD market value. "
                       "brand resolved via dim_car (car_sk INT FK, active rows). "
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
        .filter(F.col("brand").isNotNull())       
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
