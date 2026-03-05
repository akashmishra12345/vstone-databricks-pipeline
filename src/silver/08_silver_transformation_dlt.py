# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Delta Live Tables Pipeline
# MAGIC **Key Design:** All deduplication uses `dlt.apply_changes` (SCD Type 1) or
# MAGIC `dropDuplicates` on watermarked streams — no `ROW_NUMBER()` window functions
# MAGIC which are unsupported in Structured Streaming.
# MAGIC
# MAGIC ### Tables Produced:
# MAGIC | DLT Table | Source | Dedup Strategy |
# MAGIC |---|---|---|
# MAGIC | `listings_text_silver` | listings_text_bronze | apply_changes (SCD1) |
# MAGIC | `listings_text_quarantine` | listings_text_bronze | append |
# MAGIC | `listings_photo_silver` | listings_photo_bronze | apply_changes (SCD1) |
# MAGIC | `listings_photo_quarantine` | listings_photo_bronze | append |
# MAGIC | `car_catalog_silver` | car_catalog_bronze | apply_changes (SCD1) |
# MAGIC | `listings_catalog_quarantine` | car_catalog_bronze | append |
# MAGIC | `geography_silver` | geo_locations_bronze | apply_changes (SCD1) |
# MAGIC | `geography_quarantine` | geo_locations_bronze | append |
# MAGIC | `listings_silver_merged` | 4 bronze listing tables | apply_changes (SCD1) |
# MAGIC | `listings_main_quarantine` | listings_silver_merged | append |

# COMMAND ----------

import dlt
import pandas as pd
from pyspark.sql.functions import (
    pandas_udf, col, current_timestamp, lit, expr, when,
    round, date_format, upper, trim, coalesce, try_to_timestamp
)
from pyspark.sql.types import StringType

# ======================================================================================
# PIPELINE CONFIGURATION
# spark.conf.get used instead of widgets — widgets not supported in DLT context
# ======================================================================================
CATALOG  = spark.conf.get("pipeline.catalog",        "vstone_catalog")
BRONZE   = spark.conf.get("pipeline.bronze_schema",  "bronze")
SILVER   = spark.conf.get("pipeline.silver_schema",  "silver")

USD_RATE = 82.5

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

# ======================================================================================
# SHARED PANDAS UDFs
# ======================================================================================

@pandas_udf(StringType())
def standardize_text_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

@pandas_udf(StringType())
def standardize_string_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

@pandas_udf(StringType())
def clean_text_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

@pandas_udf(StringType())
def standardize_geo_pd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


# ======================================================================================
# SECTION 1: TEXT LAYER
# ======================================================================================

%md
## 1. Text Layer

@dlt.view(name="text_bronze_valid",
    comment="Valid text records: listing_id parseable from bronze")
def text_bronze_valid():
    return (
        spark.readStream
        .option("skipChangeCommits", "true")
        .table(f"{CATALOG}.{BRONZE}.listings_text_bronze")
        .withColumn("listing_id", col("id").cast("double").cast("long").cast("string"))
        .withColumn("source_file", lit("1_text.csv"))
        .withColumn("silver_load_dt", current_timestamp())
        .drop("id", "_c0", "_rescued_data")
        .filter(col("listing_id").isNotNull())
    )


@dlt.view(name="text_bronze_invalid",
    comment="Invalid text records: listing_id is null — routed to quarantine")
def text_bronze_invalid():
    return (
        spark.readStream
        .option("skipChangeCommits", "true")
        .table(f"{CATALOG}.{BRONZE}.listings_text_bronze")
        .withColumn("listing_id", col("id").cast("double").cast("long").cast("string"))
        .withColumn("source_file", lit("1_text.csv"))
        .withColumn("silver_load_dt", current_timestamp())
        .drop("id", "_c0", "_rescued_data")
        .filter(col("listing_id").isNull())
        .withColumn("quarantine_reason", lit("malformed_id_in_text_data"))
        .withColumn("quarantine_dt", current_timestamp())
    )


# Streaming target — apply_changes handles deduplication by listing_id (SCD Type 1)
dlt.create_streaming_table(
    name="listings_text_silver",
    comment="Silver: Cleaned and deduplicated text descriptions. Dedup key: listing_id.",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
        "pipelines.autoOptimize.zOrderCols": "listing_id"
    },
    expect_all={
        "valid_listing_id":    "listing_id IS NOT NULL",
        "source_file_present": "source_file IS NOT NULL",
        "load_dt_present":     "silver_load_dt IS NOT NULL"
    }
)

dlt.apply_changes(
    target="listings_text_silver",
    source="text_bronze_valid",
    keys=["listing_id"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=1
)


@dlt.table(
    name="listings_text_quarantine",
    comment="Silver Quarantine: Text records with missing listing_id.",
    table_properties={"quality": "quarantine"}
)
@dlt.expect("has_quarantine_reason", "quarantine_reason IS NOT NULL")
def listings_text_quarantine():
    return dlt.read_stream("text_bronze_invalid")


# ======================================================================================
# SECTION 2: PHOTO LAYER
# ======================================================================================

%md
## 2. Photo Layer

@dlt.view(name="photo_bronze_valid",
    comment="Valid photo records: listing_id parseable, URL standardized")
def photo_bronze_valid():
    return (
        spark.readStream
        .option("skipChangeCommits", "true")
        .table(f"{CATALOG}.{BRONZE}.listings_photo_bronze")
        .withColumn("listing_id", col("id").cast("double").cast("long").cast("string"))
        .withColumn("photo_url_clean", standardize_string_pd(col("photo_url")))
        .withColumn("source_file", lit("photos_main.csv"))
        .withColumn("silver_load_dt", current_timestamp())
        .drop("id", "_c0", "_rescued_data")
        .filter(col("listing_id").isNotNull())
    )


@dlt.view(name="photo_bronze_invalid",
    comment="Invalid photo records: listing_id null — routed to quarantine")
def photo_bronze_invalid():
    return (
        spark.readStream
        .option("skipChangeCommits", "true")
        .table(f"{CATALOG}.{BRONZE}.listings_photo_bronze")
        .withColumn("listing_id", col("id").cast("double").cast("long").cast("string"))
        .withColumn("source_file", lit("photos_main.csv"))
        .withColumn("silver_load_dt", current_timestamp())
        .drop("id", "_c0", "_rescued_data")
        .filter(col("listing_id").isNull())
        .withColumn("quarantine_reason", lit("malformed_id_in_photo_data"))
        .withColumn("quarantine_dt", current_timestamp())
    )


dlt.create_streaming_table(
    name="listings_photo_silver",
    comment="Silver: Deduplicated photo URLs. Dedup key: listing_id + photo_url_clean.",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
        "pipelines.autoOptimize.zOrderCols": "listing_id"
    },
    expect_all={
        "valid_listing_id":  "listing_id IS NOT NULL",
        "valid_photo_url":   "photo_url IS NOT NULL",
        "load_dt_present":   "silver_load_dt IS NOT NULL"
    }
)

dlt.apply_changes(
    target="listings_photo_silver",
    source="photo_bronze_valid",
    keys=["listing_id", "photo_url_clean"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=1
)


@dlt.table(
    name="listings_photo_quarantine",
    comment="Silver Quarantine: Photo records with missing listing_id.",
    table_properties={"quality": "quarantine"}
)
@dlt.expect("has_quarantine_reason", "quarantine_reason IS NOT NULL")
def listings_photo_quarantine():
    return dlt.read_stream("photo_bronze_invalid")


# ======================================================================================
# SECTION 3: CAR CATALOG LAYER
# ======================================================================================

%md
## 3. Car Catalog Layer

def _rename_catalog_columns(df):
    for rus_name, eng_name in FULL_CATALOG_MAP.items():
        if rus_name in df.columns:
            df = df.withColumnRenamed(rus_name, eng_name)
    return df


def _cast_catalog_columns(df):
    df = (df
        .withColumn("engine_volume_l",
            expr("try_cast(regexp_replace(regexp_replace(engine_volume_l, ' л', ''), ',', '.') as double)"))
        .withColumn("engine_power_hp",
            expr("try_cast(regexp_replace(engine_power_hp, ' л.с.', '') as int)"))
        .withColumn("clearance_mm",
            expr("try_cast(regexp_replace(clearance_mm, ' мм', '') as int)"))
        .withColumn("trunk_volume_l",
            expr("try_cast(regexp_replace(trunk_volume_l, ' л', '') as int)"))
        .withColumn("seats_count",
            expr("try_cast(regexp_replace(seats_count, ' мест', '') as int)"))
        .withColumn("acceleration_0_100",
            expr("try_cast(regexp_replace(acceleration_0_100, ',', '.') as double)"))
        .withColumn("max_speed_kmh",
            expr("try_cast(max_speed_kmh as int)"))
        .withColumn("source_file", lit("catalogs.csv"))
        .withColumn("silver_load_dt", current_timestamp())
    )
    for c in ["brand", "model", "generation", "trim_level",
              "fuel_type", "transmission", "drive_type", "body_type"]:
        if c in df.columns:
            df = df.withColumn(c, clean_text_pd(col(c)))
    df = df.withColumn("brand",
        when(col("brand").isin("None", "nan", "null"), None).otherwise(col("brand")))
    return df


@dlt.view(name="catalog_bronze_valid",
    comment="Valid catalog records: brand not null after rename + cast")
def catalog_bronze_valid():
    df = spark.readStream.option("skipChangeCommits", "true").table(
        f"{CATALOG}.{BRONZE}.car_catalog_bronze")
    df = _rename_catalog_columns(df)
    df = _cast_catalog_columns(df)
    return df.filter(col("brand").isNotNull())


@dlt.view(name="catalog_bronze_invalid",
    comment="Invalid catalog records: brand null — routed to quarantine")
def catalog_bronze_invalid():
    df = spark.readStream.option("skipChangeCommits", "true").table(
        f"{CATALOG}.{BRONZE}.car_catalog_bronze")
    df = _rename_catalog_columns(df)
    df = _cast_catalog_columns(df)
    return (df
        .filter(col("brand").isNull())
        .withColumn("quarantine_reason", lit("missing_brand_or_invalid_data"))
        .withColumn("quarantine_dt", current_timestamp())
    )


dlt.create_streaming_table(
    name="car_catalog_silver",
    comment="Silver: Car catalog with Russian headers translated, numeric cols cast. Dedup: brand+model+generation+trim.",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
        "pipelines.autoOptimize.zOrderCols": "brand,model"
    },
    expect_all={
        "valid_brand":     "brand IS NOT NULL",
        "valid_model":     "model IS NOT NULL",
        "load_dt_present": "silver_load_dt IS NOT NULL"
    },
    expect_all_or_drop={
        "valid_engine_volume": "engine_volume_l IS NULL OR engine_volume_l > 0",
        "valid_engine_power":  "engine_power_hp IS NULL OR engine_power_hp > 0"
    }
)

dlt.apply_changes(
    target="car_catalog_silver",
    source="catalog_bronze_valid",
    keys=["brand", "model", "generation", "trim_level",
          "engine_volume_l", "engine_power_hp"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=1
)


@dlt.table(
    name="listings_catalog_quarantine",
    comment="Silver Quarantine: Catalog records with missing/invalid brand.",
    table_properties={"quality": "quarantine"}
)
@dlt.expect("has_quarantine_reason", "quarantine_reason IS NOT NULL")
def listings_catalog_quarantine():
    return dlt.read_stream("catalog_bronze_invalid")


# ======================================================================================
# SECTION 4: GEOGRAPHY LAYER
# ======================================================================================

%md
## 4. Geography Layer

@dlt.view(name="geo_bronze_valid",
    comment="Valid geo records: coordinates within Russia bounding box (41-82N, 19-180E)")
def geo_bronze_valid():
    df = (spark.readStream
        .option("skipChangeCommits", "true")
        .table(f"{CATALOG}.{BRONZE}.geo_locations_bronze")
        .select(
            col("name_padesh").alias("city_name"),
            col("greate_padesh").alias("city_prepositional"),
            col("lat").cast("double").alias("latitude"),
            col("lon").cast("double").alias("longitude"),
            lit("geography.csv").alias("source_file"),
            current_timestamp().alias("silver_load_dt")
        )
        .withColumn("city_name", standardize_geo_pd(col("city_name")))
    )
    return df.filter(
        col("latitude").isNotNull() & col("longitude").isNotNull() &
        col("latitude").between(41, 82) & col("longitude").between(19, 180)
    )


@dlt.view(name="geo_bronze_invalid",
    comment="Invalid geo records: null or out-of-Russia coordinates")
def geo_bronze_invalid():
    df = (spark.readStream
        .option("skipChangeCommits", "true")
        .table(f"{CATALOG}.{BRONZE}.geo_locations_bronze")
        .select(
            col("name_padesh").alias("city_name"),
            col("greate_padesh").alias("city_prepositional"),
            col("lat").cast("double").alias("latitude"),
            col("lon").cast("double").alias("longitude"),
            lit("geography.csv").alias("source_file"),
            current_timestamp().alias("silver_load_dt")
        )
        .withColumn("city_name", standardize_geo_pd(col("city_name")))
    )
    is_invalid = (
        col("latitude").isNull()  | col("longitude").isNull() |
        ~col("latitude").between(41, 82) | ~col("longitude").between(19, 180)
    )
    return (df
        .filter(is_invalid)
        .withColumn("quarantine_reason", lit("coordinates_outside_russia_or_null"))
        .withColumn("quarantine_dt", current_timestamp())
    )


dlt.create_streaming_table(
    name="geography_silver",
    comment="Silver: Validated Russian city coordinates, deduplicated by city_prepositional.",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
        "pipelines.autoOptimize.zOrderCols": "city_prepositional"
    },
    expect_all={
        "valid_city_name":          "city_name IS NOT NULL",
        "valid_city_prepositional": "city_prepositional IS NOT NULL",
        "load_dt_present":          "silver_load_dt IS NOT NULL"
    },
    expect_all_or_drop={
        "valid_lat": "latitude BETWEEN 41 AND 82",
        "valid_lon": "longitude BETWEEN 19 AND 180"
    }
)

dlt.apply_changes(
    target="geography_silver",
    source="geo_bronze_valid",
    keys=["city_prepositional"],
    sequence_by=col("silver_load_dt"),
    stored_as_scd_type=1
)


@dlt.table(
    name="geography_quarantine",
    comment="Silver Quarantine: Geo records outside Russia bounding box or with null coordinates.",
    table_properties={"quality": "quarantine"}
)
@dlt.expect("has_quarantine_reason", "quarantine_reason IS NOT NULL")
def geography_quarantine():
    return dlt.read_stream("geo_bronze_invalid")


# ======================================================================================
# SECTION 5: MAIN LISTINGS — UNION + TRANSFORM + ENRICH
# ======================================================================================

%md
## 5. Main Listings — Union, Transform & Enrich

@dlt.view(
    name="listings_all_bronze_unioned",
    comment="Union of all 4 bronze listing sources"
)
def listings_all_bronze_unioned():
    df_csv     = spark.readStream.option("skipChangeCommits","true").table(f"{CATALOG}.{BRONZE}.listings_csv_copyinto")
    df_json    = spark.readStream.option("skipChangeCommits","true").table(f"{CATALOG}.{BRONZE}.listings_json_autoloader")
    df_xml     = spark.readStream.option("skipChangeCommits","true").table(f"{CATALOG}.{BRONZE}.listings_xml_pyspark")
    df_csv_dlt = spark.readStream.option("skipChangeCommits","true").table(f"{CATALOG}.{BRONZE}.listings_csv_dlt")

    return (df_csv
        .unionByName(df_json,    allowMissingColumns=True)
        .unionByName(df_xml,     allowMissingColumns=True)
        .unionByName(df_csv_dlt, allowMissingColumns=True)
    )


@dlt.view(
    name="listings_transformed_valid",
    comment="Valid listings: cast + enriched. Dedup handled by apply_changes downstream."
)
def listings_transformed_valid():
    df = dlt.read_stream("listings_all_bronze_unioned")

    df_prep = df.select(
        expr("try_cast(id as long)").cast("string").alias("listing_id"),
        coalesce(
            try_to_timestamp(col("date"), lit("dd.MM.yyyy")),
            try_to_timestamp(col("date"), lit("yyyy-MM-dd'T'HH:mm:ss'Z'"))
        ).alias("listing_date"),
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

    df_enriched = (df_prep
        .withColumn("price_usd",     round(col("price_rub") / USD_RATE, 2))
        .withColumn("car_age_years", lit(2023) - col("manufacture_year").cast("integer"))
        .withColumn("price_category",
            when(col("price_rub") < 300000,                lit("BUDGET"))
            .when(col("price_rub").between(300000, 700000),  lit("MID_RANGE"))
            .when(col("price_rub").between(700001, 1500000), lit("PREMIUM"))
            .when(col("price_rub") > 1500000,               lit("LUXURY"))
            .otherwise(lit("UNKNOWN"))
        )
        .withColumn("listing_year",   date_format(col("listing_date"), "yyyy").cast("integer"))
        .withColumn("listing_month",  date_format(col("listing_date"), "MM").cast("integer"))
        .withColumn("brand_std",      upper(trim(col("brand"))))
        .withColumn("silver_load_dt", current_timestamp())
    )

    return df_enriched.filter(
        col("listing_id").isNotNull() &
        col("price_rub").isNotNull()  &
        col("listing_date").isNotNull()
    )


@dlt.view(
    name="listings_transformed_invalid",
    comment="Invalid listings: failed ID/price/date checks — routed to quarantine"
)
def listings_transformed_invalid():
    df = dlt.read_stream("listings_all_bronze_unioned")

    df_prep = df.select(
        expr("try_cast(id as long)").cast("string").alias("listing_id"),
        coalesce(
            try_to_timestamp(col("date"), lit("dd.MM.yyyy")),
            try_to_timestamp(col("date"), lit("yyyy-MM-dd'T'HH:mm:ss'Z'"))
        ).alias("listing_date"),
        expr("try_cast(regexp_replace(cost, '[^0-9.]', '') as double)").alias("price_rub"),
        col("currency"),
        standardize_text_pd(col("marka")).alias("brand"),
        standardize_text_pd(col("model")).alias("model"),
        expr("try_cast(try_cast(year as double) as int)").alias("manufacture_year"),
        col("source_file"),
        col("load_dt").alias("bronze_load_dt"),
        current_timestamp().alias("silver_load_dt")
    )

    is_invalid = (
        col("listing_id").isNull() |
        col("price_rub").isNull()  |
        col("listing_date").isNull()
    )

    return (df_prep
        .filter(is_invalid)
        .withColumn("quarantine_reason",
            when(col("listing_id").isNull(),   lit("MISSING_OR_MALFORMED_ID"))
            .when(col("price_rub").isNull(),    lit("INVALID_PRICE_FORMAT"))
            .when(col("listing_date").isNull(), lit("UNPARSABLE_DATE_FORMAT"))
            .otherwise(lit("DATA_QUALITY_ISSUE"))
        )
        .withColumn("quarantine_dt", current_timestamp())
    )


dlt.create_streaming_table(
    name="listings_silver_merged",
    comment="Silver: Unified enriched car listings from all 4 bronze sources. Dedup key: listing_id.",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
        "pipelines.autoOptimize.zOrderCols": "listing_id,brand_std"
    },
    expect_all={
        "listing_id_not_null":    "listing_id IS NOT NULL",
        "price_usd_positive":     "price_usd > 0",
        "silver_load_present":    "silver_load_dt IS NOT NULL",
        "valid_manufacture_year": "manufacture_year IS NULL OR manufacture_year BETWEEN 1900 AND 2025",
        "reasonable_mileage":     "mileage_km IS NULL OR mileage_km BETWEEN 0 AND 1000000",
        "known_price_category":   "price_category != 'UNKNOWN'",
        "valid_car_age":          "car_age_years IS NULL OR car_age_years BETWEEN 0 AND 100"
    }
)

dlt.apply_changes(
    target="listings_silver_merged",
    source="listings_transformed_valid",
    keys=["listing_id"],
    sequence_by=col("bronze_load_dt"),
    stored_as_scd_type=1
)


@dlt.table(
    name="listings_main_quarantine",
    comment="Silver Quarantine: Listings that failed ID, price or date parsing.",
    table_properties={"quality": "quarantine"}
)
@dlt.expect("has_quarantine_reason", "quarantine_reason IS NOT NULL")
@dlt.expect("has_quarantine_dt",     "quarantine_dt IS NOT NULL")
def listings_main_quarantine():
    return dlt.read_stream("listings_transformed_invalid")
