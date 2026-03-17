# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer Test Suite

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

@pytest.fixture(scope="session")
def spark():
    return DatabricksSession.builder.getOrCreate()


CATALOG = "vstone_catalog"
GOLD    = f"{CATALOG}.gold"
SILVER  = f"{CATALOG}.silver"

G = lambda t: f"{GOLD}.{t}"
S = lambda t: f"{SILVER}.{t}"

DIM_DATE_EXPECTED_ROWS = 7670

# All Gold tables that must exist
ALL_GOLD_TABLES = [
    "dim_date", "dim_price_category", "dim_steering",
    "dim_car", "dim_location", "dim_listing_details", "dim_listing_photos",
    "fact_listings",
    "agg_monthly_sales_trend", "agg_brand_location_performance",
    "agg_regional_market_depth", "agg_comprehensive_kpi_cube",
    "agg_top_10_brands_by_spend",
]

# SCD2 dim tables with their Silver source and natural keys
SCD2_REGISTRY = [
    {
        "name"        : "dim_car",
        "silver"      : S("car_catalog_transformation"),
        "natural_keys": ["brand", "model"],
    },
    {
        "name"        : "dim_location",
        "silver"      : S("geography_transformation"),
        "natural_keys": ["city_prepositional"],
    },
    {
        "name"        : "dim_listing_details",
        "silver"      : S("listings_text_transformation"),
        "natural_keys": ["listing_id"],
    },
    {
        "name"        : "dim_listing_photos",
        "silver"      : S("listings_photo_transformation"),
        "natural_keys": ["listing_id", "photo_url_clean"],
    },
]

SCD2_PARAMS = [pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY]

AGG_TABLES = [
    "agg_monthly_sales_trend",
    "agg_brand_location_performance",
    "agg_regional_market_depth",
    "agg_comprehensive_kpi_cube",
    "agg_top_10_brands_by_spend",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## U1 -- Unit Tests: Table Existence & Non-Empty
# MAGIC
# MAGIC Every Gold table must exist and contain at least one row.

# COMMAND ----------

@pytest.mark.parametrize("table", ALL_GOLD_TABLES)
def test_u1_gold_table_exists(spark, table):
    """U1 -- Every Gold table must exist in the catalog."""
    assert spark.catalog.tableExists(G(table)), (
        f"Gold table not found: {G(table)}. Run the Gold DLT pipeline first."
    )


@pytest.mark.parametrize("table", ALL_GOLD_TABLES)
def test_u1_gold_table_non_empty(spark, table):
    """U1 -- Every Gold table must contain at least one row."""
    count = spark.read.table(G(table)).count()
    assert count > 0, f"Gold table is empty: {G(table)}."


def test_u1_fact_listings_has_correct_columns(spark):
    """U1 -- fact_listings must have all expected columns."""
    df      = spark.read.table(G("fact_listings"))
    expected = [
        "listing_id", "listing_date", "car_sk", "location_sk",
        "price_category_key", "steering_key",
        "manufacture_year", "engine_power", "mileage_km", "has_license",
        "listing_year", "listing_month", "car_age_years",
        "price_rub", "price_usd",
        "car_age_at_listing", "is_high_mileage", "price_per_hp_usd",
        "photo_count", "word_count",
        "color_r", "color_g", "color_b",
        "bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt",
    ]
    missing = [c for c in expected if c not in df.columns]
    assert missing == [], f"fact_listings missing columns: {missing}"


def test_u1_fact_no_string_attribute_columns(spark):
   
    cols    = spark.read.table(G("fact_listings")).columns
    banned  = ["brand", "model", "city_prepositional", "city_name",
               "steering_wheel", "price_category", "fuel_type"]
    present = [c for c in banned if c in cols]
    assert present == [], (
        f"fact_listings contains raw string attribute columns: {present}. "
        " replace strings with integer surrogate keys. "
        "brand+model -> car_sk INT, city_prepositional -> location_sk INT, "
        "steering_wheel -> steering_key INT, price_category -> price_category_key INT."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U2 -- Unit Tests: Fact Column Data Types
# MAGIC
# MAGIC All FK columns must be INT, color_r/g/b must be INT, listing_id must be bigint, listing_date must be DATE.

# COMMAND ----------

def test_u2_fact_column_types_correct(spark):
   
    dtypes = dict(spark.read.table(G("fact_listings")).dtypes)

    expected = {
        "listing_id"          : "string",
        "listing_date"        : "date",
        "car_sk"              : "int",
        "location_sk"         : "int",
        "price_category_key"  : "int",
        "steering_key"        : "int",
        "manufacture_year"    : "int",
        "engine_power"        : "int",
        "mileage_km"          : "int",
        "has_license"         : "int",
        "listing_year"        : "int",
        "listing_month"       : "int",
        "car_age_years"       : "int",
        "price_rub"           : "double",
        "price_usd"           : "double",
        "car_age_at_listing"  : "int",
        "is_high_mileage"     : "boolean",
        "price_per_hp_usd"    : "double",
        "photo_count"         : "int",
        "word_count"          : "int",
        "color_r"             : "int",
        "color_g"             : "int",
        "color_b"             : "int",
    }

    wrong = [
        (col, exp, dtypes.get(col, "MISSING"))
        for col, exp in expected.items()
        if col in dtypes and not dtypes[col].startswith(exp)
    ]
    assert wrong == [], (
        f"fact_listings wrong column types: "
        f"{[(c, f'expected={e}', f'actual={a}') for c, e, a in wrong]}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U3 -- Unit Tests: FK Columns Non-Null
# MAGIC
# MAGIC Integer FK columns in fact_listings must never be null.
# MAGIC A null FK means the surrogate key computation or the dim join failed.

# COMMAND ----------

def test_u3_fact_fk_columns_non_null(spark):
    df = spark.read.table(G("fact_listings"))

    # These MUST be non-null -- CRC32 always returns INT from non-null input
    for col in ("car_sk", "location_sk"):
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, (
            f"fact_listings.{col} has {nulls:,} NULL rows. "
            f"Surrogate key computation should always produce a non-null INT."
        )

    # listing_id is the grain -- must never be null
    nulls = df.filter(F.col("listing_id").isNull()).count()
    assert nulls == 0, (
        f"fact_listings.listing_id has {nulls:,} NULL rows. "
        "Silver _LISTINGS_VALID_FILTER should have excluded all null listing_id rows."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U4 -- Unit Tests: Derived Column Correctness
# MAGIC
# MAGIC Gold computes three new measures from Silver columns.
# MAGIC Each is verified against its exact formula from the pipeline.

# COMMAND ----------

def test_u4_car_age_at_listing_correct(spark):
    df  = spark.read.table(G("fact_listings"))
    bad = df.filter(
        F.col("listing_date").isNotNull() &
        F.col("manufacture_year").isNotNull() &
        F.col("car_age_at_listing").isNotNull() &
        (F.col("car_age_at_listing") != (F.year("listing_date") - F.col("manufacture_year")))
    ).count()
    assert bad == 0, (
        f"{bad:,} rows where car_age_at_listing != year(listing_date) - manufacture_year."
    )


def test_u4_is_high_mileage_correct(spark):
    df = spark.read.table(G("fact_listings"))

    # Rows with mileage > 100000 must be True
    bad_high = df.filter(
        F.col("mileage_km").isNotNull() &
        (F.col("mileage_km") > 100000) &
        (F.col("is_high_mileage") == False)
    ).count()

    # Rows with mileage <= 100000 must be False
    bad_low = df.filter(
        F.col("mileage_km").isNotNull() &
        (F.col("mileage_km") <= 100000) &
        (F.col("is_high_mileage") == True)
    ).count()

    assert bad_high == 0, f"{bad_high:,} rows with mileage > 100000 marked is_high_mileage=False."
    assert bad_low  == 0, f"{bad_low:,} rows with mileage <= 100000 marked is_high_mileage=True."


def test_u4_price_per_hp_usd_correct(spark):
    df  = spark.read.table(G("fact_listings"))
    bad = df.filter(
        F.col("price_usd").isNotNull() &
        F.col("engine_power").isNotNull() &
        (F.col("engine_power") > 0) &
        F.col("price_per_hp_usd").isNotNull() &
        (F.abs(F.col("price_per_hp_usd") -
               F.round(F.col("price_usd") / F.col("engine_power"), 2)) > 0.01)
    ).count()
    assert bad == 0, (
        f"{bad:,} rows where price_per_hp_usd != round(price_usd / engine_power, 2)."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U5 -- Unit Tests: SCD2 Metadata Columns

# COMMAND ----------

@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_u5_scd2_metadata_columns_present(spark, entry):
    cols = spark.read.table(G(entry["name"])).columns
    for required in ("__START_AT", "__END_AT"):
        assert required in cols, (
            f"[{entry['name']}] Missing SCD2 column '{required}'. "
            "apply_changes(stored_as_scd_type=2) should add this automatically."
        )


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_u5_scd2_has_active_rows(spark, entry):
    """U5 -- Every SCD2 dim must have at least one active row (__END_AT IS NULL)."""
    active = spark.read.table(G(entry["name"])).filter(F.col("__END_AT").isNull()).count()
    assert active > 0, (
        f"[{entry['name']}] Zero active rows (__END_AT IS NULL). "
        "Dim has no current data -- pipeline may not have completed."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U6 -- Unit Tests: Audit Chain

# COMMAND ----------

def test_u6_fact_audit_chain_complete(spark):
    df     = spark.read.table(G("fact_listings"))
    audit  = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]
    missing_cols = [c for c in audit if c not in df.columns]
    assert missing_cols == [], f"fact_listings missing audit columns: {missing_cols}"

    for col in audit:
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, (
            f"fact_listings.{col} has {nulls:,} NULL rows. "
            "Every row must carry the full audit chain."
        )


def test_u6_gold_load_dt_is_timestamp(spark):
    """U6 -- gold_load_dt must be TIMESTAMP type in fact_listings."""
    dtypes = dict(spark.read.table(G("fact_listings")).dtypes)
    assert dtypes.get("gold_load_dt", "").startswith("timestamp"), (
        f"fact_listings.gold_load_dt is '{dtypes.get('gold_load_dt')}', expected timestamp."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U7 -- Unit Tests: Dim Static Correctness

# COMMAND ----------

def test_u7_dim_date_row_count(spark):
    cnt = spark.read.table(G("dim_date")).count()
    assert cnt == DIM_DATE_EXPECTED_ROWS, (
        f"dim_date has {cnt:,} rows, expected {DIM_DATE_EXPECTED_ROWS:,} "
        "(2010-01-01 to 2030-12-31 inclusive)."
    )


def test_u7_dim_price_category_correct(spark):
    df     = spark.read.table(G("dim_price_category"))
    cnt    = df.count()
    assert cnt == 5, f"dim_price_category has {cnt} rows, expected 5."

    labels = {r["price_category"] for r in df.select("price_category").collect()}
    expected = {"BUDGET", "MID_RANGE", "PREMIUM", "LUXURY", "UNKNOWN"}
    assert labels == expected, (
        f"dim_price_category labels mismatch. Expected: {expected}. Found: {labels}."
    )


@pytest.mark.parametrize("agg_table", AGG_TABLES)
def test_u7_agg_tables_non_empty(spark, agg_table):
    """U7 -- All 5 aggregate tables must be non-empty."""
    cnt = spark.read.table(G(agg_table)).count()
    assert cnt > 0, f"Aggregate table is empty: {G(agg_table)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## R1 -- Reconciliation: fact_listings Count == Silver

# COMMAND ----------

def test_r1_fact_count_exactly_equals_silver(spark):
    silver_cnt = spark.read.table(S("listings_silver_merged")).count()
    gold_cnt   = spark.read.table(G("fact_listings")).count()

    assert gold_cnt == silver_cnt, (
        f"fact_listings count mismatch -- EXACT EQUALITY REQUIRED (Vasu).\n"
        f"  Silver listings_silver_merged: {silver_cnt:,}\n"
        f"  Gold   fact_listings         : {gold_cnt:,}\n"
        f"  Difference                   : {abs(gold_cnt - silver_cnt):,}\n"
        "No tolerance is acceptable. Every Silver row must produce one Gold row."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## R2 + R3 -- Reconciliation: Row-Level Integrity (Both Directions)
# MAGIC
# MAGIC **R2 (Silver → Gold):** Every Silver `listing_id` must exist in Gold -- no drops.
# MAGIC
# MAGIC **R3 (Gold → Silver):** Every Gold `listing_id` must exist in Silver -- no invention.
# MAGIC
# MAGIC Both directions must pass. R2 alone would miss the case where Gold has extra invented rows.

# COMMAND ----------

def test_r2_no_silver_listings_dropped(spark):
    silver_ids = spark.read.table(S("listings_silver_merged")).select("listing_id")
    gold_ids   = spark.read.table(G("fact_listings")).select("listing_id")

    dropped = silver_ids.join(gold_ids, on="listing_id", how="left_anti").count()

    assert dropped == 0, (
        f"R2 FAILED: {dropped:,} Silver listing_id(s) not found in fact_listings.\n"
        "These Silver rows were silently dropped during Gold transformation."
    )


def test_r3_no_gold_listings_invented(spark):
    silver_ids = spark.read.table(S("listings_silver_merged")).select("listing_id")
    gold_ids   = spark.read.table(G("fact_listings")).select("listing_id")

    invented = gold_ids.join(silver_ids, on="listing_id", how="left_anti").count()

    assert invented == 0, (
        f"R3 FAILED: {invented:,} fact_listings listing_id(s) not traceable to Silver.\n"
        "Gold contains rows that were not produced by the Silver pipeline."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## R4 -- Reconciliation: Reconstruct Silver from Gold (JOIN)

# COMMAND ----------

def test_r4_reconstruct_silver_listings_via_dim_join(spark):
    fact      = spark.read.table(G("fact_listings"))
    dim_price = spark.read.table(G("dim_price_category")).select("price_category_key", "price_category")
    dim_steer = spark.read.table(G("dim_steering")).select("steering_key", "steering_wheel")

    reconstructed = (
        fact
        .join(dim_price, on="price_category_key", how="left")
        .join(dim_steer, on="steering_key",        how="left")
        .select(
            "listing_id",
            "listing_date",
            "manufacture_year",
            "engine_power",
            "mileage_km",
            "price_rub",
            "price_usd",
            "has_license",
            "listing_year",
            "listing_month",
            "price_category",
            "steering_wheel",
        )
    )

    silver = (
        spark.read.table(S("listings_silver_merged"))
        .select(
            "listing_id",
            F.col("listing_date").cast("date").alias("listing_date"),
            "manufacture_year",
            "engine_power",
            "mileage_km",
            "price_rub",
            "price_usd",
            "has_license",
            "listing_year",
            "listing_month",
            "price_category",
            "steering_wheel",
        )
    )

    # Direction 1: Silver rows missing from reconstruction
    missing = silver.subtract(reconstructed).count()
    assert missing == 0, (
        f"R4: {missing:,} Silver rows cannot be reconstructed by joining fact + dims.\n"
        "Data was lost or corrupted during the Silver->Gold transformation."
    )

    # Direction 2: Reconstruction has rows not in Silver 
    extra = reconstructed.subtract(silver).count()
    assert extra == 0, (
        f"R4: {extra:,} reconstructed rows have no Silver origin.\n"
        "Gold fact + dim join produced rows that do not exist in Silver."
    )


def test_r4_reconstruct_car_specs_via_dim_car(spark):
    fact    = spark.read.table(G("fact_listings")).select("listing_id", "car_sk")
    dim_car = (
        spark.read.table(G("dim_car"))
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "brand", "model")
        .distinct()
    )

    matched = fact.join(dim_car, on="car_sk", how="inner").count()
    assert matched > 0, (
        "R4: fact_listings joined to dim_car on car_sk produced ZERO matches.\n"
        "car_sk = crc32(lower(brand)|lower(model)) formula may not be applied\n"
        "consistently between fact_listings and dim_car_source."
    )

    # No hash collision: each car_sk must map to exactly one (brand, model) in dim_car
    collision = (
        dim_car
        .groupBy("car_sk")
        .agg(F.countDistinct(F.concat_ws("|", "brand", "model")).alias("combos"))
        .filter(F.col("combos") > 1)
        .count()
    )
    assert collision == 0, (
        f"R4: {collision:,} car_sk values in dim_car map to more than one brand+model.\n"
        "CRC32 hash collision detected -- two different brand+model combos share a car_sk."
    )

    total_fact_sk  = fact.select("car_sk").distinct().count()
    matched_sk     = (
        fact.select("car_sk").distinct()
        .join(dim_car.select("car_sk").distinct(), on="car_sk", how="inner")
        .count()
    )
    coverage_pct = round(matched_sk / total_fact_sk * 100, 1) if total_fact_sk > 0 else 0
    print(
        f"  INFO [car_sk catalog coverage] "
        f"{matched_sk:,} of {total_fact_sk:,} distinct car_sk values ({coverage_pct}%) "
        f"resolve to dim_car. Unmatched = cars in listings not present in the catalog. "
        f"This is expected -- catalog and listings are separate datasets."
    )


def test_r4_reconstruct_text_via_dim_listing_details(spark):
    fact     = spark.read.table(G("fact_listings")).select("listing_id", "word_count")
    dim_text = (
        spark.read.table(G("dim_listing_details"))
        .filter(F.col("__END_AT").isNull())
        .select("listing_id", "text")
        .withColumn(
            "expected_word_count",
            F.size(F.split(F.trim(F.coalesce(F.col("text"), F.lit(""))), r"\s+"))
        )
    )

    # Fact listings that have a text entry: join fact -> dim
    matched = fact.join(dim_text, on="listing_id", how="inner")

    # word_count in fact must match the actual word count of the text in dim
    mismatches = matched.filter(
        F.col("word_count") != F.col("expected_word_count")
    ).count()

    assert mismatches == 0, (
        f"R4: {mismatches:,} listings have word_count in fact that does not match "
        "the actual word count of text in dim_listing_details."
    )

    # Informational: text coverage rate 
    total_fact   = fact.count()
    matched_cnt  = matched.count()
    coverage_pct = round(matched_cnt / total_fact * 100, 1) if total_fact > 0 else 0
    print(
        f"  INFO [text coverage] "
        f"{matched_cnt:,} of {total_fact:,} fact listings ({coverage_pct}%) "
        f"have text in dim_listing_details. "
        f"Unmatched = listings with no description or filtered from Silver."
    )


def test_r4_photo_count_matches_dim_listing_photos(spark):
    fact_counts = (
        spark.read.table(G("fact_listings"))
        .select("listing_id", F.col("photo_count").alias("fact_count"))
    )
    dim_counts = (
        spark.read.table(G("dim_listing_photos"))
        .filter(F.col("__END_AT").isNull())
        .groupBy("listing_id")
        .agg(F.count("photo_url_clean").alias("dim_count"))
    )

    mismatches = (
        fact_counts
        .join(dim_counts, on="listing_id", how="left")
        .filter(
            F.col("fact_count") != F.coalesce(F.col("dim_count"), F.lit(0))
        )
        .count()
    )
    assert mismatches == 0, (
        f"R4: {mismatches:,} listings have mismatched photo_count between fact and dim.\n"
        "fact_listings.photo_count must equal count(photo_url_clean) from dim_listing_photos."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## R5 -- Reconciliation: SCD2 Active Row Count == Silver Distinct Keys
# MAGIC
# MAGIC The number of active rows (`__END_AT IS NULL`) in each SCD2 dim must exactly
# MAGIC equal the number of distinct natural keys in its Silver source.

# COMMAND ----------

@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_r5_scd2_active_count_matches_silver(spark, entry):
    silver_cnt = (
        spark.read.table(entry["silver"])
        .select(*entry["natural_keys"])
        .distinct()
        .count()
    )
    gold_cnt = (
        spark.read.table(G(entry["name"]))
        .filter(F.col("__END_AT").isNull())
        .count()
    )
    assert gold_cnt == silver_cnt, (
        f"[{entry['name']}] Active row count mismatch.\n"
        f"  Silver distinct natural keys: {silver_cnt:,}\n"
        f"  Gold active rows (__END_AT IS NULL): {gold_cnt:,}\n"
        f"  Difference: {abs(gold_cnt - silver_cnt):,}\n"
        f"  Natural keys: {entry['natural_keys']}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## RI1 -- Referential Integrity: No Orphan FK Keys

# COMMAND ----------

def test_ri1_no_orphan_listing_dates(spark):
    
    fact_dates = (
        spark.read.table(G("fact_listings"))
        .select("listing_date")
        .filter(F.col("listing_date").isNotNull())
        .distinct()
    )
    dim_dates = spark.read.table(G("dim_date")).select(F.col("date_key").alias("listing_date"))

    orphans = fact_dates.join(dim_dates, on="listing_date", how="left_anti").count()
    assert orphans == 0, (
        f"RI1: {orphans:,} distinct listing_date values in fact not in dim_date.\n"
        "These dates fall outside dim_date range (2010-01-01 to 2030-12-31)."
    )


def test_ri1_no_orphan_car_sk(spark):
    fact_sk = (
        spark.read.table(G("fact_listings"))
        .select("car_sk")
        .filter(F.col("car_sk").isNotNull())
        .distinct()
    )
    dim_sk = (
        spark.read.table(G("dim_car"))
        .filter(F.col("__END_AT").isNull())
        .select("car_sk")
        .distinct()
    )

    # car_sk must never be null in fact (CRC32 always returns INT)
    nulls = spark.read.table(G("fact_listings")).filter(F.col("car_sk").isNull()).count()
    assert nulls == 0, (
        f"RI1: fact_listings.car_sk has {nulls:,} NULL values. "
        "crc32() always returns a non-null INT -- null means the column is missing."
    )

    collision = (
        spark.read.table(G("dim_car"))
        .filter(F.col("__END_AT").isNull())
        .groupBy("car_sk")
        .agg(F.count("*").alias("cnt"))
        .filter(F.col("cnt") > 1)
        .count()
    )
    assert collision == 0, (
        f"RI1: {collision:,} car_sk values appear more than once in dim_car (active rows). "
        "CRC32 hash collision -- two different brand+model combos share a surrogate key."
    )

    
    matched = fact_sk.join(dim_sk, on="car_sk", how="inner").count()
    assert matched > 0, (
        "RI1: ZERO fact car_sk values match dim_car. "
        "car_sk formula may be applied differently in fact vs dim_car_source."
    )

    total    = fact_sk.count()
    coverage = round(matched / total * 100, 1) if total > 0 else 0
    print(
        f"  INFO [car_sk catalog coverage] "
        f"{matched:,} of {total:,} distinct car_sk ({coverage}%) resolve to dim_car. "
        f"Unmatched = brand+model in listings not found in catalogs.csv. Expected gap."
    )


def test_ri1_no_orphan_location_sk(spark):
    # 1. location_sk must never be NULL in fact
    nulls = spark.read.table(G("fact_listings")).filter(F.col("location_sk").isNull()).count()
    assert nulls == 0, (
        f"RI1: fact_listings.location_sk has {nulls:,} NULL values. "
        "crc32() always returns a non-null INT -- NULL means the column is missing."
    )

    # 2. No duplicate location_sk in dim_location active rows (no CRC32 collision)
    collision = (
        spark.read.table(G("dim_location"))
        .filter(F.col("__END_AT").isNull())
        .groupBy("location_sk")
        .agg(F.count("*").alias("cnt"))
        .filter(F.col("cnt") > 1)
        .count()
    )
    assert collision == 0, (
        f"RI1: {collision:,} location_sk values appear more than once in dim_location "
        "(active rows). CRC32 hash collision -- two different cities share a surrogate key."
    )

    fact_sk = (
        spark.read.table(G("fact_listings"))
        .select("location_sk").distinct()
    )
    dim_sk = (
        spark.read.table(G("dim_location"))
        .filter(F.col("__END_AT").isNull())
        .select("location_sk").distinct()
    )
    matched = fact_sk.join(dim_sk, on="location_sk", how="inner").count()
    assert matched > 0, (
        "RI1: ZERO fact location_sk values match dim_location. "
        "location_sk formula may be applied differently in fact vs dim_location_source."
    )

    # Informational: geo coverage rate 
    total    = fact_sk.count()
    coverage = round(matched / total * 100, 1) if total > 0 else 0
    print(
        f"  INFO [location_sk geo coverage] "
        f"{matched:,} of {total:,} distinct location_sk ({coverage}%) resolve to dim_location. "
        f"Unmatched = seller-entered city names not found in geo reference file. Expected gap."
    )


def test_ri1_no_orphan_price_category_key(spark):
    fact_keys = (
        spark.read.table(G("fact_listings"))
        .select("price_category_key")
        .filter(F.col("price_category_key").isNotNull())
        .distinct()
    )
    dim_keys = spark.read.table(G("dim_price_category")).select("price_category_key")

    orphans = fact_keys.join(dim_keys, on="price_category_key", how="left_anti").count()
    assert orphans == 0, (
        f"RI1: {orphans:,} price_category_key values in fact not in dim_price_category."
    )


def test_ri1_no_orphan_steering_key(spark):
    fact_keys = (
        spark.read.table(G("fact_listings"))
        .select("steering_key")
        .filter(F.col("steering_key").isNotNull())
        .distinct()
    )
    dim_keys = spark.read.table(G("dim_steering")).select("steering_key")

    orphans = fact_keys.join(dim_keys, on="steering_key", how="left_anti").count()
    assert orphans == 0, (
        f"RI1: {orphans:,} steering_key values in fact not in dim_steering."
    )


def test_ri1_no_orphan_listing_details(spark):
    # No duplicate active listing_id in dim_listing_details
    dup_ids = (
        spark.read.table(G("dim_listing_details"))
        .filter(F.col("__END_AT").isNull())
        .groupBy("listing_id")
        .agg(F.count("*").alias("cnt"))
        .filter(F.col("cnt") > 1)
        .count()
    )
    assert dup_ids == 0, (
        f"RI1: {dup_ids:,} listing_id(s) appear more than once in dim_listing_details "
        "(active rows only). SCD2 apply_changes() should produce one active row per key."
    )

    # Every dim_listing_details listing_id must trace to its Silver source
    silver_text_ids = (
        spark.read.table(S("listings_text_transformation"))
        .select("listing_id")
        .distinct()
    )
    dim_ids = (
        spark.read.table(G("dim_listing_details"))
        .filter(F.col("__END_AT").isNull())
        .select("listing_id")
        .distinct()
    )
    invented = dim_ids.join(silver_text_ids, on="listing_id", how="left_anti").count()
    assert invented == 0, (
        f"RI1: {invented:,} dim_listing_details listing_id(s) not traceable to "
        "Silver listings_text_transformation. Gold invented listing_id values."
    )

    # Informational: how many fact listings have text
    fact_ids     = spark.read.table(G("fact_listings")).select("listing_id").distinct()
    fact_with_txt = fact_ids.join(dim_ids, on="listing_id", how="inner").count()
    total_fact   = fact_ids.count()
    pct          = round(fact_with_txt / total_fact * 100, 1) if total_fact > 0 else 0
    print(
        f"  INFO [text coverage] {fact_with_txt:,} of {total_fact:,} "
        f"fact listings ({pct}%) have text in dim_listing_details."
    )
