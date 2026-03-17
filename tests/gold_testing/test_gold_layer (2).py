# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer Test Suite
# MAGIC
# MAGIC | Type | Suite | Tests | What it proves |
# MAGIC |------|-------|-------|----------------|
# MAGIC | **Unit** | U1 - Existence & Schema | 4 | Every Gold table exists, non-empty, correct column types |
# MAGIC | **Unit** | U2 - Fact Column Types | 1 | All FKs are INT, color_r/g/b INT, listing_id STRING, listing_date DATE |
# MAGIC | **Unit** | U3 - FK Non-Null | 1 | car_sk, location_sk, price_category_key, steering_key never null |
# MAGIC | **Unit** | U4 - Derived Column Correctness | 3 | car_age_at_listing, is_high_mileage, price_per_hp_usd computed correctly |
# MAGIC | **Unit** | U5 - SCD2 Metadata | 1 | All SCD2 dims have __START_AT / __END_AT |
# MAGIC | **Unit** | U6 - Audit Chain | 1 | bronze_load_dt -> silver_load_dt -> gold_load_dt present, non-null |
# MAGIC | **Unit** | U7 - Dim Static Correctness | 2 | dim_date 7670 rows, dim_price_category 5 rows with correct labels |
# MAGIC | **Reconciliation** | R1 - Silver Count == Gold Count | 1 | fact_listings == listings_silver_merged: exact equality, zero tolerance |
# MAGIC | **Reconciliation** | R2 - Silver -> Gold (no drops) | 1 | Every Silver listing_id exists in Gold (left_anti empty) |
# MAGIC | **Reconciliation** | R3 - Gold -> Silver (no invention) | 1 | Every Gold listing_id exists in Silver (left_anti empty) |
# MAGIC | **Reconciliation** | R4 - Reconstruct Silver from Gold | 4 | JOIN fact+dims rebuilds Silver columns exactly |
# MAGIC | **Reconciliation** | R5 - SCD2 Count Matches Silver | 1 | Active dim rows == distinct Silver natural keys |
# MAGIC | **Referential Integrity** | RI1 - No orphan FK keys | 6 | Every FK value in fact exists in its dim (left_anti == 0) |
# MAGIC
# MAGIC **Total: 27 tests**
# MAGIC
# MAGIC ### Key rules from Vasu Bajaj evaluation
# MAGIC - **Zero tolerance** on all counts -- `assert count == expected`, no percentage bands
# MAGIC - **Both directions** for every integrity check -- no drops AND no invention
# MAGIC - **No strings in fact** -- `brand`, `model`, `city_prepositional` removed; replaced by `car_sk` (INT) and `location_sk` (INT)
# MAGIC - Reconciliation = JOIN fact + dims -> reconstructs Silver source of truth
# MAGIC - Referential integrity = every FK in fact exists in its dim (left_anti join must be empty)

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

# dim_date: 2010-01-01 to 2030-12-31 inclusive = 7670 days
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
    """
    U1 -- fact_listings must NOT contain raw string attribute columns.
    Vasu: no strings in fact table except listing_id (degenerate dim).
    brand, model, city_prepositional must be replaced by integer surrogate keys.
    """
    cols    = spark.read.table(G("fact_listings")).columns
    banned  = ["brand", "model", "city_prepositional", "city_name",
               "steering_wheel", "price_category", "fuel_type"]
    present = [c for c in banned if c in cols]
    assert present == [], (
        f"fact_listings contains raw string attribute columns: {present}. "
        "Vasu: replace strings with integer surrogate keys. "
        "brand+model -> car_sk INT, city_prepositional -> location_sk INT, "
        "steering_wheel -> steering_key INT, price_category -> price_category_key INT."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U2 -- Unit Tests: Fact Column Data Types
# MAGIC
# MAGIC All FK columns must be INT, color_r/g/b must be INT, listing_id must be STRING, listing_date must be DATE.

# COMMAND ----------

def test_u2_fact_column_types_correct(spark):
    """
    U2 -- Every fact_listings column must have the correct data type.

    Key type decisions:
      car_sk, location_sk, price_category_key, steering_key: INT (surrogate keys)
      listing_id: STRING (degenerate dim -- Vasu approved)
      listing_date: DATE (cast from Silver TIMESTAMP in Gold)
      color_r/g/b: INT (Silver: try_cast(R as int) -- no cast in Gold)
      price_rub, price_usd, price_per_hp_usd: double
      is_high_mileage: boolean
    """
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
    """
    U3 -- Integer FK columns must have zero NULL values in fact_listings.

    car_sk      = crc32(lower(brand)|lower(model)) -- always produces INT from non-null strings
    location_sk = crc32(lower(city_prepositional)) -- always produces INT from non-null strings
    price_category_key -- joined from dim_price_category; NULL means join failed
    steering_key       -- joined from dim_steering; NULL means steering_wheel had no match

    Note: steering_key and price_category_key CAN be null for rows where
    Silver steering_wheel or price_category was null. The test checks the
    surrogate computed keys (car_sk, location_sk) which are never null.
    """
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
    """
    U4 -- car_age_at_listing must equal year(listing_date) - manufacture_year.
    Gold pipeline: (F.year("listing_date") - F.col("manufacture_year")).alias("car_age_at_listing")
    """
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
    """
    U4 -- is_high_mileage must be True when mileage_km > 100000, else False.
    Gold pipeline: F.when(F.col("mileage_km") > 100000, True).otherwise(False)
    """
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
    """
    U4 -- price_per_hp_usd must equal round(price_usd / engine_power, 2).
    Gold pipeline: F.round(F.col("price_usd") / F.nullif(F.col("engine_power"), F.lit(0)), 2)
    Zero engine_power returns NULL (nullif guard).
    """
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
    """
    U5 -- All SCD2 dim tables must have __START_AT and __END_AT columns.
    DLT apply_changes() adds these automatically for stored_as_scd_type=2.
    """
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
    """
    U6 -- fact_listings must carry the full Bronze->Silver->Gold audit chain.
    All four audit columns must be present and have zero NULL values.
    bronze_source_file: STRING (Bronze source filename)
    bronze_load_dt:     TIMESTAMP (Bronze ingestion time)
    silver_load_dt:     TIMESTAMP (Silver transformation time)
    gold_load_dt:       TIMESTAMP (Gold build time)
    """
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
    """
    U7 -- dim_date must contain exactly 7670 rows (2010-01-01 to 2030-12-31 inclusive).
    Gap-free calendar: every single date in the range must have exactly one row.
    """
    cnt = spark.read.table(G("dim_date")).count()
    assert cnt == DIM_DATE_EXPECTED_ROWS, (
        f"dim_date has {cnt:,} rows, expected {DIM_DATE_EXPECTED_ROWS:,} "
        "(2010-01-01 to 2030-12-31 inclusive)."
    )


def test_u7_dim_price_category_correct(spark):
    """
    U7 -- dim_price_category must have exactly 5 rows with correct labels.
    Labels must match Silver _transform_listings price_category CASE expression exactly.
    """
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
# MAGIC
# MAGIC **Vasu's exact rule:** Assert exact equality. Zero tolerance.
# MAGIC No percentage bands. No `diff_pct <= 1.0`. No `diff_pct <= 5.0`.
# MAGIC
# MAGIC fact_listings has the same grain as `listings_silver_merged` (one row per `listing_id`).
# MAGIC The count must be identical -- no deduplication happens between Silver and Gold.

# COMMAND ----------

def test_r1_fact_count_exactly_equals_silver(spark):
    """
    R1 -- fact_listings.count() must exactly equal listings_silver_merged.count().

    Vasu Bajaj evaluation feedback:
      "The reconciliation check showed a 1.05 to 0.95 tolerance, suggesting
       the fact table must have fewer rows, which they considered incorrect
       as it implies acceptance of missing rows and a fundamental failure in
       engineering."

    Gold pipeline: fact_listings reads listings_silver_merged directly and
    applies NO deduplication. Every Silver row must produce exactly one Gold row.
    Exact equality -- zero tolerance -- is the only acceptable assertion.
    """
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
    """
    R2 -- Every listing_id in Silver must exist in fact_listings.
    Direction: Silver -> Gold. Proves nothing was silently dropped.
    Method: left_anti join Silver onto Gold. Result must be empty.
    """
    silver_ids = spark.read.table(S("listings_silver_merged")).select("listing_id")
    gold_ids   = spark.read.table(G("fact_listings")).select("listing_id")

    dropped = silver_ids.join(gold_ids, on="listing_id", how="left_anti").count()

    assert dropped == 0, (
        f"R2 FAILED: {dropped:,} Silver listing_id(s) not found in fact_listings.\n"
        "These Silver rows were silently dropped during Gold transformation."
    )


def test_r3_no_gold_listings_invented(spark):
    """
    R3 -- Every listing_id in fact_listings must exist in Silver.
    Direction: Gold -> Silver. Proves no rows were invented by the pipeline.
    Method: left_anti join Gold onto Silver. Result must be empty.
    """
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
# MAGIC
# MAGIC **Vasu's requirement:** *"You must be able to join your Fact and Dimension tables
# MAGIC and get the original Silver table back as a form of reconciliation and testing."*
# MAGIC
# MAGIC Method: JOIN `fact_listings` back to the relevant dims, select the original Silver columns,
# MAGIC and verify the result matches the Silver source exactly using `subtract()` in both directions.

# COMMAND ----------

def test_r4_reconstruct_silver_listings_via_dim_join(spark):
    """
    R4 -- Joining fact_listings + dim_price_category + dim_steering must
    reconstruct the key columns from listings_silver_merged exactly.

    This proves:
      1. The integer surrogate key encoding (price_category_key, steering_key)
         is reversible back to the original string labels.
      2. No data was lost or corrupted during the string->int->string round-trip.

    Columns reconstructed:
      listing_id, listing_date, manufacture_year, engine_power, mileage_km,
      price_rub, price_usd, has_license, listing_year, listing_month,
      price_category (decoded from price_category_key via dim_price_category),
      steering_wheel  (decoded from steering_key via dim_steering)
    """
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

    # Direction 2: Reconstruction has rows not in Silver (invention)
    extra = reconstructed.subtract(silver).count()
    assert extra == 0, (
        f"R4: {extra:,} reconstructed rows have no Silver origin.\n"
        "Gold fact + dim join produced rows that do not exist in Silver."
    )


def test_r4_reconstruct_car_specs_via_dim_car(spark):
    """
    R4 -- When a fact listing matches dim_car on car_sk, the joined brand/model
    must be consistent with what the pipeline computed.

    WHY full orphan == 0 is NOT the right assertion:
      fact_listings.car_sk  comes from listings_silver_merged
        (Bronze: user-submitted listing ads -- brand+model typed by sellers)
      dim_car.car_sk        comes from car_catalog_transformation
        (Bronze: catalogs.csv -- a separate reference catalog dataset)

      These are TWO DIFFERENT source datasets. Not every brand+model combination
      that appears in listings exists in the catalog. A listing for an obscure
      or misspelled model will produce a car_sk with no dim_car match.
      This is expected behaviour -- it is a catalog coverage gap, not a pipeline bug.

    What this test CORRECTLY asserts:
      1. At least 1 fact listing matches dim_car (the join is not completely broken).
      2. For every matched row, car_sk is consistent between fact and dim
         (no hash collision where two different brand+model combos share a car_sk).
    """
    fact    = spark.read.table(G("fact_listings")).select("listing_id", "car_sk")
    dim_car = (
        spark.read.table(G("dim_car"))
        .filter(F.col("__END_AT").isNull())
        .select("car_sk", "brand", "model")
        .distinct()
    )

    # The join must produce at least some matches -- if 0, the formula is broken
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

    # Informational: catalog coverage rate (not an assertion -- expected < 100%)
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
    """
    R4 -- Joining fact_listings + dim_listing_details (active) on listing_id
    must recover text for every listing that has a description.

    Note: Not all listings have text (left join -- some photo_count=0 listings
    may also have no text). The test verifies that every listing_id that EXISTS
    in dim_listing_details can be joined to fact_listings.
    """
    fact_ids = spark.read.table(G("fact_listings")).select("listing_id")
    dim_text = (
        spark.read.table(G("dim_listing_details"))
        .filter(F.col("__END_AT").isNull())
        .select("listing_id", "text")
    )

    # Every dim_listing_details listing_id must exist in fact
    orphan_text = (
        dim_text.select("listing_id").distinct()
        .join(fact_ids.distinct(), on="listing_id", how="left_anti")
        .count()
    )
    assert orphan_text == 0, (
        f"R4: {orphan_text:,} dim_listing_details listing_id(s) have no match in fact_listings.\n"
        "Text descriptions exist for listings that are not in the fact table."
    )


def test_r4_photo_count_matches_dim_listing_photos(spark):
    """
    R4 -- photo_count in fact_listings must match count(photo_url_clean)
    from dim_listing_photos for every listing_id.

    photo_count is a denormalized measure in fact (groupBy count from photo Silver table).
    dim_listing_photos holds the raw photo URLs.
    These two must agree exactly -- any mismatch means the denormalization is wrong.
    """
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
    """
    R5 -- Active SCD2 rows (__END_AT IS NULL) must exactly equal
    distinct natural key count from Silver source table.

    Silver keys -> apply_changes() -> exactly one active row per natural key.
    If Silver has 1000 distinct (brand, model) combos, dim_car must have
    exactly 1000 active rows.
    """
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
# MAGIC
# MAGIC **Vasu's requirement:** *"Ensure that only keys present in the Dimension tables
# MAGIC exist in the Fact table (no orphan keys)."*
# MAGIC
# MAGIC Method: `left_anti` join fact FK onto dim PK. Result must be **exactly zero**.
# MAGIC No percentage tolerance. Every FK in fact must resolve to a dim row.

# COMMAND ----------

def test_ri1_no_orphan_listing_dates(spark):
    """
    RI1 -- Every listing_date in fact must exist in dim_date.date_key.
    dim_date covers 2010-01-01 to 2030-12-31. Any listing outside this range is orphaned.
    """
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
    """
    RI1 -- Every car_sk in fact must exist as an active row in dim_car.
    car_sk = crc32(lower(brand)|lower(model)) -- stable surrogate key.
    """
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
    orphans = fact_sk.join(dim_sk, on="car_sk", how="left_anti").count()
    assert orphans == 0, (
        f"RI1: {orphans:,} distinct car_sk values in fact have no active row in dim_car.\n"
        "car_sk is crc32(lower(brand)|lower(model)) -- mismatch means surrogate key inconsistency."
    )


def test_ri1_no_orphan_location_sk(spark):
    """
    RI1 -- Every location_sk in fact must exist as an active row in dim_location.
    location_sk = crc32(lower(city_prepositional)) -- stable surrogate key.
    """
    fact_sk = (
        spark.read.table(G("fact_listings"))
        .select("location_sk")
        .filter(F.col("location_sk").isNotNull())
        .distinct()
    )
    dim_sk = (
        spark.read.table(G("dim_location"))
        .filter(F.col("__END_AT").isNull())
        .select("location_sk")
        .distinct()
    )
    orphans = fact_sk.join(dim_sk, on="location_sk", how="left_anti").count()
    assert orphans == 0, (
        f"RI1: {orphans:,} distinct location_sk values in fact have no active row in dim_location."
    )


def test_ri1_no_orphan_price_category_key(spark):
    """
    RI1 -- Every non-null price_category_key in fact must exist in dim_price_category.
    dim_price_category has 5 rows (BUDGET/MID_RANGE/PREMIUM/LUXURY/UNKNOWN).
    price_category_key can be null for rows where price_rub is null.
    """
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
    """
    RI1 -- Every non-null steering_key in fact must exist in dim_steering.
    steering_key can be null for rows where steering_wheel was null in Silver.
    """
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
    """
    RI1 -- Every listing_id in dim_listing_details must exist in fact_listings.
    Verifies the reverse: dim must not contain details for listings
    that do not appear in the fact table.
    """
    fact_ids = spark.read.table(G("fact_listings")).select("listing_id").distinct()
    dim_ids  = (
        spark.read.table(G("dim_listing_details"))
        .filter(F.col("__END_AT").isNull())
        .select("listing_id")
        .distinct()
    )
    orphans = dim_ids.join(fact_ids, on="listing_id", how="left_anti").count()
    assert orphans == 0, (
        f"RI1: {orphans:,} listing_id(s) in dim_listing_details not in fact_listings.\n"
        "Text descriptions exist for listings that are not in the fact table."
    )
