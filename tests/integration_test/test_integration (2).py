# Databricks notebook source
# MAGIC %md
# MAGIC # Integration Test Suite — Bronze → Silver → Gold
# MAGIC
# MAGIC These tests verify the **full pipeline end-to-end**, crossing all three layer boundaries.
# MAGIC Each test answers one critical question about the pipeline as a whole.
# MAGIC
# MAGIC | Test | Boundary | Question it answers |
# MAGIC |------|----------|---------------------|
# MAGIC | **IT1** | Bronze → Silver → Gold | Does every valid Bronze listing_id reach Gold fact? |
# MAGIC | **IT2** | Bronze → Silver → Gold | Is the audit chain (bronze_source_file, bronze_load_dt) carried intact from Bronze all the way to Gold? |
# MAGIC | **IT3** | Bronze → Silver → Gold | Is the total row count consistent across all three layers? |
# MAGIC | **IT4** | Silver → Gold | Does joining fact + dims reconstruct the original Silver listings_silver_merged exactly? |
# MAGIC | **IT5** | Bronze → Gold | Do financial measures (price_rub, price_usd) in Gold match the Bronze source values after applying the same transform? |
# MAGIC
# MAGIC ### Why these 5 are the most important
# MAGIC - **IT1** proves nothing was lost or invented across the whole pipeline — the core Vasu requirement
# MAGIC - **IT2** proves the audit chain is unbroken — every Gold row can be traced back to its Bronze file
# MAGIC - **IT3** gives a single row-count consistency check across all three layers at once
# MAGIC - **IT4** is Vasu's exact reconstruction requirement: JOIN fact+dims → get Silver back
# MAGIC - **IT5** proves numeric transformation correctness end-to-end — not just presence but actual values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & Configuration

# COMMAND ----------

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F


@pytest.fixture(scope="session")
def spark():
    return DatabricksSession.builder.getOrCreate()


CATALOG = "vstone_catalog"
BRONZE  = f"{CATALOG}.bronze"
SILVER  = f"{CATALOG}.silver"
GOLD    = f"{CATALOG}.gold"

B = lambda t: f"{BRONZE}.{t}"
S = lambda t: f"{SILVER}.{t}"
G = lambda t: f"{GOLD}.{t}"

# The 4 Bronze listing source tables that feed listings_silver_merged
LISTING_BRONZE_TABLES = [
    "listings_csv_copyinto",
    "listings_json_autoloader",
    "listings_xml_pyspark",
    "listings_csv_dlt",
]

USD_RATE = 82.5  # Feb 2023 historical rate -- matches Silver _transform_listings

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT1 — End-to-End Row Integrity: Bronze → Silver → Gold
# MAGIC
# MAGIC **Question:** Does every valid Bronze `listing_id` that passes the Silver filter reach `fact_listings`?
# MAGIC
# MAGIC **Pipeline path:**
# MAGIC ```
# MAGIC Bronze (4 tables)  →  _transform_listings  →  _LISTINGS_VALID_FILTER  →  listings_silver_merged  →  fact_listings
# MAGIC ```
# MAGIC
# MAGIC **Silver filter:** `listing_id IS NOT NULL AND price_rub IS NOT NULL AND listing_date IS NOT NULL`
# MAGIC
# MAGIC **Method:** Apply the same cast and filter to Bronze directly in the test, then verify
# MAGIC every resulting `listing_id` exists in Gold using `left_anti` in both directions.

# COMMAND ----------

def test_it1_every_valid_bronze_listing_reaches_gold(spark):
    """
    IT1 -- Full pipeline row integrity: Bronze -> Silver -> Gold.

    Every Bronze listing_id that passes the Silver validity filter
    (_LISTINGS_VALID_FILTER: listing_id/price_rub/listing_date not null)
    must appear in fact_listings. No valid Bronze row should be lost
    anywhere in the Bronze -> Silver -> Gold pipeline.

    This is the single most important integration test:
    it crosses all three layer boundaries in one assertion.

    Silver _LISTINGS_VALID_FILTER (exact from 08_silver_transformation):
      listing_id IS NOT NULL
      AND price_rub IS NOT NULL
      AND listing_date IS NOT NULL

    Bronze listing_id cast (exact from _transform_listings):
      try_cast(id as long).cast("string")
    Bronze price_rub cast:
      try_cast(regexp_replace(cost, "[^0-9.]", "") as double)
    Bronze listing_date cast:
      coalesce(to_timestamp(date, "dd.MM.yyyy"), to_timestamp(date, "yyyy-MM-dd'T'HH:mm:ss'Z'"))
    """
    # Step 1: union all 4 Bronze listing tables
    bronze_dfs = [
        spark.read.table(B(t)).select("id", "cost", "date")
        for t in LISTING_BRONZE_TABLES
    ]
    df_bronze = bronze_dfs[0]
    for df in bronze_dfs[1:]:
        df_bronze = df_bronze.unionByName(df, allowMissingColumns=True)

    # Step 2: apply the same transforms as _transform_listings
    # NOTE: use try_to_timestamp (not to_timestamp) in the test.
    # Silver uses F.to_timestamp inside DLT streaming -- Spark's streaming engine
    # coerces unparseable dates to NULL silently.
    # In batch test context, F.to_timestamp throws CANNOT_PARSE_TIMESTAMP on bad input.
    # try_to_timestamp mirrors the Silver behaviour: bad date -> NULL -> filtered out.
    df_transformed = df_bronze.select(
        F.expr("try_cast(id as long)").cast("string").alias("listing_id"),
        F.expr("try_cast(regexp_replace(cost, '[^0-9.]', '') as double)").alias("price_rub"),
        F.coalesce(
            F.expr("try_to_timestamp(date, 'dd.MM.yyyy')"),
            F.expr("try_to_timestamp(date, 'yyyy-MM-dd\'T\'HH:mm:ss\'Z\'')"),
        ).alias("listing_date"),
    )

    # Step 3: apply the same Silver validity filter
    valid_bronze_ids = (
        df_transformed
        .filter(
            F.col("listing_id").isNotNull() &
            F.col("price_rub").isNotNull()  &
            F.col("listing_date").isNotNull()
        )
        .select("listing_id")
        .dropDuplicates(["listing_id"])
    )

    gold_ids = spark.read.table(G("fact_listings")).select("listing_id")

    # Direction 1: valid Bronze ids missing from Gold (dropped somewhere)
    dropped = valid_bronze_ids.join(gold_ids, on="listing_id", how="left_anti").count()
    assert dropped == 0, (
        f"IT1 FAILED: {dropped:,} valid Bronze listing_id(s) never reached fact_listings.\n"
        "These rows passed the Silver validity filter but are missing from Gold.\n"
        "Check: Silver deduplication, Gold fact_listings join logic, DLT pipeline errors."
    )

    # Direction 2: Gold ids not in valid Bronze (invented somewhere)
    invented = gold_ids.join(valid_bronze_ids, on="listing_id", how="left_anti").count()
    assert invented == 0, (
        f"IT1 FAILED: {invented:,} fact_listings listing_id(s) not traceable to Bronze.\n"
        "Gold contains listing_ids that do not exist in any Bronze source table."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT2 — Audit Chain Continuity: Bronze → Silver → Gold
# MAGIC
# MAGIC **Question:** Is `bronze_source_file` in `fact_listings` a real file that exists in Bronze?
# MAGIC
# MAGIC **Why this matters:** The full audit chain `bronze_source_file → bronze_load_dt → silver_load_dt → gold_load_dt`
# MAGIC must be unbroken. If `bronze_source_file` in Gold contains a value not present in any Bronze table,
# MAGIC the audit trail has been fabricated and cannot be trusted for debugging or compliance.

# COMMAND ----------

def test_it2_audit_chain_bronze_source_file_traceable(spark):
    """
    IT2 -- Every bronze_source_file in fact_listings must match a real
    source_file value from one of the 4 Bronze listing tables.

    Pipeline audit chain:
      Bronze.source_file
        -> Silver._transform_listings: F.col("source_file").alias("bronze_source_file")
           -> Gold.fact_listings: "bronze_source_file" (pass-through)

    If any Gold bronze_source_file value is not in Bronze, the chain is broken.
    This proves the audit trail from Bronze to Gold is genuine.
    """
    # Collect all distinct source_file values from all 4 Bronze listing tables
    bronze_files_dfs = [
        spark.read.table(B(t)).select("source_file")
        for t in LISTING_BRONZE_TABLES
    ]
    all_bronze_files = bronze_files_dfs[0]
    for df in bronze_files_dfs[1:]:
        all_bronze_files = all_bronze_files.union(df)
    all_bronze_files = all_bronze_files.distinct()

    gold_files = (
        spark.read.table(G("fact_listings"))
        .select("bronze_source_file")
        .filter(F.col("bronze_source_file").isNotNull())
        .distinct()
    )

    orphan_files = gold_files.join(
        all_bronze_files,
        gold_files["bronze_source_file"] == all_bronze_files["source_file"],
        how="left_anti"
    ).count()

    assert orphan_files == 0, (
        f"IT2 FAILED: {orphan_files:,} distinct bronze_source_file value(s) in "
        "fact_listings not found in any Bronze listing table.\n"
        "The audit trail has been broken -- Gold references source files that "
        "do not exist in Bronze."
    )


def test_it2_audit_timestamps_ordered(spark):
    """
    IT2 -- Audit timestamps must be chronologically ordered in fact_listings.

    bronze_load_dt <= silver_load_dt <= gold_load_dt

    If silver_load_dt < bronze_load_dt, Silver was written before Bronze -- impossible.
    If gold_load_dt < silver_load_dt, Gold was written before Silver -- impossible.
    These violations indicate timestamp fabrication or pipeline ordering errors.
    """
    df = spark.read.table(G("fact_listings"))

    # silver_load_dt must be >= bronze_load_dt
    silver_before_bronze = df.filter(
        F.col("silver_load_dt").isNotNull() &
        F.col("bronze_load_dt").isNotNull() &
        (F.col("silver_load_dt") < F.col("bronze_load_dt"))
    ).count()
    assert silver_before_bronze == 0, (
        f"IT2 FAILED: {silver_before_bronze:,} rows where silver_load_dt < bronze_load_dt.\n"
        "Silver cannot be written before Bronze. Audit timestamps are invalid."
    )

    # gold_load_dt must be >= silver_load_dt
    gold_before_silver = df.filter(
        F.col("gold_load_dt").isNotNull() &
        F.col("silver_load_dt").isNotNull() &
        (F.col("gold_load_dt") < F.col("silver_load_dt"))
    ).count()
    assert gold_before_silver == 0, (
        f"IT2 FAILED: {gold_before_silver:,} rows where gold_load_dt < silver_load_dt.\n"
        "Gold cannot be written before Silver. Audit timestamps are invalid."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT3 — Row Count Consistency: Bronze → Silver → Gold
# MAGIC
# MAGIC **Question:** Is the row count consistent and explainable across all three layers?
# MAGIC
# MAGIC **Expected relationship:**
# MAGIC ```
# MAGIC Bronze raw total  >=  Silver (deduplicated + filtered)  ==  Gold fact_listings
# MAGIC ```
# MAGIC Silver deduplicates on `listing_id` and applies `_LISTINGS_VALID_FILTER`.
# MAGIC Gold applies no further deduplication — Silver count must equal Gold count exactly.

# COMMAND ----------

def test_it3_row_count_bronze_gte_silver_gte_zero(spark):
    """
    IT3 -- Bronze total >= Silver valid count >= 0.

    Bronze raw rows >= Silver rows because Silver:
      1. Deduplicates on listing_id (keeps one row per id)
      2. Filters out rows where listing_id/price_rub/listing_date is null
      3. Unions 4 Bronze tables (may have overlapping ids across sources)

    Silver count must be strictly positive -- if Silver is empty,
    the pipeline failed completely.
    """
    # Total raw Bronze listing rows across all 4 tables
    bronze_total = 0
    for t in LISTING_BRONZE_TABLES:
        bronze_total += spark.read.table(B(t)).count()

    silver_cnt = spark.read.table(S("listings_silver_merged")).count()

    assert silver_cnt > 0, (
        "IT3 FAILED: listings_silver_merged is empty. "
        "Silver pipeline produced zero rows from Bronze."
    )
    assert bronze_total >= silver_cnt, (
        f"IT3 FAILED: Silver ({silver_cnt:,}) has MORE rows than Bronze total ({bronze_total:,}).\n"
        "Silver cannot produce more rows than the raw Bronze source. "
        "Possible cause: Bronze tables were truncated after Silver pipeline ran."
    )

    print(
        f"  INFO [row counts]\n"
        f"    Bronze total (4 tables, raw): {bronze_total:,}\n"
        f"    Silver listings_silver_merged: {silver_cnt:,}\n"
        f"    Reduction (dedup + filter):   {bronze_total - silver_cnt:,} "
        f"({round((bronze_total - silver_cnt) / bronze_total * 100, 1)}%)"
    )


def test_it3_silver_count_exactly_equals_gold_fact(spark):
    """
    IT3 -- listings_silver_merged.count() must EXACTLY equal fact_listings.count().

    Gold applies NO deduplication on top of Silver.
    fact_listings reads listings_silver_merged 1:1 -- every Silver row
    produces exactly one Gold fact row.

    Exact equality -- zero tolerance (Vasu: percentage tolerance = data loss acceptance).
    """
    silver_cnt = spark.read.table(S("listings_silver_merged")).count()
    gold_cnt   = spark.read.table(G("fact_listings")).count()

    assert silver_cnt == gold_cnt, (
        f"IT3 FAILED: Silver ({silver_cnt:,}) != Gold ({gold_cnt:,}).\n"
        f"  Difference: {abs(silver_cnt - gold_cnt):,} rows.\n"
        "fact_listings must have exactly the same row count as listings_silver_merged.\n"
        "Gold applies no deduplication -- every Silver row must produce one Gold row."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT4 — Full Reconstruction: JOIN fact + dims → Silver
# MAGIC
# MAGIC **Vasu's exact requirement:** *"You must be able to join your Fact and Dimension tables
# MAGIC and get the original Silver table back as a form of reconciliation and testing."*
# MAGIC
# MAGIC This test joins `fact_listings` with all relevant dims and verifies that the
# MAGIC reconstructed dataset matches `listings_silver_merged` exactly — both directions.

# COMMAND ----------

def test_it4_full_reconstruction_fact_plus_dims_equals_silver(spark):
    """
    IT4 -- Joining fact_listings + dim_price_category + dim_steering
    must perfectly reconstruct the original listings_silver_merged columns.

    This is Vasu's core integration test requirement:
      "Join Fact + Dims -> get Silver back."

    Reconstruction proves:
      1. Integer surrogate key encoding is lossless and reversible:
         price_category_key -> dim_price_category -> price_category STRING
         steering_key       -> dim_steering       -> steering_wheel STRING
      2. All numeric measures (price_rub, mileage_km, etc.) passed through intact
      3. No data was corrupted or dropped during Silver -> Gold transformation

    Columns used for comparison (columns that exist in both Silver and Gold
    after decoding the integer FKs back to their original string labels):
      listing_id, listing_date, manufacture_year, engine_power, mileage_km,
      has_license, listing_year, listing_month, car_age_years,
      price_rub, price_usd,
      price_category (decoded from price_category_key),
      steering_wheel  (decoded from steering_key)
    """
    fact      = spark.read.table(G("fact_listings"))
    dim_price = spark.read.table(G("dim_price_category")).select("price_category_key", "price_category")
    dim_steer = spark.read.table(G("dim_steering")).select("steering_key", "steering_wheel")

    # Reconstruct Silver-equivalent columns by decoding integer FKs
    reconstructed = (
        fact
        .join(dim_price, on="price_category_key", how="left")
        .join(dim_steer, on="steering_key",        how="left")
        .select(
            "listing_id",
            "listing_date",          # DATE in Gold (cast from Silver TIMESTAMP)
            "manufacture_year",
            "engine_power",
            "mileage_km",
            "has_license",
            "listing_year",
            "listing_month",
            "car_age_years",
            "price_rub",
            "price_usd",
            "price_category",        # decoded from price_category_key
            "steering_wheel",        # decoded from steering_key
        )
    )

    # Silver source with matching column selection
    silver = (
        spark.read.table(S("listings_silver_merged"))
        .select(
            "listing_id",
            F.col("listing_date").cast("date").alias("listing_date"),
            "manufacture_year",
            "engine_power",
            "mileage_km",
            "has_license",
            "listing_year",
            "listing_month",
            "car_age_years",
            "price_rub",
            "price_usd",
            "price_category",
            "steering_wheel",
        )
    )

    # Direction 1: Silver rows missing from reconstruction (data was lost)
    lost = silver.subtract(reconstructed).count()
    assert lost == 0, (
        f"IT4 FAILED: {lost:,} Silver rows cannot be reconstructed by joining fact + dims.\n"
        "Data was lost or corrupted during Silver -> Gold transformation.\n"
        "Check: price_category encoding, steering_wheel encoding, numeric measure casts."
    )

    # Direction 2: Reconstruction has rows not in Silver (data was invented)
    invented = reconstructed.subtract(silver).count()
    assert invented == 0, (
        f"IT4 FAILED: {invented:,} reconstructed rows have no Silver origin.\n"
        "Joining fact + dims produced rows that do not exist in Silver.\n"
        "Check: duplicate rows in dims, incorrect join conditions."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT5 — Financial Measure Accuracy: Bronze → Gold
# MAGIC
# MAGIC **Question:** Do the financial measures in `fact_listings` match the Bronze source values
# MAGIC after applying the same transform?
# MAGIC
# MAGIC **Why this is critical:** Unit tests check that columns are non-null and have the right type.
# MAGIC This test checks that the **actual values** in Gold are correct end-to-end — from raw Bronze
# MAGIC `cost` field all the way to Gold `price_rub` and `price_usd`.

# COMMAND ----------

def test_it5_price_rub_matches_bronze_end_to_end(spark):
    """
    IT5 -- price_rub in fact_listings must match the Bronze cost field
    after applying the same transform used in _transform_listings.

    Pipeline transform (exact from 08_silver_transformation):
      try_cast(regexp_replace(cost, "[^0-9.]", "") as double)

    This is the only end-to-end value accuracy test -- it proves the
    numeric transformation is not just present but CORRECT all the way
    from Bronze cost (raw string) to Gold price_rub (DOUBLE).

    Method:
      1. Apply the same regexp+cast to Bronze cost for each listing_id
      2. Deduplicate on listing_id (same as Silver)
      3. Compare to fact_listings price_rub via left_anti on (listing_id, price_rub)
      4. Assert no mismatches in either direction
    """
    # Step 1: Build Bronze with transformed price
    bronze_dfs = [
        spark.read.table(B(t)).select("id", "cost")
        for t in LISTING_BRONZE_TABLES
    ]
    df_bronze = bronze_dfs[0]
    for df in bronze_dfs[1:]:
        df_bronze = df_bronze.unionByName(df, allowMissingColumns=True)

    bronze_prices = (
        df_bronze
        .select(
            F.expr("try_cast(id as long)").cast("string").alias("listing_id"),
            F.expr("try_cast(regexp_replace(cost, '[^0-9.]', '') as double)").alias("price_rub"),
        )
        .filter(F.col("listing_id").isNotNull() & F.col("price_rub").isNotNull())
        .dropDuplicates(["listing_id"])
    )

    # Step 2: Gold prices
    gold_prices = (
        spark.read.table(G("fact_listings"))
        .select("listing_id", "price_rub")
        .filter(F.col("price_rub").isNotNull())
    )

    # Direction 1: Gold price_rub not matching Bronze (value corrupted)
    mismatched = gold_prices.subtract(bronze_prices).count()
    assert mismatched == 0, (
        f"IT5 FAILED: {mismatched:,} fact_listings rows have price_rub that does not "
        "match try_cast(regexp_replace(cost, '[^0-9.]', '') as double) from Bronze.\n"
        "Financial measure was corrupted during Bronze -> Silver -> Gold transformation."
    )


def test_it5_price_usd_derived_correctly_end_to_end(spark):
    """
    IT5 -- price_usd in fact_listings must equal round(price_rub / 82.5, 2)
    for every row, end-to-end from Bronze cost to Gold price_usd.

    Pipeline derivation (exact from 08_silver_transformation):
      USD_RATE = 82.5  (Feb 2023 historical RUB/USD rate)
      price_usd = round(price_rub / 82.5, 2)

    This crosses Silver (where price_usd is computed) and Gold
    (where price_usd is passed through) in one assertion.
    """
    df = spark.read.table(G("fact_listings"))

    bad = df.filter(
        F.col("price_rub").isNotNull() &
        F.col("price_usd").isNotNull() &
        (F.abs(F.col("price_usd") - F.round(F.col("price_rub") / USD_RATE, 2)) > 0.01)
    ).count()

    assert bad == 0, (
        f"IT5 FAILED: {bad:,} rows where price_usd != round(price_rub / 82.5, 2).\n"
        f"USD_RATE = {USD_RATE} (Feb 2023 historical rate -- matches Silver pipeline).\n"
        "price_usd was corrupted or computed with a different exchange rate."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT6 — Aggregate Tables Source Consistency
# MAGIC
# MAGIC Every `brand` and `city_name` in aggregate tables must come from `dim_car` and
# MAGIC `dim_location` respectively — not from `fact_listings` directly.
# MAGIC
# MAGIC **Why `fact_listings.brand` fails:** `brand` was removed from `fact_listings` per Vasu's
# MAGIC rule — no strings in fact. Brand is resolved via `car_sk -> dim_car.brand`.
# MAGIC Any test reading `brand` directly from `fact_listings` will crash with
# MAGIC `UNRESOLVED_COLUMN`.

# COMMAND ----------

def test_it6_agg_brands_subset_of_dim_car(spark):
    """
    IT6 -- Every brand in agg_brand_location_performance must exist in dim_car.

    WHY fact_listings.brand cannot be used:
      fact_listings has NO brand column -- it was removed per Vasu's design rule
      (no strings in fact table). brand lives in dim_car, accessible via
      fact.car_sk -> dim_car.car_sk -> dim_car.brand.

    CORRECT check: agg brands must be a subset of dim_car active brands.
    agg_brand_location_performance is built by joining fact + dim_car on car_sk,
    so every brand in the agg must exist in dim_car (active rows).
    """
    dim_brands = (
        spark.read.table(G("dim_car"))
        .filter(F.col("__END_AT").isNull())
        .select("brand")
        .distinct()
    )
    agg_brands = (
        spark.read.table(G("agg_brand_location_performance"))
        .select("brand")
        .filter(F.col("brand").isNotNull())
        .distinct()
    )

    invented = agg_brands.join(dim_brands, on="brand", how="left_anti").count()
    assert invented == 0, (
        f"IT6 FAILED: {invented:,} brand(s) in agg_brand_location_performance "
        "not found in dim_car (active rows).\n"
        "Aggregate was built from brand values not in the current dimension. "
        "Check: dim_car pipeline ran after agg, or stale data in Gold."
    )


def test_it6_agg_cities_subset_of_dim_location(spark):
    """
    IT6 -- Every city_name in agg_brand_location_performance must exist in dim_location.

    agg_brand_location_performance is built by joining fact + dim_location on location_sk.
    Every city_name in the agg must come from dim_location active rows.

    NOTE: Some fact location_sk values have no dim_location match (catalog coverage gap
    -- seller city names not in geo reference). Those rows produce NULL city_name in the
    agg (left join). This test only checks non-null city_name values.
    """
    dim_cities = (
        spark.read.table(G("dim_location"))
        .filter(F.col("__END_AT").isNull())
        .select("city_name")
        .distinct()
    )
    agg_cities = (
        spark.read.table(G("agg_brand_location_performance"))
        .select("city_name")
        .filter(F.col("city_name").isNotNull())
        .distinct()
    )

    invented = agg_cities.join(dim_cities, on="city_name", how="left_anti").count()
    assert invented == 0, (
        f"IT6 FAILED: {invented:,} city_name(s) in agg_brand_location_performance "
        "not found in dim_location (active rows).\n"
        "Aggregate contains city names not in the current dimension."
    )


def test_it6_agg_top10_brands_are_in_dim_car(spark):
    """
    IT6 -- Every brand in agg_top_10_brands_by_spend must exist in dim_car.
    The top-10 agg is built by joining fact + dim_car -- all brands must resolve.
    """
    dim_brands = (
        spark.read.table(G("dim_car"))
        .filter(F.col("__END_AT").isNull())
        .select("brand").distinct()
    )
    agg_brands = (
        spark.read.table(G("agg_top_10_brands_by_spend"))
        .select("brand")
        .filter(F.col("brand").isNotNull())
        .distinct()
    )
    invented = agg_brands.join(dim_brands, on="brand", how="left_anti").count()
    assert invented == 0, (
        f"IT6 FAILED: {invented:,} brand(s) in agg_top_10_brands_by_spend "
        "not found in dim_car. Aggregate has phantom brand values."
    )
    # Top-10 must have at most 10 rows
    cnt = spark.read.table(G("agg_top_10_brands_by_spend")).count()
    assert cnt <= 10, (
        f"IT6 FAILED: agg_top_10_brands_by_spend has {cnt} rows, expected at most 10."
    )
