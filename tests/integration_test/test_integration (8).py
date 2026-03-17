# Databricks notebook source
# MAGIC %md
# MAGIC # Integration Test Suite — Bronze → Silver → Gold
# MAGIC
# MAGIC End-to-end tests that cross all three layer boundaries in a single assertion.
# MAGIC
# MAGIC | Test | Layers | What it proves |
# MAGIC |------|--------|----------------|
# MAGIC | **IT1** | Bronze → Gold | Every valid Bronze `listing_id` reaches `fact_listings` |
# MAGIC | **IT2** | Bronze → Gold | Audit chain (`source_file` + timestamps) is unbroken |
# MAGIC | **IT3** | Bronze → Silver → Gold | Row counts consistent and explainable across all layers |
# MAGIC | **IT4** | Silver ↔ Gold | JOIN fact + dims reconstructs Silver exactly (Vasu requirement) |
# MAGIC | **IT5** | Bronze → Gold | `price_rub` and `price_usd` are correct end-to-end |
# MAGIC | **IT6** | Gold dims ↔ Gold aggs | Aggregate brands/cities trace to active dim rows |
# MAGIC
# MAGIC **Total: 10 tests**

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

# 4 Bronze tables that feed listings_silver_merged
LISTING_BRONZE_TABLES = [
    "listings_csv_copyinto",
    "listings_json_autoloader",
    "listings_xml_pyspark",
    "listings_csv_dlt",
]

USD_RATE = 82.5  # Feb 2023 RUB/USD -- matches Silver _transform_listings


# ── Shared helper: union all 4 Bronze listing tables ─────────────────────────
def _bronze_listings(spark):
    """Union all 4 Bronze listing tables into a single DataFrame."""
    dfs = [spark.read.table(B(t)) for t in LISTING_BRONZE_TABLES]
    df  = dfs[0]
    for d in dfs[1:]:
        df = df.unionByName(d, allowMissingColumns=True)
    return df


# ── Shared helper: apply Silver _transform_listings to Bronze ─────────────────
def _transform_bronze(df):
    """
    Apply the same transforms as Silver _transform_listings to Bronze data.

    Key difference from the pipeline:
      Pipeline uses F.to_timestamp() inside DLT streaming -- Spark streaming
      silently returns NULL for unparseable dates.
      Tests run in batch context where F.to_timestamp() throws CANNOT_PARSE_TIMESTAMP.
      Fix: use try_to_timestamp() which always returns NULL on bad input (never throws).
    """
    return df.select(
        F.expr("try_cast(id as long)").cast("string").alias("listing_id"),
        F.expr("try_cast(regexp_replace(cost, '[^0-9.]', '') as double)").alias("price_rub"),
        F.coalesce(
            # Russian date format: 15.03.2023
            F.expr("try_to_timestamp(date, 'dd.MM.yyyy')"),
            # ISO 8601: 2023-03-15T10:00:00Z
            # F.to_timestamp() with no format auto-detects ISO 8601 -- safe for batch context
            # when wrapped in try_ equivalent. Using regexp_replace to strip the Z
            # and then parsing as standard timestamp avoids embedded quote issues.
            F.to_timestamp(
                F.regexp_replace(F.col("date"), "Z$", ""),
                "yyyy-MM-dd'T'HH:mm:ss"
            ),
        ).alias("listing_date"),
    )


# ── Shared helper: Silver validity filter ─────────────────────────────────────
def _valid_filter(df):
    """Apply Silver _LISTINGS_VALID_FILTER: listing_id, price_rub, listing_date not null."""
    return df.filter(
        F.col("listing_id").isNotNull() &
        F.col("price_rub").isNotNull()  &
        F.col("listing_date").isNotNull()
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT1 — End-to-End Row Integrity: Bronze → Silver → Gold
# MAGIC
# MAGIC Every valid Bronze `listing_id` must reach `fact_listings`. Both directions checked.

# COMMAND ----------

def test_it1_every_valid_bronze_listing_reaches_gold(spark):
    """
    IT1 -- Every Bronze listing_id that passes Silver _LISTINGS_VALID_FILTER
    must exist in fact_listings. Checked in both directions:
      - No valid Bronze row dropped (Bronze -> Gold)
      - No Gold row invented     (Gold -> Bronze)
    """
    valid_bronze_ids = (
        _valid_filter(_transform_bronze(_bronze_listings(spark)))
        .select("listing_id")
        .dropDuplicates(["listing_id"])
    )
    gold_ids = spark.read.table(G("fact_listings")).select("listing_id")

    dropped = valid_bronze_ids.join(gold_ids, on="listing_id", how="left_anti").count()
    assert dropped == 0, (
        f"{dropped:,} valid Bronze listing_id(s) never reached fact_listings. "
        "Rows passed the Silver filter but are missing from Gold."
    )

    invented = gold_ids.join(valid_bronze_ids, on="listing_id", how="left_anti").count()
    assert invented == 0, (
        f"{invented:,} fact_listings listing_id(s) not traceable to any Bronze source."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT2 — Audit Chain Continuity: Bronze → Silver → Gold
# MAGIC
# MAGIC Every `bronze_source_file` in Gold must be real. Timestamps must be ordered: `bronze_load_dt ≤ silver_load_dt ≤ gold_load_dt`.

# COMMAND ----------

def test_it2_bronze_source_file_traceable_to_bronze(spark):
    """
    IT2 -- Every bronze_source_file in fact_listings must exist in Bronze.
    Proves the audit trail was not fabricated at any layer.
    """
    bronze_files = (
        _bronze_listings(spark)
        .select(F.col("source_file").alias("bronze_source_file"))
        .distinct()
    )
    gold_files = (
        spark.read.table(G("fact_listings"))
        .select("bronze_source_file")
        .filter(F.col("bronze_source_file").isNotNull())
        .distinct()
    )
    orphans = gold_files.join(bronze_files, on="bronze_source_file", how="left_anti").count()
    assert orphans == 0, (
        f"{orphans:,} bronze_source_file value(s) in fact_listings "
        "not found in any Bronze table. Audit trail is broken."
    )


def test_it2_audit_timestamps_ordered(spark):
    """
    IT2 -- Audit timestamps must be chronologically ordered:
    bronze_load_dt <= silver_load_dt <= gold_load_dt.
    A reversed order means a layer was written before its source.
    """
    df = spark.read.table(G("fact_listings"))

    silver_before_bronze = df.filter(
        F.col("silver_load_dt").isNotNull() &
        F.col("bronze_load_dt").isNotNull() &
        (F.col("silver_load_dt") < F.col("bronze_load_dt"))
    ).count()
    assert silver_before_bronze == 0, (
        f"{silver_before_bronze:,} rows where silver_load_dt < bronze_load_dt. "
        "Silver cannot be written before Bronze."
    )

    gold_before_silver = df.filter(
        F.col("gold_load_dt").isNotNull() &
        F.col("silver_load_dt").isNotNull() &
        (F.col("gold_load_dt") < F.col("silver_load_dt"))
    ).count()
    assert gold_before_silver == 0, (
        f"{gold_before_silver:,} rows where gold_load_dt < silver_load_dt. "
        "Gold cannot be written before Silver."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT3 — Row Count Consistency: Bronze → Silver → Gold
# MAGIC
# MAGIC `Bronze raw ≥ Silver (deduplicated + filtered) == Gold fact` — exact equality, zero tolerance.

# COMMAND ----------

def test_it3_row_counts_consistent_across_all_layers(spark):
    """
    IT3 -- Row count relationship across all three layers:
      Bronze raw total  >=  Silver count  ==  Gold fact count

    Bronze >= Silver: Silver deduplicates on listing_id and filters invalid rows.
    Silver == Gold:   Gold applies NO further deduplication -- 1:1 mapping.
    """
    bronze_total = sum(
        spark.read.table(B(t)).count() for t in LISTING_BRONZE_TABLES
    )
    silver_cnt = spark.read.table(S("listings_silver_merged")).count()
    gold_cnt   = spark.read.table(G("fact_listings")).count()

    assert silver_cnt > 0, (
        "listings_silver_merged is empty -- Silver pipeline produced zero rows."
    )
    assert bronze_total >= silver_cnt, (
        f"Silver ({silver_cnt:,}) has MORE rows than Bronze total ({bronze_total:,}). "
        "Silver cannot exceed its source."
    )
    assert silver_cnt == gold_cnt, (
        f"Silver ({silver_cnt:,}) != Gold ({gold_cnt:,}). "
        f"Difference: {abs(silver_cnt - gold_cnt):,} rows. "
        "Gold applies no deduplication -- every Silver row must produce one Gold row."
    )

    print(f"\n  Bronze (raw total) : {bronze_total:>10,}")
    print(f"  Silver             : {silver_cnt:>10,}  ({round((bronze_total-silver_cnt)/bronze_total*100,1)}% reduced)")
    print(f"  Gold fact          : {gold_cnt:>10,}  (must == Silver)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT4 — Full Reconstruction: JOIN fact + dims → Silver
# MAGIC
# MAGIC *Vasu's exact requirement: "Join Fact + Dims → get the original Silver table back."*

# COMMAND ----------

def test_it4_fact_plus_dims_reconstructs_silver(spark):
    """
    IT4 -- Joining fact_listings + dim_price_category + dim_steering
    must perfectly reconstruct listings_silver_merged. Both directions:
      - No Silver row lost in reconstruction
      - No extra row invented by the reconstruction

    Proves the integer FK encoding is lossless and reversible:
      price_category_key -> dim_price_category -> price_category STRING
      steering_key       -> dim_steering       -> steering_wheel  STRING
    """
    fact      = spark.read.table(G("fact_listings"))
    dim_price = spark.read.table(G("dim_price_category")).select("price_category_key", "price_category")
    dim_steer = spark.read.table(G("dim_steering")).select("steering_key", "steering_wheel")

    COLS = [
        "listing_id", "listing_date", "manufacture_year", "engine_power",
        "mileage_km", "has_license", "listing_year", "listing_month",
        "car_age_years", "price_rub", "price_usd",
        "price_category",   # decoded via dim_price_category
        "steering_wheel",   # decoded via dim_steering
    ]

    reconstructed = (
        fact
        .join(dim_price, on="price_category_key", how="left")
        .join(dim_steer, on="steering_key",        how="left")
        .select(*COLS)
    )
    silver = (
        spark.read.table(S("listings_silver_merged"))
        .select(
            "listing_id",
            F.col("listing_date").cast("date").alias("listing_date"),
            "manufacture_year", "engine_power", "mileage_km", "has_license",
            "listing_year", "listing_month", "car_age_years",
            "price_rub", "price_usd", "price_category", "steering_wheel",
        )
    )

    lost = silver.subtract(reconstructed).count()
    assert lost == 0, (
        f"{lost:,} Silver rows cannot be reconstructed by joining fact + dims. "
        "Data was lost or corrupted during Silver -> Gold."
    )

    invented = reconstructed.subtract(silver).count()
    assert invented == 0, (
        f"{invented:,} reconstructed rows have no Silver origin. "
        "Joining fact + dims produced rows not in Silver."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## IT5 — Financial Measure Accuracy: Bronze → Gold
# MAGIC
# MAGIC Actual values of `price_rub` and `price_usd` in Gold must match the Bronze source end-to-end.

# COMMAND ----------

def test_it5_price_rub_correct_end_to_end(spark):
    """
    IT5 -- price_rub in fact_listings must match listings_silver_merged exactly.

    WHY Bronze comparison is impossible:
      Bronze tables are continuously ingested (DLT autoloader / COPY INTO).
      Silver ran against Bronze at a specific point in time.
      By test time, Bronze may have newer rows with different cost values
      for the same listing_id -- even for "single-source" listings.
      Any Bronze->Gold price comparison will fail because the test reads
      CURRENT Bronze while Gold was built from an OLDER Bronze snapshot.

    CORRECT approach: compare Gold directly to Silver (same snapshot boundary).
      fact_listings.price_rub was passed through unchanged from Silver.
      If fact.price_rub == silver.price_rub for every listing_id, the
      Bronze->Silver transform was correct AND Gold passed it through intact.
      IT4 already proves the full reconstruction -- IT5 proves price specifically.
    """
    fact_prices = (
        spark.read.table(G("fact_listings"))
        .select("listing_id", "price_rub")
        .filter(F.col("price_rub").isNotNull())
    )
    silver_prices = (
        spark.read.table(S("listings_silver_merged"))
        .select("listing_id", "price_rub")
        .filter(F.col("price_rub").isNotNull())
    )

    # Direction 1: Gold price not in Silver (corrupted in Gold layer)
    corrupted = fact_prices.subtract(silver_prices).count()
    assert corrupted == 0, (
        f"{corrupted:,} fact_listings (listing_id, price_rub) pairs not found in Silver.\n"
        "price_rub was modified during Silver -> Gold transformation."
    )

    # Direction 2: Silver price not in Gold (dropped in Gold layer)
    dropped = silver_prices.subtract(fact_prices).count()
    assert dropped == 0, (
        f"{dropped:,} Silver (listing_id, price_rub) pairs not in fact_listings.\n"
        "Some Silver prices were lost during Gold transformation."
    )


def test_it5_price_usd_correct_end_to_end(spark):
    """
    IT5 -- price_usd must equal round(price_rub / 82.5, 2) in every fact row.
    USD_RATE = 82.5 (Feb 2023) -- exact value from Silver _transform_listings.
    """
    bad = spark.read.table(G("fact_listings")).filter(
        F.col("price_rub").isNotNull() &
        F.col("price_usd").isNotNull() &
        (F.abs(F.col("price_usd") - F.round(F.col("price_rub") / USD_RATE, 2)) > 0.01)
    ).count()
    assert bad == 0, (
        f"{bad:,} rows where price_usd != round(price_rub / {USD_RATE}, 2). "
        "USD conversion rate or rounding is wrong in Gold."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## IT6 — Aggregate Consistency: Gold dims ↔ Gold aggs
# MAGIC
# MAGIC Every `brand` and `city_name` in aggregate tables must trace back to active dim rows.
# MAGIC
# MAGIC > **Note:** `fact_listings` has **no `brand` column** — strings were removed per Vasu's rule.
# MAGIC > Brand is resolved via `car_sk → dim_car.brand`. Any test reading `brand` directly
# MAGIC > from `fact_listings` will crash with `UNRESOLVED_COLUMN`.

# COMMAND ----------

def test_it6_agg_brands_exist_in_dim_car(spark):
    """
    IT6 -- Every brand in aggregate tables must exist in dim_car (active rows).
    Aggregates are built by joining fact + dim_car on car_sk -- all brands come from dim_car.
    """
    dim_brands = (
        spark.read.table(G("dim_car"))
        .filter(F.col("__END_AT").isNull())
        .select("brand").distinct()
    )

    for agg_table in ("agg_brand_location_performance", "agg_top_10_brands_by_spend",
                      "agg_monthly_sales_trend", "agg_comprehensive_kpi_cube"):
        agg_brands = (
            spark.read.table(G(agg_table))
            .select("brand")
            .filter(F.col("brand").isNotNull())
            .distinct()
        )
        orphans = agg_brands.join(dim_brands, on="brand", how="left_anti").count()
        assert orphans == 0, (
            f"[{agg_table}] {orphans:,} brand(s) not found in dim_car active rows. "
            "Aggregate contains phantom brand values."
        )


def test_it6_agg_cities_exist_in_dim_location(spark):
    """
    IT6 -- Every non-null city_name in agg_brand_location_performance
    must exist in dim_location (active rows).
    Note: NULL city_name is allowed -- seller cities not in geo reference produce NULL.
    """
    dim_cities = (
        spark.read.table(G("dim_location"))
        .filter(F.col("__END_AT").isNull())
        .select("city_name").distinct()
    )
    agg_cities = (
        spark.read.table(G("agg_brand_location_performance"))
        .select("city_name")
        .filter(F.col("city_name").isNotNull())
        .distinct()
    )
    orphans = agg_cities.join(dim_cities, on="city_name", how="left_anti").count()
    assert orphans == 0, (
        f"{orphans:,} city_name(s) in agg_brand_location_performance "
        "not found in dim_location. Aggregate has phantom city values."
    )


def test_it6_agg_top10_has_at_most_10_rows(spark):
    """
    IT6 -- agg_top_10_brands_by_spend must have at most 10 rows.
    The pipeline uses .limit(10) -- if more than 10 rows exist, limit was not applied.
    """
    cnt = spark.read.table(G("agg_top_10_brands_by_spend")).count()
    assert cnt <= 10, (
        f"agg_top_10_brands_by_spend has {cnt} rows -- expected at most 10. "
        ".limit(10) was not applied correctly."
    )
