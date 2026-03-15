# Databricks notebook source
# MAGIC %md
# MAGIC # Integration Test Suite -- Bronze -> Silver -> Gold Pipeline
# MAGIC
# MAGIC Tests the **cross-layer contracts** that unit tests on individual layers cannot catch.
# MAGIC
# MAGIC | Test | Contract | Layers |
# MAGIC |------|----------|--------|
# MAGIC | T-1 | Every Bronze row reaches Silver -- no silent drops | Bronze -> Silver |
# MAGIC | T-2 | Silver `listing_id` PKs flow into `fact_listings` with zero loss and zero invention | Silver -> Gold |
# MAGIC | T-3 | Full audit chain unbroken: `bronze_load_dt` <= `silver_load_dt` <= `gold_load_dt` non-null on every fact row | Bronze -> Silver -> Gold |
# MAGIC | T-4 | Every Silver dimension key is an active SCD2 row in Gold | Silver -> Gold |
# MAGIC | T-5 | `fact_listings` derived measures are internally self-consistent | Gold self-check |
# MAGIC | T-6 | Star-schema FK join rates meet the 80% minimum on all dimension joins | Gold self-check |
# MAGIC | T-7 | Aggregate tables are consistent with `fact_listings` | Gold self-check |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config & Fixtures

# COMMAND ----------

@pytest.fixture(scope="session")
def spark():
    return DatabricksSession.builder.getOrCreate()


CONFIG = {
    "catalog": "vstone_catalog",
    "bronze":  "bronze",
    "silver":  "silver",
    "gold":    "gold",
}

BRONZE = f"{CONFIG['catalog']}.{CONFIG['bronze']}"
SILVER = f"{CONFIG['catalog']}.{CONFIG['silver']}"
GOLD   = f"{CONFIG['catalog']}.{CONFIG['gold']}"

# Minimum acceptable FK join rate between fact and any dimension
MIN_JOIN_RATE_PCT = 80.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def _union_bronze(spark, sources: list):
    """Unions Bronze source tables tolerating missing columns."""
    df = None
    for src in sources:
        b  = spark.read.table(src)
        df = b if df is None else df.unionByName(b, allowMissingColumns=True)
    return df


def _id_main(col): return F.expr(f"try_cast(`{col}` as long)").cast("string")
def _id_dbl(col):  return F.col(f"`{col}`").cast("double").cast("long").cast("string")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T-1 -- Bronze -> Silver: No Silent Row Drops
# MAGIC
# MAGIC **Contract:** Every distinct Bronze key must appear in Silver or quarantine

# COMMAND ----------

_BRONZE_TO_SILVER_MAP = [
    {
        "name"       : "listings_silver_merged",
        "sources"    : [
            f"{BRONZE}.listings_csv_copyinto", f"{BRONZE}.listings_json_autoloader",
            f"{BRONZE}.listings_xml_pyspark",  f"{BRONZE}.listings_csv_dlt",
        ],
        "bronze_key" : "id",
        "cast_fn"    : _id_main,
        "silver"     : f"{SILVER}.listings_silver_merged",
        "quarantine" : f"{SILVER}.listings_main_quarantine",
        "silver_key" : "listing_id",
    },
    {
        "name"       : "listings_text_transformation",
        "sources"    : [f"{BRONZE}.listings_text"],
        "bronze_key" : "id",
        "cast_fn"    : _id_dbl,
        "silver"     : f"{SILVER}.listings_text_transformation",
        "quarantine" : f"{SILVER}.listings_text_quarantine",
        "silver_key" : "listing_id",
    },
    {
        "name"       : "listings_photo_transformation",
        "sources"    : [f"{BRONZE}.listings_photo"],
        "bronze_key" : "id",
        "cast_fn"    : _id_dbl,
        "silver"     : f"{SILVER}.listings_photo_transformation",
        "quarantine" : f"{SILVER}.listings_photo_quarantine",
        "silver_key" : "listing_id",
    },
    {
        "name"       : "geography_transformation",
        "sources"    : [f"{BRONZE}.geo_locations"],
        "bronze_key" : "name_padesh",
        "cast_fn"    : lambda col: F.col(f"`{col}`").cast("string"),
        "silver"     : f"{SILVER}.geography_transformation",
        "quarantine" : f"{SILVER}.geography_quarantine",
        "silver_key" : "city_name",
    },
]

_IT1_PARAMS = [pytest.param(e, id=e["name"]) for e in _BRONZE_TO_SILVER_MAP]


@pytest.mark.parametrize("entry", _IT1_PARAMS)
def test_it1_no_silent_drops_bronze_to_silver(spark, entry):
    """
    IT-1: Every distinct Bronze key must land in Silver OR Quarantine.
    Uses left_anti join: Bronze keys absent from both destinations = silent drop.

    Why cross-layer: unit tests check Silver and Bronze independently.
    Only here do we verify the Bronze->Silver handoff is complete end-to-end.
    """
    # All distinct non-null Bronze keys after normalisation
    bronze_keys = (
        _union_bronze(spark, entry["sources"])
        .select(entry["cast_fn"](entry["bronze_key"]).alias("_key"))
        .filter(F.col("_key").isNotNull())
        .distinct()
    )

    # Silver keys
    silver_keys = (
        spark.read.table(entry["silver"])
        .select(F.col(entry["silver_key"]).cast("string").alias("_key"))
        .filter(F.col("_key").isNotNull())
    )

    # Quarantine keys -- same silver_key column exists in quarantine
    quar_df = spark.read.table(entry["quarantine"])
    quar_key_col = entry["silver_key"] if entry["silver_key"] in quar_df.columns else quar_df.columns[0]
    quar_keys = (
        quar_df
        .select(F.col(quar_key_col).cast("string").alias("_key"))
        .filter(F.col("_key").isNotNull())
    )

    destination_keys = silver_keys.unionByName(quar_keys).distinct()
    silent_drops     = bronze_keys.join(destination_keys, on="_key", how="left_anti").count()

    assert silent_drops == 0, (
        f"[IT-1 | {entry['name']}] {silent_drops:,} Bronze key(s) vanished -- "
        "not found in Silver OR Quarantine. Silent data loss detected."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## T-2 -- Silver -> Gold: `listing_id` Lineage Through `fact_listings`
# MAGIC
# MAGIC **Contract:** Every `listing_id` in `listings_silver_merged` must appear in
# MAGIC `fact_listings`, with no drops and no invented IDs.

# COMMAND ----------

def test_it2_silver_listing_ids_fully_in_fact(spark):
    """
    IT-2a: Every Silver listing_id must exist in fact_listings (no drops).
    A missing ID means a listing was lost crossing from Silver to Gold.
    """
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id").distinct()
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id").distinct()
    dropped    = silver_ids.join(gold_ids, on="listing_id", how="left_anti").count()

    assert dropped == 0, (
        f"[IT-2a] {dropped:,} Silver listing_id(s) are missing from fact_listings. "
        "Listings were lost in the Silver -> Gold transformation."
    )


def test_it2_fact_contains_no_invented_listing_ids(spark):
    """
    IT-2b: fact_listings must not contain listing_ids absent from Silver.
    An invented ID means the Gold pipeline fabricated a record with no Silver origin.
    """
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id").distinct()
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id").distinct()
    invented   = gold_ids.join(silver_ids, on="listing_id", how="left_anti").count()

    assert invented == 0, (
        f"[IT-2b] {invented:,} fact_listings listing_id(s) have no Silver origin. "
        "The Gold pipeline invented records not present in Silver."
    )


def test_it2_fact_row_count_within_tolerance_of_silver(spark):
    """
    IT-2c: fact_listings row count must be within 1% of listings_silver_merged.
    Larger divergence signals a bulk join explosion or accidental deduplication.
    """
    silver_cnt = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    gold_cnt   = spark.read.table(f"{GOLD}.fact_listings").count()
    diff_pct   = abs(gold_cnt - silver_cnt) / max(silver_cnt, 1) * 100

    assert diff_pct <= 1.0, (
        f"[IT-2c] fact_listings row count diverges from Silver by {diff_pct:.3f}% "
        f"(Silver={silver_cnt:,} | Gold={gold_cnt:,}). "
        "Likely cause: join fanout or unintended deduplication in Gold pipeline."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## T-3 -- Unbroken Audit Chain: Bronze -> Silver -> Gold
# MAGIC
# MAGIC **Contract:** `fact_listings` must carry all four audit columns fully populated
# MAGIC and in the correct chronological order:
# MAGIC `bronze_load_dt` <= `silver_load_dt` <= `gold_load_dt`.

# COMMAND ----------

def test_it3_full_audit_chain_present_and_non_null(spark):
    """
    IT-3a: All four audit columns must exist and be fully populated in fact_listings.
    A null in any audit column means the lineage chain is broken at that layer boundary.
    """
    df          = spark.read.table(f"{GOLD}.fact_listings")
    audit_chain = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]

    missing = [c for c in audit_chain if c not in df.columns]
    assert missing == [], (
        f"[IT-3a] fact_listings missing audit chain columns: {missing}. "
        "The full Bronze->Silver->Gold lineage cannot be traced."
    )
    for col in audit_chain:
        null_cnt = df.filter(F.col(col).isNull()).count()
        assert null_cnt == 0, (
            f"[IT-3a] fact_listings.{col} has {null_cnt:,} NULL rows. "
            "Audit chain is broken at this column."
        )


def test_it3_audit_timestamps_are_chronologically_ordered(spark):
    """
    IT-3b: bronze_load_dt must not be later than silver_load_dt,
    and silver_load_dt must not be later than gold_load_dt.

    A violation means a timestamp was set incorrectly -- e.g. silver_load_dt
    was set to current_timestamp() during a Gold pipeline re-run, making it
    appear newer than gold_load_dt.
    """
    df = spark.read.table(f"{GOLD}.fact_listings").filter(
        F.col("bronze_load_dt").isNotNull() &
        F.col("silver_load_dt").isNotNull() &
        F.col("gold_load_dt").isNotNull()
    )
    bronze_after_silver = df.filter(F.col("bronze_load_dt") > F.col("silver_load_dt")).count()
    silver_after_gold   = df.filter(F.col("silver_load_dt") > F.col("gold_load_dt")).count()

    assert bronze_after_silver == 0, (
        f"[IT-3b] {bronze_after_silver:,} rows where bronze_load_dt > silver_load_dt. "
        "Bronze timestamp is newer than Silver -- audit clock is inverted."
    )
    assert silver_after_gold == 0, (
        f"[IT-3b] {silver_after_gold:,} rows where silver_load_dt > gold_load_dt. "
        "Silver timestamp is newer than Gold -- audit clock is inverted."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## T-4 -- Silver -> SCD2 Gold Dimensions: No Key Loss or Invention
# MAGIC
# MAGIC **Contract:** Every distinct Silver dimension key must have exactly one active
# MAGIC (`__END_AT IS NULL`) row in its corresponding Gold SCD2 table.

# COMMAND ----------

_SCD2_REGISTRY = [
    {
        "name"   : "dim_car",
        "silver" : f"{SILVER}.car_catalog_transformation",
        "keys"   : ["brand", "model"],
    },
    {
        "name"   : "dim_location",
        "silver" : f"{SILVER}.geography_transformation",
        "keys"   : ["city_prepositional"],
    },
    {
        "name"   : "dim_listing_details",
        "silver" : f"{SILVER}.listings_text_transformation",
        "keys"   : ["listing_id"],
    },
    {
        "name"   : "dim_listing_photos",
        "silver" : f"{SILVER}.listings_photo_transformation",
        "keys"   : ["listing_id", "photo_url_clean"],
    },
]

_SCD2_PARAMS = [pytest.param(e, id=e["name"]) for e in _SCD2_REGISTRY]


@pytest.mark.parametrize("entry", _SCD2_PARAMS)
def test_it4_silver_keys_present_as_active_scd2_rows(spark, entry):
    """
    IT-4a: Every distinct Silver key must appear as an active Gold SCD2 row.
    Missing active rows mean apply_changes dropped or closed a key that should
    still be current.
    """
    silver_keys = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_active = (
        spark.read.table(f"{GOLD}.{entry['name']}")
        .filter(F.col("__END_AT").isNull())
        .select(*entry["keys"])
        .distinct()
    )
    missing = silver_keys.join(gold_active, on=entry["keys"], how="left_anti").count()

    assert missing == 0, (
        f"[IT-4a | {entry['name']}] {missing:,} Silver key(s) have no active Gold row. "
        f"Keys checked: {entry['keys']}. "
        "apply_changes may have dropped or prematurely closed these keys."
    )


@pytest.mark.parametrize("entry", _SCD2_PARAMS)
def test_it4_scd2_no_invented_gold_keys(spark, entry):
    """
    IT-4b: Gold active SCD2 rows must not contain keys absent from Silver.
    An invented key means apply_changes wrote a Gold row with no Silver provenance.
    """
    silver_keys = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_active = (
        spark.read.table(f"{GOLD}.{entry['name']}")
        .filter(F.col("__END_AT").isNull())
        .select(*entry["keys"])
        .distinct()
    )
    invented = gold_active.join(silver_keys, on=entry["keys"], how="left_anti").count()

    assert invented == 0, (
        f"[IT-4b | {entry['name']}] {invented:,} active Gold key(s) not found in Silver. "
        f"Keys: {entry['keys']}. Gold has fabricated dimension records."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## T-5 -- `fact_listings` Derived Measures Self-Consistency
# MAGIC
# MAGIC **Contract:** Gold-level derived columns (`car_age_at_listing`, `is_high_mileage`,
# MAGIC `price_per_hp_usd`, `photo_count`) must be internally consistent with the source
# MAGIC columns in the same row.

# COMMAND ----------

def test_it5_car_age_at_listing_consistent_with_columns(spark):
    """
    IT-5a: car_age_at_listing must equal year(listing_date) - manufacture_year.
    Gold computes this independently of Silver car_age_years (which uses lit(2023)).
    A mismatch means the Gold pipeline used the wrong column or year reference.
    """
    df = spark.read.table(f"{GOLD}.fact_listings").filter(
        F.col("listing_date").isNotNull() &
        F.col("manufacture_year").isNotNull() &
        F.col("car_age_at_listing").isNotNull()
    )
    bad = df.filter(
        F.col("car_age_at_listing") !=
        (F.year(F.col("listing_date")) - F.col("manufacture_year"))
    ).count()

    assert bad == 0, (
        f"[IT-5a] {bad:,} rows where car_age_at_listing != year(listing_date) - manufacture_year. "
        "Gold derived column is inconsistent with its source columns."
    )


def test_it5_is_high_mileage_consistent_with_mileage_km(spark):
    """
    IT-5b: is_high_mileage must be True iff mileage_km > 100,000.
    Mirrors the Gold pipeline: F.when(mileage_km > 100000, True).otherwise(False).
    """
    df = spark.read.table(f"{GOLD}.fact_listings").filter(
        F.col("mileage_km").isNotNull() & F.col("is_high_mileage").isNotNull()
    )
    bad = df.filter(
        ((F.col("mileage_km") > 100000)  & ~F.col("is_high_mileage")) |
        ((F.col("mileage_km") <= 100000) &  F.col("is_high_mileage"))
    ).count()

    assert bad == 0, (
        f"[IT-5b] {bad:,} rows where is_high_mileage contradicts mileage_km > 100,000. "
        "Boolean flag is inconsistent with its source measure."
    )


def test_it5_photo_count_is_non_negative(spark):
    """
    IT-5c: photo_count must be >= 0 for every row.
    Gold computes this via a left join + coalesce(count, 0).
    A null or negative value signals a join error in the photo denormalization step.
    """
    df  = spark.read.table(f"{GOLD}.fact_listings")
    bad = df.filter(
        F.col("photo_count").isNull() | (F.col("photo_count") < 0)
    ).count()

    assert bad == 0, (
        f"[IT-5c] {bad:,} rows with null or negative photo_count. "
        "Gold photo denormalization join (coalesce -> 0) is broken."
    )


def test_it5_price_per_hp_null_only_when_engine_or_price_missing(spark):
    """
    IT-5d: price_per_hp_usd must be non-null whenever engine_power > 0 and price_usd > 0.
    Gold computes: round(price_usd / nullif(engine_power, 0), 2).
    A null when both inputs are valid means the nullif() logic is wrong.
    """
    df  = spark.read.table(f"{GOLD}.fact_listings")
    bad = df.filter(
        F.col("engine_power").isNotNull() & (F.col("engine_power") > 0) &
        F.col("price_usd").isNotNull()    & (F.col("price_usd")    > 0) &
        F.col("price_per_hp_usd").isNull()
    ).count()

    assert bad == 0, (
        f"[IT-5d] {bad:,} rows where engine_power > 0 and price_usd > 0 "
        "but price_per_hp_usd is NULL. "
        "Gold nullif(engine_power, 0) division is not working correctly."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## T-6 -- Star Schema FK Join Rates >= 80%
# MAGIC
# MAGIC **Contract:** Joining `fact_listings` to each dimension must match at least 60%
# MAGIC of fact rows. A low join rate means FK values in the fact table do not align with
# MAGIC dimension keys -- making the star schema analytically broken.

# COMMAND ----------

_JOIN_CHECKS = [
    {
        "dim"  : "dim_date",
        "sql"  : (
            "SELECT COUNT(*) AS c "
            f"FROM {GOLD}.fact_listings f "
            f"JOIN {GOLD}.dim_date d ON cast(f.listing_date AS date) = d.date_key"
        ),
        "desc" : "fact.listing_date -> dim_date.date_key",
    },
    {
        "dim"  : "dim_car",
        "sql"  : (
            "SELECT COUNT(*) AS c "
            f"FROM {GOLD}.fact_listings f "
            f"JOIN {GOLD}.dim_car d "
            "ON lower(trim(f.brand)) = lower(trim(d.brand)) "
            "AND lower(trim(f.model)) = lower(trim(d.model)) "
            "WHERE d.__END_AT IS NULL"
        ),
        "desc" : "fact.(brand, model) -> dim_car active rows",
    },
    {
        "dim"  : "dim_location",
        "sql"  : (
            "SELECT COUNT(*) AS c "
            f"FROM {GOLD}.fact_listings f "
            f"JOIN {GOLD}.dim_location d ON f.location_key = d.city_prepositional "
            "WHERE d.__END_AT IS NULL"
        ),
        "desc" : "fact.location_key -> dim_location active rows",
    },
    {
        "dim"  : "dim_listing_details",
        "sql"  : (
            "SELECT COUNT(*) AS c "
            f"FROM {GOLD}.fact_listings f "
            f"JOIN {GOLD}.dim_listing_details d ON f.listing_id = d.listing_id "
            "WHERE d.__END_AT IS NULL"
        ),
        "desc" : "fact.listing_id -> dim_listing_details active rows",
    },
    {
        "dim"  : "dim_listing_photos",
        "sql"  : (
            f"SELECT COUNT(DISTINCT f.listing_id) AS c "
            f"FROM {GOLD}.fact_listings f "
            f"JOIN {GOLD}.dim_listing_photos d ON f.listing_id = d.listing_id "
            "WHERE d.__END_AT IS NULL"
        ),
        "desc" : "fact.listing_id -> dim_listing_photos active rows (distinct match)",
    },
]

_JOIN_PARAMS = [pytest.param(e, id=e["dim"]) for e in _JOIN_CHECKS]


@pytest.mark.parametrize("entry", _JOIN_PARAMS)
def test_it6_fact_dim_join_rate_meets_minimum(spark, entry):
    """
    IT-6: fact_listings -> dimension join rate must be >= 60%.
    A low rate means FK values do not align with dimension keys,
    making the star schema broken for analytics.
    """
    fact_cnt = spark.read.table(f"{GOLD}.fact_listings").count()
    joined   = spark.sql(entry["sql"]).collect()[0]["c"]
    rate     = round(joined / max(fact_cnt, 1) * 100, 2)

    assert rate >= MIN_JOIN_RATE_PCT, (
        f"[IT-6 | {entry['dim']}] Join rate {rate:.2f}% is below minimum {MIN_JOIN_RATE_PCT}%. "
        f"Matched={joined:,} / Total={fact_cnt:,}. "
        f"Join: {entry['desc']}."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## T-7 -- Aggregate Tables Consistent with `fact_listings`
# MAGIC
# MAGIC **Contract:** All 5 aggregate tables must be non-empty, their listing volume
# MAGIC totals must reconcile with `fact_listings`, and `agg_top_10_brands_by_spend`
# MAGIC must contain <= 10 rows.

# COMMAND ----------

_AGG_TABLES = [
    "agg_monthly_sales_trend",
    "agg_brand_location_performance",
    "agg_regional_market_depth",
    "agg_comprehensive_kpi_cube",
    "agg_top_10_brands_by_spend",
]
_AGG_PARAMS = [pytest.param(t, id=t) for t in _AGG_TABLES]


@pytest.mark.parametrize("agg_table", _AGG_PARAMS)
def test_it7_aggregate_tables_are_non_empty(spark, agg_table):
    """
    IT-7a: Every aggregate table must contain at least 1 row.
    An empty aggregate means dlt.read("fact_listings") returned nothing,
    or the GROUP BY produced no groups -- both signal a broken pipeline dependency.
    """
    cnt = spark.read.table(f"{GOLD}.{agg_table}").count()
    assert cnt > 0, (
        f"[IT-7a | {agg_table}] Aggregate table is empty. "
        "Either fact_listings is empty or the DLT dependency chain is broken."
    )


def test_it7_monthly_agg_total_listings_reconciles_with_fact(spark):
    """
    IT-7b: SUM(total_listings) across agg_monthly_sales_trend must equal
    fact_listings row count exactly.
    Every fact row belongs to exactly one (month, brand, price_category) group.
    A higher sum = row explosion (cross-join bug); a lower sum = rows dropped.
    """
    fact_cnt = spark.read.table(f"{GOLD}.fact_listings").count()
    agg_sum  = (
        spark.read.table(f"{GOLD}.agg_monthly_sales_trend")
        .agg(F.sum("total_listings").alias("s"))
        .collect()[0]["s"] or 0
    )
    assert agg_sum == fact_cnt, (
        f"[IT-7b] agg_monthly_sales_trend SUM(total_listings)={agg_sum:,} "
        f"!= fact_listings COUNT={fact_cnt:,}. "
        "Aggregate is not a complete, non-overlapping partition of the fact table."
    )


def test_it7_top10_brands_has_at_most_10_rows(spark):
    """
    IT-7c: agg_top_10_brands_by_spend must contain <= 10 rows.
    The Gold pipeline applies .limit(10). More than 10 rows means the LIMIT
    was removed or the table accumulated rows from multiple pipeline runs.
    """
    cnt = spark.read.table(f"{GOLD}.agg_top_10_brands_by_spend").count()
    assert cnt <= 10, (
        f"[IT-7c] agg_top_10_brands_by_spend has {cnt} rows -- expected <= 10. "
        "The .limit(10) in the Gold pipeline was bypassed or the table "
        "accumulated rows from multiple pipeline runs."
    )


def test_it7_agg_brand_location_brands_subset_of_fact(spark):
    """
    IT-7d: Every brand in agg_brand_location_performance must exist in fact_listings.
    An unknown brand in the aggregate means the GROUP BY ran on stale or phantom data
    that is not present in the current fact table.
    """
    fact_brands = spark.read.table(f"{GOLD}.fact_listings").select("brand").distinct()
    agg_brands  = spark.read.table(f"{GOLD}.agg_brand_location_performance").select("brand").distinct()
    invented    = agg_brands.join(fact_brands, on="brand", how="left_anti").count()

    assert invented == 0, (
        f"[IT-7d] {invented:,} brand(s) in agg_brand_location_performance "
        "are not present in fact_listings. "
        "Aggregate contains brands that do not exist in the fact table."
    )

