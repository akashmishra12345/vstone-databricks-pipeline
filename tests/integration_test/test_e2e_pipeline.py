# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Pipeline Integration Test Suite
# MAGIC Implements a PyTest framework using Databricks Connect to validate
# MAGIC the full Bronze → Silver → Gold pipeline across all layers.
# MAGIC
# MAGIC | Suite | What it checks                                                                     |
# MAGIC |-------|------------------------------------------------------------------------------------|
# MAGIC | T1    | All expected tables exist in each layer (Table Existence)                          |
# MAGIC | T2    | Row counts are non-zero and data flows Bronze → Silver → Gold (Volume Propagation) |
# MAGIC | T3    | Audit columns are populated end-to-end (bronze_load_dt → gold_load_dt)             |
# MAGIC | T4    | listing_id lineage is traceable from Bronze through Silver to Gold                 |
# MAGIC | T5    | Silver transformations produce expected clean/derived columns (Transform Integrity) |
# MAGIC | T6    | Gold Star Schema is well-joined; SCD2 dims have valid timelines (E2E Quality)      |
# MAGIC | T7    | No critical nulls in key columns across all layers (Null Sentinel)                |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark Fixture & Config

# COMMAND ----------

@pytest.fixture(scope="session")
def spark():
    """Initializes the Databricks Connect session for the entire test suite."""
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

MIN_JOIN_RATE_PCT = 60.0
DIM_DATE_EXPECTED = 7670

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registry
# MAGIC Single source of truth for all test suites.
# MAGIC Add a new entry here and all relevant suites pick it up automatically.

# COMMAND ----------

# ── Bronze tables ─────────────────────────────────────────────────────────────
BRONZE_TABLES = [
    "listings_raw",
]

# ── Silver tables ─────────────────────────────────────────────────────────────
SILVER_TABLES = [
    "listings_silver_merged",
    "car_catalog_transformation",
    "geography_transformation",
    "listings_text_transformation",
    "listings_photo_transformation",
]

# ── Gold tables (non-SCD2) ────────────────────────────────────────────────────
ALL_GOLD_TABLES = [
    "dim_date", "fact_listings", "agg_monthly_sales_trend", "agg_brand_location_performance",
    "agg_regional_market_depth", "agg_comprehensive_kpi_cube", "agg_top_10_brands_by_spend",
]

# ── SCD2 dimension tables ─────────────────────────────────────────────────────
SCD2_DIM_TABLES = ["dim_car", "dim_location", "dim_listing_details", "dim_listing_photos"]

# ── All gold (combined) for existence checks ──────────────────────────────────
ALL_GOLD_ALL = ALL_GOLD_TABLES + SCD2_DIM_TABLES

# ── SCD2 registry: natural keys and Silver source for each dim ────────────────
SCD2_REGISTRY = [
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

# ── Silver column expectations: columns expected in each Silver table ──────────
SILVER_SCHEMA_REGISTRY = [
    {
        "name"     : "listings_silver_merged",
        "required" : [
            "listing_id", "brand", "model", "price_rub", "price_usd",
            "mileage", "year", "location", "silver_load_dt",
            "bronze_load_dt", "bronze_source_file",
        ],
    },
    {
        "name"     : "car_catalog_transformation",
        "required" : ["brand", "model", "silver_load_dt"],
    },
    {
        "name"     : "geography_transformation",
        "required" : ["city_prepositional", "silver_load_dt"],
    },
    {
        "name"     : "listings_text_transformation",
        "required" : ["listing_id", "silver_load_dt"],
    },
    {
        "name"     : "listings_photo_transformation",
        "required" : ["listing_id", "photo_url_clean", "silver_load_dt"],
    },
]

# ── Star schema join registry ─────────────────────────────────────────────────
JOIN_REGISTRY = [
    {
        "dim"           : "dim_date",
        "condition"     : "f.listing_date = d.date_key",
        "active_filter" : None,
    },
    {
        "dim"           : "dim_car",
        "condition"     : "lower(trim(f.brand)) = lower(trim(d.brand)) AND lower(trim(f.model)) = lower(trim(d.model))",
        "active_filter" : "__END_AT IS NULL",
    },
    {
        "dim"           : "dim_location",
        "condition"     : "f.location_key = d.city_prepositional",
        "active_filter" : "__END_AT IS NULL",
    },
    {
        "dim"           : "dim_listing_details",
        "condition"     : "f.listing_id = d.listing_id",
        "active_filter" : "__END_AT IS NULL",
    },
    {
        "dim"           : "dim_listing_photos",
        "condition"     : "f.listing_id = d.listing_id",
        "active_filter" : "__END_AT IS NULL",
    },
]

# ── Null sentinel registry: critical columns that must never be null ───────────
NULL_SENTINEL_REGISTRY = [
    {"layer": BRONZE, "table": "listings_raw",              "col": "listing_id"},
    {"layer": BRONZE, "table": "listings_raw",              "col": "bronze_load_dt"},
    {"layer": SILVER, "table": "listings_silver_merged",    "col": "listing_id"},
    {"layer": SILVER, "table": "listings_silver_merged",    "col": "silver_load_dt"},
    {"layer": SILVER, "table": "listings_silver_merged",    "col": "price_rub"},
    {"layer": GOLD,   "table": "fact_listings",             "col": "listing_id"},
    {"layer": GOLD,   "table": "fact_listings",             "col": "gold_load_dt"},
    {"layer": GOLD,   "table": "fact_listings",             "col": "price_rub"},
]

# ── Parametrize helpers ───────────────────────────────────────────────────────
BRONZE_TABLE_PARAMS    = [pytest.param(t, id=t) for t in BRONZE_TABLES]
SILVER_TABLE_PARAMS    = [pytest.param(t, id=t) for t in SILVER_TABLES]
GOLD_TABLE_PARAMS      = [pytest.param(t, id=t) for t in ALL_GOLD_ALL]
SCD2_DIM_PARAMS        = [pytest.param(t, id=t) for t in SCD2_DIM_TABLES]
SCD2_PARAMS            = [pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY]
JOIN_PARAMS            = [pytest.param(e, id=e["dim"])  for e in JOIN_REGISTRY]
SILVER_SCHEMA_PARAMS   = [pytest.param(e, id=e["name"]) for e in SILVER_SCHEMA_REGISTRY]
NULL_SENTINEL_PARAMS   = [
    pytest.param(e, id=f"{e['table']}.{e['col']}") for e in NULL_SENTINEL_REGISTRY
]

print(
    f"Registry loaded — "
    f"{len(BRONZE_TABLES)} Bronze | "
    f"{len(SILVER_TABLES)} Silver | "
    f"{len(ALL_GOLD_ALL)} Gold tables | "
    f"{len(SCD2_REGISTRY)} SCD2 dims | "
    f"{len(JOIN_REGISTRY)} joins | "
    f"{len(NULL_SENTINEL_REGISTRY)} null sentinels registered."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Table Existence
# MAGIC Verifies every expected table exists in each layer (Bronze, Silver, Gold).
# MAGIC A missing table means the pipeline did not complete successfully.

# COMMAND ----------

@pytest.mark.parametrize("table", BRONZE_TABLE_PARAMS)
def test_t1_bronze_table_exists(spark, table):
    """T1 — Every Bronze table must be accessible (pipeline ingestion ran)."""
    try:
        spark.read.table(f"{BRONZE}.{table}").limit(1).collect()
    except Exception as exc:
        pytest.fail(f"[{BRONZE}.{table}] Table not accessible: {exc}")


@pytest.mark.parametrize("table", SILVER_TABLE_PARAMS)
def test_t1_silver_table_exists(spark, table):
    """T1 — Every Silver table must be accessible (Silver transformation ran)."""
    try:
        spark.read.table(f"{SILVER}.{table}").limit(1).collect()
    except Exception as exc:
        pytest.fail(f"[{SILVER}.{table}] Table not accessible: {exc}")


@pytest.mark.parametrize("table", GOLD_TABLE_PARAMS)
def test_t1_gold_table_exists(spark, table):
    """T1 — Every Gold table must be accessible (Gold transformation ran)."""
    try:
        spark.read.table(f"{GOLD}.{table}").limit(1).collect()
    except Exception as exc:
        pytest.fail(f"[{GOLD}.{table}] Table not accessible: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Volume Propagation (Bronze → Silver → Gold)
# MAGIC Verifies that data flows through each layer with non-zero row counts.
# MAGIC Gold's fact_listings must be within 1% of Silver's listings_silver_merged.

# COMMAND ----------

@pytest.mark.parametrize("table", BRONZE_TABLE_PARAMS)
def test_t2_bronze_table_non_empty(spark, table):
    """T2 — Bronze tables must have at least 1 row (ingestion produced data)."""
    cnt = spark.read.table(f"{BRONZE}.{table}").count()
    assert cnt > 0, f"[{BRONZE}.{table}] Bronze table is empty — ingestion may have failed."


@pytest.mark.parametrize("table", SILVER_TABLE_PARAMS)
def test_t2_silver_table_non_empty(spark, table):
    """T2 — Silver tables must have at least 1 row (Silver transformation produced data)."""
    cnt = spark.read.table(f"{SILVER}.{table}").count()
    assert cnt > 0, f"[{SILVER}.{table}] Silver table is empty — transformation may have failed."


@pytest.mark.parametrize("table", GOLD_TABLE_PARAMS)
def test_t2_gold_table_non_empty(spark, table):
    """T2 — Gold tables must have at least 1 row (Gold transformation produced data)."""
    cnt = spark.read.table(f"{GOLD}.{table}").count()
    assert cnt > 0, f"[{GOLD}.{table}] Gold table is empty — Gold transformation may have failed."


def test_t2_bronze_to_silver_volume_propagation(spark):
    """T2 — listings_silver_merged must be within 5% of Bronze listings_raw (no mass drop)."""
    bronze_cnt = spark.read.table(f"{BRONZE}.listings_raw").count()
    silver_cnt = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    diff_pct   = abs(silver_cnt - bronze_cnt) / max(bronze_cnt, 1) * 100
    assert diff_pct <= 5.0, (
        f"Silver listings_silver_merged dropped >5% of Bronze rows — "
        f"Bronze={bronze_cnt:,} | Silver={silver_cnt:,} | Diff={diff_pct:.2f}%"
    )


def test_t2_silver_to_gold_volume_propagation(spark):
    """T2 — fact_listings row count must be within 1% of listings_silver_merged (Silver→Gold)."""
    silver_cnt = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    gold_cnt   = spark.read.table(f"{GOLD}.fact_listings").count()
    diff_pct   = abs(gold_cnt - silver_cnt) / max(silver_cnt, 1) * 100
    assert diff_pct <= 1.0, (
        f"fact_listings row count diverges from Silver by {diff_pct:.3f}% — "
        f"Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )


def test_t2_dim_date_row_count(spark):
    """T2 — dim_date must contain exactly 7,670 rows (2010–2030 calendar spine)."""
    date_cnt = spark.read.table(f"{GOLD}.dim_date").count()
    assert date_cnt == DIM_DATE_EXPECTED, (
        f"dim_date row count mismatch — Expected={DIM_DATE_EXPECTED:,} | Actual={date_cnt:,}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — End-to-End Audit Column Propagation
# MAGIC Verifies the audit chain (bronze_load_dt → silver_load_dt → gold_load_dt)
# MAGIC is present and non-null in fact_listings.

# COMMAND ----------

def test_t3_bronze_audit_columns_present_and_non_null(spark):
    """T3 — Bronze listings_raw must carry bronze_load_dt and bronze_source_file (non-null)."""
    df         = spark.read.table(f"{BRONZE}.listings_raw")
    audit_cols = ["bronze_load_dt", "bronze_source_file"]
    missing    = [c for c in audit_cols if c not in df.columns]
    assert missing == [], f"listings_raw missing Bronze audit columns: {missing}"
    for col in audit_cols:
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, f"listings_raw.{col} has {nulls:,} NULL rows."


def test_t3_silver_audit_columns_present_and_non_null(spark):
    """T3 — listings_silver_merged must carry bronze + silver audit columns (non-null)."""
    df         = spark.read.table(f"{SILVER}.listings_silver_merged")
    audit_cols = ["bronze_load_dt", "bronze_source_file", "silver_load_dt"]
    missing    = [c for c in audit_cols if c not in df.columns]
    assert missing == [], f"listings_silver_merged missing audit columns: {missing}"
    for col in audit_cols:
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, f"listings_silver_merged.{col} has {nulls:,} NULL rows."


def test_t3_fact_listings_full_audit_chain(spark):
    """T3 — fact_listings must carry the complete Bronze→Silver→Gold audit chain (non-null)."""
    df          = spark.read.table(f"{GOLD}.fact_listings")
    audit_chain = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]
    missing     = [c for c in audit_chain if c not in df.columns]
    assert missing == [], f"fact_listings missing audit chain columns: {missing}"
    for col in audit_chain:
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, f"fact_listings.{col} has {nulls:,} NULL rows."


@pytest.mark.parametrize("table", GOLD_TABLE_PARAMS)
def test_t3_gold_load_dt_present_and_non_null(spark, table):
    """T3 — gold_load_dt must exist and be fully non-null in every Gold table."""
    df = spark.read.table(f"{GOLD}.{table}")
    assert "gold_load_dt" in df.columns, f"[{table}] Missing column: gold_load_dt"
    nulls = df.filter(F.col("gold_load_dt").isNull()).count()
    assert nulls == 0, f"[{table}] gold_load_dt has {nulls:,} NULL rows."


@pytest.mark.parametrize("table", SCD2_DIM_PARAMS)
def test_t3_scd2_silver_load_dt_non_null(spark, table):
    """T3 — SCD2 dims must carry silver_load_dt (lineage timestamp) and it must be non-null."""
    df = spark.read.table(f"{GOLD}.{table}")
    assert "silver_load_dt" in df.columns, f"[{table}] Missing column: silver_load_dt"
    nulls = df.filter(F.col("silver_load_dt").isNull()).count()
    assert nulls == 0, f"[{table}] silver_load_dt has {nulls:,} NULL rows."

# COMMAND ----------

# MAGIC %md
# MAGIC ## T4 — listing_id Lineage (Bronze → Silver → Gold)
# MAGIC Verifies listing_ids flow correctly through every pipeline layer
# MAGIC with no drops or invented IDs at any stage.

# COMMAND ----------

def test_t4_bronze_to_silver_no_missing_ids(spark):
    """T4 — Every Bronze listing_id must appear in listings_silver_merged."""
    bronze_ids = spark.read.table(f"{BRONZE}.listings_raw").select("listing_id")
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    missing    = bronze_ids.subtract(silver_ids).count()
    assert missing == 0, (
        f"listings_silver_merged is missing {missing:,} Bronze listing_id(s) — "
        f"data was dropped in the Bronze→Silver step."
    )


def test_t4_bronze_to_silver_no_invented_ids(spark):
    """T4 — listings_silver_merged must not contain listing_ids absent from Bronze."""
    bronze_ids = spark.read.table(f"{BRONZE}.listings_raw").select("listing_id")
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    invented   = silver_ids.subtract(bronze_ids).count()
    assert invented == 0, (
        f"listings_silver_merged contains {invented:,} invented listing_id(s) not present in Bronze."
    )


def test_t4_silver_to_gold_no_missing_ids(spark):
    """T4 — Every Silver listing_id must appear in fact_listings (nothing dropped)."""
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    missing    = silver_ids.subtract(gold_ids).count()
    assert missing == 0, (
        f"fact_listings is missing {missing:,} Silver listing_id(s) — "
        f"data was dropped in the Silver→Gold step."
    )


def test_t4_silver_to_gold_no_invented_ids(spark):
    """T4 — fact_listings must not contain listing_ids absent from Silver."""
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    invented   = gold_ids.subtract(silver_ids).count()
    assert invented == 0, (
        f"fact_listings contains {invented:,} invented listing_id(s) not traceable to Silver."
    )


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t4_scd2_silver_keys_present_in_gold(spark, entry):
    """T4 — Every distinct Silver natural key must appear as a CURRENT Gold SCD2 row."""
    silver_keys      = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_active_keys = (
        spark.read.table(f"{GOLD}.{entry['name']}")
        .filter("__END_AT IS NULL")
        .select(*entry["keys"])
    )
    missing = silver_keys.subtract(gold_active_keys).count()
    assert missing == 0, (
        f"[{entry['name']}] {missing:,} Silver key(s) missing from Gold active rows."
    )


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t4_scd2_no_invented_gold_keys(spark, entry):
    """T4 — Gold SCD2 active rows must not contain keys absent from Silver."""
    silver_keys      = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_active_keys = (
        spark.read.table(f"{GOLD}.{entry['name']}")
        .filter("__END_AT IS NULL")
        .select(*entry["keys"])
    )
    invented = gold_active_keys.subtract(silver_keys).count()
    assert invented == 0, (
        f"[{entry['name']}] {invented:,} invented key(s) in Gold not traceable to Silver."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 — Silver Transform Integrity
# MAGIC Verifies Silver tables have the expected columns produced by each transformation,
# MAGIC and that key derived fields are non-null.

# COMMAND ----------

@pytest.mark.parametrize("entry", SILVER_SCHEMA_PARAMS)
def test_t5_silver_required_columns_present(spark, entry):
    """T5 — Each Silver table must contain all required columns from its transformation."""
    cols    = spark.read.table(f"{SILVER}.{entry['name']}").columns
    missing = [c for c in entry["required"] if c not in cols]
    assert missing == [], (
        f"[{entry['name']}] Missing required Silver columns: {missing}"
    )


def test_t5_silver_price_usd_non_null(spark):
    """T5 — listings_silver_merged.price_usd must be non-null (currency conversion ran)."""
    df    = spark.read.table(f"{SILVER}.listings_silver_merged")
    nulls = df.filter(F.col("price_usd").isNull()).count()
    assert nulls == 0, (
        f"listings_silver_merged.price_usd has {nulls:,} NULL rows — "
        f"currency conversion may have failed."
    )


def test_t5_silver_listing_ids_unique(spark):
    """T5 — listings_silver_merged must not have duplicate listing_ids after dedup."""
    total    = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    distinct = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id").distinct().count()
    assert total == distinct, (
        f"listings_silver_merged has {total - distinct:,} duplicate listing_id(s) — "
        f"deduplication step may have been skipped."
    )


def test_t5_silver_year_in_valid_range(spark):
    """T5 — listings_silver_merged.year must be between 1900 and current year (no corrupt data)."""
    from datetime import datetime
    current_year = datetime.now().year
    df = spark.read.table(f"{SILVER}.listings_silver_merged")
    if "year" not in df.columns:
        pytest.skip("Column 'year' not present in listings_silver_merged — skipping range check.")
    bad_rows = df.filter((F.col("year") < 1900) | (F.col("year") > current_year)).count()
    assert bad_rows == 0, (
        f"listings_silver_merged has {bad_rows:,} rows with 'year' outside valid range "
        f"[1900, {current_year}]."
    )


def test_t5_silver_car_catalog_has_distinct_brand_model(spark):
    """T5 — car_catalog_transformation must have unique brand+model combinations."""
    total    = spark.read.table(f"{SILVER}.car_catalog_transformation").count()
    distinct = (
        spark.read.table(f"{SILVER}.car_catalog_transformation")
        .select("brand", "model").distinct().count()
    )
    assert total == distinct, (
        f"car_catalog_transformation has {total - distinct:,} duplicate brand+model rows."
    )


def test_t5_silver_geography_city_non_null(spark):
    """T5 — geography_transformation.city_prepositional must be non-null (geo transform ran)."""
    df    = spark.read.table(f"{SILVER}.geography_transformation")
    nulls = df.filter(F.col("city_prepositional").isNull()).count()
    assert nulls == 0, (
        f"geography_transformation.city_prepositional has {nulls:,} NULL rows."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T6 — Gold E2E Quality (Star Schema & SCD2)
# MAGIC Verifies FK join rates ≥ 60% in the star schema and SCD2 timeline consistency.

# COMMAND ----------

def test_t6_fact_listings_all_columns_present(spark):
    """T6 — fact_listings must contain all FK, derived, financial, and audit columns."""
    fact_cols      = spark.read.table(f"{GOLD}.fact_listings").columns
    fk_cols        = ["listing_id", "listing_date", "brand", "model", "location_key"]
    derived_cols   = ["car_age_at_listing", "is_high_mileage", "price_per_hp_usd", "photo_count"]
    financial_cols = ["price_rub", "price_usd", "price_category", "car_age_years", "listing_year", "listing_month"]
    audit_cols     = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]
    all_required   = fk_cols + derived_cols + financial_cols + audit_cols
    missing        = [c for c in all_required if c not in fact_cols]
    assert missing == [], (
        f"fact_listings missing columns: {missing} | Total columns present: {len(fact_cols)}"
    )


def test_t6_dim_date_all_columns_present(spark):
    """T6 — dim_date must contain all calendar and audit columns."""
    date_cols     = spark.read.table(f"{GOLD}.dim_date").columns
    date_required = [
        "date_key", "year", "quarter", "month", "month_name",
        "week_of_year", "day", "day_name", "is_weekend", "gold_load_dt",
    ]
    missing = [c for c in date_required if c not in date_cols]
    assert missing == [], f"dim_date missing columns: {missing}"


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t6_scd2_metadata_columns_present(spark, entry):
    """T6 — SCD2 dims must carry __START_AT, __END_AT, and silver_load_dt."""
    cols = spark.read.table(f"{GOLD}.{entry['name']}").columns
    for required in ("__START_AT", "__END_AT", "silver_load_dt"):
        assert required in cols, (
            f"[{entry['name']}] Missing SCD2 metadata column: {required}"
        )


@pytest.mark.parametrize("entry", JOIN_PARAMS)
def test_t6_fact_dim_join_rate(spark, entry):
    """T6 — Join rate between fact_listings and each dimension must be ≥ 60%."""
    fact_cnt     = spark.read.table(f"{GOLD}.fact_listings").count()
    where_clause = f"WHERE d.{entry['active_filter']}" if entry["active_filter"] else ""
    sql          = (
        f"SELECT COUNT(*) AS c "
        f"FROM {GOLD}.fact_listings f "
        f"JOIN {GOLD}.{entry['dim']} d ON {entry['condition']} {where_clause}"
    )
    joined = spark.sql(sql).collect()[0]["c"]
    rate   = round(joined / max(fact_cnt, 1) * 100, 2)
    assert rate >= MIN_JOIN_RATE_PCT, (
        f"[fact → {entry['dim']}] Join rate {rate}% is below minimum {MIN_JOIN_RATE_PCT}% — "
        f"Matched={joined:,} / Total={fact_cnt:,}"
    )


@pytest.mark.parametrize("entry", [pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY if e["name"] != "dim_listing_photos"])
def test_t6_scd2_one_active_row_per_key(spark, entry):
    """T6 — Each natural key must have exactly one active row (__END_AT IS NULL)."""
    key_str = ", ".join(entry["keys"])
    dup_sql = (
        f"SELECT COUNT(*) AS c FROM ("
        f"SELECT {key_str}, COUNT(*) AS cnt FROM {GOLD}.{entry['name']} "
        f"WHERE __END_AT IS NULL GROUP BY {key_str} HAVING cnt > 1)"
    )
    dups = spark.sql(dup_sql).collect()[0]["c"]
    assert dups == 0, (
        f"[{entry['name']}] {dups:,} natural key(s) have more than one active row."
    )


@pytest.mark.parametrize("entry", [pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY if e["name"] != "dim_listing_photos"])
def test_t6_scd2_timeline_consistency(spark, entry):
    """T6 — For historical rows, __START_AT must always be strictly before __END_AT."""
    timeline_sql = (
        f"SELECT COUNT(*) AS c FROM {GOLD}.{entry['name']} "
        f"WHERE __END_AT IS NOT NULL AND __START_AT >= __END_AT"
    )
    violations = spark.sql(timeline_sql).collect()[0]["c"]
    assert violations == 0, (
        f"[{entry['name']}] {violations:,} historical rows have __START_AT >= __END_AT."
    )


@pytest.mark.parametrize("entry", [pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY if e["name"] != "dim_listing_photos"])
def test_t6_scd2_every_key_has_active_row(spark, entry):
    """T6 — Every natural key must have at least one active row (__END_AT IS NULL)."""
    key_str        = ", ".join(entry["keys"])
    missing_active = (
        f"SELECT COUNT(*) AS c FROM ("
        f"SELECT {key_str} FROM {GOLD}.{entry['name']} "
        f"GROUP BY {key_str} HAVING SUM(CAST(__END_AT IS NULL AS INT)) = 0)"
    )
    orphans = spark.sql(missing_active).collect()[0]["c"]
    assert orphans == 0, (
        f"[{entry['name']}] {orphans:,} natural key(s) have no active row at all."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T7 — Null Sentinel (Critical Columns)
# MAGIC Sweeps critical columns across all three layers for NULL values.
# MAGIC Any NULL in these columns indicates a pipeline failure upstream.

# COMMAND ----------

@pytest.mark.parametrize("entry", NULL_SENTINEL_PARAMS)
def test_t7_critical_column_non_null(spark, entry):
    """T7 — Critical columns must never be NULL in any pipeline layer."""
    df    = spark.read.table(f"{entry['layer']}.{entry['table']}")
    col   = entry["col"]
    assert col in df.columns, (
        f"[{entry['table']}] Column '{col}' does not exist — cannot run null check."
    )
    nulls = df.filter(F.col(col).isNull()).count()
    assert nulls == 0, (
        f"[{entry['layer']}.{entry['table']}.{col}] has {nulls:,} NULL rows — "
        f"pipeline may have failed upstream."
    )
