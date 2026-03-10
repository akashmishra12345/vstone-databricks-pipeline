# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer Test Suite
# MAGIC Implements a PyTest framework to validate
# MAGIC the Silver → Gold transition across all fact, dimension, and aggregate tables.
# MAGIC
# MAGIC | Suite | What it checks                                                      |
# MAGIC |-------|---------------------------------------------------------------------|
# MAGIC | T1    | Gold row counts vs Silver source of truth (Reconciliation)          |
# MAGIC | T2    | Every Silver listing_id present in fact_listings (Row Integrity)    |
# MAGIC | T3    | `gold_load_dt` on all tables; `__START_AT/__END_AT` on SCD2 dims   |
# MAGIC | T4    | All FK + derived + financial columns present (Schema Integrity)     |
# MAGIC | T5    | FK join rates ≥ 60%, orphan check (Star Schema Joins)              |
# MAGIC | T6    | One CURRENT row per key, END_AT consistency (SCD2 Integrity)       |

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
    "gold":    "gold",
    "silver":  "silver",
}

GOLD   = f"{CONFIG['catalog']}.{CONFIG['gold']}"
SILVER = f"{CONFIG['catalog']}.{CONFIG['silver']}"

MIN_JOIN_RATE_PCT = 60.0
DIM_DATE_EXPECTED = 7670

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registry
# MAGIC Single source of truth for all test suites.
# MAGIC Add a new entry here and all relevant suites pick it up automatically.

# COMMAND ----------

# Gold tables that carry gold_load_dt (excludes SCD2 dims which use load_dt)
ALL_GOLD_TABLES = [
    "dim_date", "fact_listings", "agg_monthly_sales_trend", "agg_brand_location_performance",
    "agg_regional_market_depth", "agg_comprehensive_kpi_cube", "agg_top_10_brands_by_spend",
]

# SCD2 dims use load_dt instead of gold_load_dt
SCD2_DIM_TABLES = ["dim_car", "dim_location", "dim_listing_details", "dim_listing_photos"]

# SCD2 dimension tables and their natural keys
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

# Star schema join configs: (dim_table, join_condition, active_filter_or_None)
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

# Parametrize lists
SCD2_PARAMS      = [pytest.param(e, id=e["name"])  for e in SCD2_REGISTRY]
JOIN_PARAMS      = [pytest.param(e, id=e["dim"])   for e in JOIN_REGISTRY]
GOLD_TABLE_PARAMS  = [pytest.param(t, id=t) for t in ALL_GOLD_TABLES]
SCD2_DIM_PARAMS   = [pytest.param(t, id=t) for t in SCD2_DIM_TABLES]

print(f"Registry loaded — {len(ALL_GOLD_TABLES)} Gold tables | "
      f"{len(SCD2_REGISTRY)} SCD2 dims | {len(JOIN_REGISTRY)} joins registered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Reconciliation (Silver → Gold)
# MAGIC Verifies Gold row counts match Silver source of truth.
# MAGIC Fact table is 1:1 with Silver; SCD2 dims compare CURRENT rows (\_\_END_AT IS NULL).

# COMMAND ----------

def test_t1_fact_listings_matches_silver(spark):
    """T1 — fact_listings row count must be within 1% of listings_silver_merged."""
    silver_cnt = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    gold_cnt   = spark.read.table(f"{GOLD}.fact_listings").count()
    diff_pct   = abs(gold_cnt - silver_cnt) / max(silver_cnt, 1) * 100
    assert diff_pct <= 1.0, (
        f"fact_listings row count diverges from Silver by {diff_pct:.3f}% — "
        f"Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )


def test_t1_dim_car_current_matches_silver(spark):
    """T1 — dim_car CURRENT rows (__END_AT IS NULL) must equal DISTINCT brand+model in Silver."""
    silver_cnt = spark.read.table(f"{SILVER}.car_catalog_transformation").select("brand", "model").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_car").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_car CURRENT mismatch — Silver DISTINCT={silver_cnt:,} | dim_car CURRENT={gold_cnt:,}"
    )


def test_t1_dim_location_current_matches_silver(spark):
    """T1 — dim_location CURRENT rows must equal DISTINCT city_prepositional in Silver."""
    silver_cnt = spark.read.table(f"{SILVER}.geography_transformation").select("city_prepositional").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_location").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_location CURRENT mismatch — Silver DISTINCT={silver_cnt:,} | dim_location CURRENT={gold_cnt:,}"
    )


def test_t1_dim_listing_details_current_matches_silver(spark):
    """T1 — dim_listing_details CURRENT rows must equal DISTINCT listing_id in Silver."""
    silver_cnt = spark.read.table(f"{SILVER}.listings_text_transformation").select("listing_id").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_listing_details").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_listing_details CURRENT mismatch — Silver DISTINCT={silver_cnt:,} | Gold CURRENT={gold_cnt:,}"
    )


def test_t1_dim_listing_photos_current_matches_silver(spark):
    """T1 — dim_listing_photos CURRENT rows must equal DISTINCT listing_id+photo_url_clean in Silver."""
    silver_cnt = spark.read.table(f"{SILVER}.listings_photo_transformation").select("listing_id", "photo_url_clean").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_listing_photos").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_listing_photos CURRENT mismatch — Silver DISTINCT={silver_cnt:,} | Gold CURRENT={gold_cnt:,}"
    )


def test_t1_dim_date_row_count(spark):
    """T1 — dim_date must contain exactly 7,670 rows (2010–2030)."""
    date_cnt = spark.read.table(f"{GOLD}.dim_date").count()
    assert date_cnt == DIM_DATE_EXPECTED, (
        f"dim_date row count mismatch — Expected={DIM_DATE_EXPECTED:,} | Actual={date_cnt:,}"
    )


@pytest.mark.parametrize("agg_table", [
    pytest.param(t, id=t) for t in [
        "agg_monthly_sales_trend", "agg_brand_location_performance",
        "agg_regional_market_depth", "agg_comprehensive_kpi_cube", "agg_top_10_brands_by_spend",
    ]
])
def test_t1_agg_tables_non_empty(spark, agg_table):
    """T1 — All aggregate tables must contain at least 1 row."""
    cnt = spark.read.table(f"{GOLD}.{agg_table}").count()
    assert cnt > 0, f"[{agg_table}] Aggregate table is empty."

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Row-to-Row Integrity
# MAGIC Verifies every Silver listing_id is present in fact_listings
# MAGIC and every SCD2 dimension CURRENT row is traceable to Silver.

# COMMAND ----------

def test_t2_fact_listings_no_missing_silver_ids(spark):
    """T2 — Every Silver listing_id must exist in fact_listings (nothing dropped)."""
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    missing    = silver_ids.subtract(gold_ids).count()
    assert missing == 0, f"fact_listings is missing {missing:,} Silver listing_id(s)."


def test_t2_fact_listings_no_invented_ids(spark):
    """T2 — fact_listings must not contain listing_ids absent from Silver."""
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    invented   = gold_ids.subtract(silver_ids).count()
    assert invented == 0, f"fact_listings contains {invented:,} invented listing_id(s) not in Silver."


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t2_scd2_no_missing_silver_keys(spark, entry):
    """T2 — Every distinct Silver key must appear as an active Gold row (__END_AT IS NULL)."""
    silver_keys     = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_active_keys = spark.read.table(f"{GOLD}.{entry['name']}").filter("__END_AT IS NULL").select(*entry["keys"])
    missing = silver_keys.subtract(gold_active_keys).count()
    assert missing == 0, (
        f"[{entry['name']}] {missing:,} Silver key(s) missing from Gold active rows."
    )


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t2_scd2_no_invented_gold_keys(spark, entry):
    """T2 — Gold active rows must not contain keys absent from Silver."""
    silver_keys      = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_active_keys = spark.read.table(f"{GOLD}.{entry['name']}").filter("__END_AT IS NULL").select(*entry["keys"])
    invented = gold_active_keys.subtract(silver_keys).count()
    assert invented == 0, (
        f"[{entry['name']}] {invented:,} invented key(s) in Gold active rows not traceable to Silver."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — Audit Columns
# MAGIC Verifies `gold_load_dt` exists and is non-null on all Gold tables.
# MAGIC Verifies SCD2 metadata columns and full Bronze→Silver→Gold audit chain.

# COMMAND ----------

@pytest.mark.parametrize("table", GOLD_TABLE_PARAMS)
def test_t3_gold_load_dt_present_and_non_null(spark, table):
    """T3 — gold_load_dt must exist and have zero null values in every Gold table."""
    df = spark.read.table(f"{GOLD}.{table}")
    assert "gold_load_dt" in df.columns, f"[{table}] Missing column: gold_load_dt"
    nulls = df.filter(F.col("gold_load_dt").isNull()).count()
    assert nulls == 0, f"[{table}] gold_load_dt has {nulls:,} NULL rows."


@pytest.mark.parametrize("table", SCD2_DIM_PARAMS)
def test_t3_scd2_load_dt_present_and_non_null(spark, table):
    """T3 — SCD2 dims carry silver_load_dt as their lineage timestamp — must exist and be non-null."""
    df = spark.read.table(f"{GOLD}.{table}")
    assert "silver_load_dt" in df.columns, f"[{table}] Missing column: silver_load_dt"
    nulls = df.filter(F.col("silver_load_dt").isNull()).count()
    assert nulls == 0, f"[{table}] silver_load_dt has {nulls:,} NULL rows."


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t3_scd2_metadata_columns_present(spark, entry):
    """T3 — SCD2 dims must have __START_AT, __END_AT and silver_load_dt columns."""
    cols = spark.read.table(f"{GOLD}.{entry['name']}").columns
    for required in ("__START_AT", "__END_AT", "silver_load_dt"):
        assert required in cols, (
            f"[{entry['name']}] Missing SCD2 metadata column: {required}"
        )


def test_t3_fact_listings_full_audit_chain(spark):
    """T3 — fact_listings must carry the full bronze→silver→gold audit chain."""
    fact_cols     = spark.read.table(f"{GOLD}.fact_listings").columns
    audit_chain   = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]
    missing_audit = [c for c in audit_chain if c not in fact_cols]
    assert missing_audit == [], (
        f"fact_listings missing audit chain columns: {missing_audit}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T4 — Schema Integrity
# MAGIC Verifies all FK, derived, financial, and audit columns are present
# MAGIC in fact_listings and dim_date.

# COMMAND ----------

def test_t4_fact_listings_all_columns_present(spark):
    """T4 — fact_listings must contain all FK, derived, financial, and audit columns."""
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


def test_t4_dim_date_all_columns_present(spark):
    """T4 — dim_date must contain all calendar and audit columns."""
    date_cols     = spark.read.table(f"{GOLD}.dim_date").columns
    date_required = [
        "date_key", "year", "quarter", "month", "month_name",
        "week_of_year", "day", "day_name", "is_weekend", "gold_load_dt",
    ]
    missing = [c for c in date_required if c not in date_cols]
    assert missing == [], f"dim_date missing columns: {missing}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 — Star Schema Join Quality
# MAGIC Verifies join rates between fact_listings and each dimension are ≥ 60%.
# MAGIC SCD2 dimensions are joined on CURRENT rows only (__END_AT IS NULL).

# COMMAND ----------

@pytest.mark.parametrize("entry", JOIN_PARAMS)
def test_t5_fact_dim_join_rate(spark, entry):
    """T5 — Join rate between fact_listings and each dimension must be ≥ 60%."""
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## T6 — SCD2 Integrity
# MAGIC Verifies exactly one CURRENT row per natural key, timeline consistency,
# MAGIC and that every key has at least one active row.

# COMMAND ----------

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
    """T6 — For historical rows, __START_AT must always be before __END_AT."""
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
