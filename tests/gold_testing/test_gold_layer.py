# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer Test Suite
# MAGIC
# MAGIC | Suite | What it checks |
# MAGIC |---|---|
# MAGIC | T1 | Reconciliation — Gold row counts match Silver source of truth |
# MAGIC | T2 | Row Integrity — every Silver key present in Gold, no invented keys |
# MAGIC | T3 | Audit Columns — gold_load_dt, silver_load_dt, __START_AT/__END_AT |
# MAGIC | T4 | Fact ↔ Silver Reconciliation — JOIN fact + dims rebuilds Silver |
# MAGIC | T5 | Referential Integrity — no orphan FK keys in fact table |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & Spark Session

# COMMAND ----------

from pyspark.sql import functions as F

# pytest is only available in CI/CD via databricks-connect.
# In Databricks notebook, spark is already in scope — pytest is not needed.
try:
    import pytest

    @pytest.fixture(scope="session")
    def spark():
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.getOrCreate()

except ImportError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration & Registry

# COMMAND ----------

CATALOG = "vstone_catalog"
GOLD    = f"{CATALOG}.gold"
SILVER  = f"{CATALOG}.silver"

# sequence('2010-01-01', '2030-12-31', interval 1 day) = 7,670 rows
DIM_DATE_EXPECTED = 7670

SCD2_REGISTRY = [
    {"name": "dim_car",             "silver": f"{SILVER}.car_catalog_transformation",   "keys": ["brand", "model"]},
    {"name": "dim_location",        "silver": f"{SILVER}.geography_transformation",      "keys": ["city_prepositional"]},
    {"name": "dim_listing_details", "silver": f"{SILVER}.listings_text_transformation",  "keys": ["listing_id"]},
    {"name": "dim_listing_photos",  "silver": f"{SILVER}.listings_photo_transformation", "keys": ["listing_id", "photo_url_clean"]},
]

_results = []

def run_test(name, fn):
    try:
        fn()
        _results.append(("PASS", name))
        print(f"  \u2713  {name}")
    except AssertionError as e:
        _results.append(("FAIL", name, str(e)))
        print(f"  \u2717  {name}\n       \u2192 {e}")
    except Exception as e:
        _results.append(("ERROR", name, str(e)))
        print(f"  !  {name}  [ERROR]\n       \u2192 {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Reconciliation (Silver → Gold)
# MAGIC
# MAGIC Verifies Gold row counts match their Silver source of truth.
# MAGIC Each SCD2 dim is compared against its own Silver source table, not fact_listings.

# COMMAND ----------

def _t1_fact_matches_silver():
    silver_cnt = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    gold_cnt   = spark.read.table(f"{GOLD}.fact_listings").count()
    diff_pct   = abs(gold_cnt - silver_cnt) / max(silver_cnt, 1) * 100
    assert diff_pct <= 1.0, (
        f"fact_listings diverges from Silver by {diff_pct:.3f}% — "
        f"Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def _t1_dim_car_matches_silver():
    silver_cnt = spark.read.table(f"{SILVER}.car_catalog_transformation").select("brand", "model").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_car").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_car CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def _t1_dim_location_matches_silver():
    silver_cnt = spark.read.table(f"{SILVER}.geography_transformation").select("city_prepositional").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_location").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_location CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def _t1_dim_listing_details_matches_silver():
    silver_cnt = spark.read.table(f"{SILVER}.listings_text_transformation").select("listing_id").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_listing_details").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_listing_details CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def _t1_dim_listing_photos_matches_silver():
    silver_cnt = spark.read.table(f"{SILVER}.listings_photo_transformation").select("listing_id", "photo_url_clean").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_listing_photos").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_listing_photos CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def _t1_dim_date_row_count():
    cnt = spark.read.table(f"{GOLD}.dim_date").count()
    assert cnt == DIM_DATE_EXPECTED, (
        f"dim_date expected {DIM_DATE_EXPECTED:,} rows | got {cnt:,}"
    )

def _t1_agg_tables_non_empty():
    for tbl in [
        "agg_monthly_sales_trend", "agg_brand_location_performance",
        "agg_regional_market_depth", "agg_comprehensive_kpi_cube",
        "agg_top_10_brands_by_spend",
    ]:
        cnt = spark.read.table(f"{GOLD}.{tbl}").count()
        assert cnt > 0, f"[{tbl}] is empty."


print("=" * 60)
print("  T1 \u2014 Reconciliation")
print("=" * 60)
run_test("fact_listings matches silver",   
_t1_fact_matches_silver)
run_test("dim_car current matches car_catalog silver",          _t1_dim_car_matches_silver)
run_test("dim_location current matches geography silver",       _t1_dim_location_matches_silver)
run_test("dim_listing_details current matches text silver",     _t1_dim_listing_details_matches_silver)
run_test("dim_listing_photos current matches photo silver",     _t1_dim_listing_photos_matches_silver)
run_test(f"dim_date has {DIM_DATE_EXPECTED:,} rows",            _t1_dim_date_row_count)
run_test("all 5 agg tables non-empty",                          _t1_agg_tables_non_empty)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Row Integrity (Silver → Gold)
# MAGIC
# MAGIC Verifies every Silver listing_id flows into fact_listings with nothing dropped or invented.
# MAGIC Also verifies SCD2 dim natural keys match their Silver source in both directions.

# COMMAND ----------

def _t2_fact_no_missing_silver_ids():
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    missing    = silver_ids.subtract(gold_ids).count()
    assert missing == 0, f"fact_listings missing {missing:,} Silver listing_id(s)."

def _t2_fact_no_invented_ids():
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    invented   = gold_ids.subtract(silver_ids).count()
    assert invented == 0, f"fact_listings has {invented:,} listing_id(s) not in Silver."

def _t2_scd2_keys_both_directions():
    for entry in SCD2_REGISTRY:
        silver_keys = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
        gold_keys   = (spark.read.table(f"{GOLD}.{entry['name']}")
                       .filter("__END_AT IS NULL")
                       .select(*entry["keys"]))
        missing  = silver_keys.subtract(gold_keys).count()
        invented = gold_keys.subtract(silver_keys).count()
        assert missing == 0,  f"[{entry['name']}] {missing:,} Silver key(s) missing from Gold."
        assert invented == 0, f"[{entry['name']}] {invented:,} Gold key(s) not traceable to Silver."


print("=" * 60)
print("  T2 \u2014 Row Integrity")
print("=" * 60)
run_test("fact \u2014 no missing Silver listing_ids",            _t2_fact_no_missing_silver_ids)
run_test("fact \u2014 no invented listing_ids",                  _t2_fact_no_invented_ids)
run_test("all SCD2 dims \u2014 keys match Silver (both dirs)",   _t2_scd2_keys_both_directions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — Audit Columns
# MAGIC
# MAGIC Verifies the full Bronze → Silver → Gold audit chain on fact_listings.
# MAGIC Verifies all SCD2 dims carry the required SCD2 metadata columns.

# COMMAND ----------

def _t3_fact_audit_chain():
    df          = spark.read.table(f"{GOLD}.fact_listings")
    audit_chain = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]
    missing_col = [c for c in audit_chain if c not in df.columns]
    assert missing_col == [], f"fact_listings missing audit columns: {missing_col}"
    for col in audit_chain:
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, f"fact_listings.{col} has {nulls:,} NULL rows."

def _t3_scd2_metadata_columns():
    for entry in SCD2_REGISTRY:
        cols = spark.read.table(f"{GOLD}.{entry['name']}").columns
        for required in ("__START_AT", "__END_AT", "silver_load_dt"):
            assert required in cols, (
                f"[{entry['name']}] Missing SCD2 column: {required}"
            )


print("=" * 60)
print("  T3 \u2014 Audit Columns")
print("=" * 60)
run_test("fact_listings \u2014 full audit chain present and non-null",      _t3_fact_audit_chain)
run_test("all SCD2 dims \u2014 __START_AT/__END_AT/silver_load_dt present", _t3_scd2_metadata_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T4 — Fact ↔ Silver Reconciliation (Reverse Join)

# COMMAND ----------

def _t4_reconstruct_silver_listings():
    fact      = spark.read.table(f"{GOLD}.fact_listings")
    dim_price = spark.read.table(f"{GOLD}.dim_price_category").select("price_category_key", "price_category")
    dim_steer = spark.read.table(f"{GOLD}.dim_steering").select("steering_key", "steering_wheel")

    reconstructed = (
        fact
        .join(dim_price, on="price_category_key", how="left")
        .join(dim_steer, on="steering_key",        how="left")
        .select(
            "listing_id",
            "listing_date",
            "brand",
            "model",
            "price_rub",
            "price_usd",
            "price_category",
            "mileage_km",
            "manufacture_year",
            "engine_power",
            "steering_wheel",
            F.col("location_key").alias("city_prepositional"),
        )
    )

    silver = (
        spark.read.table(f"{SILVER}.listings_silver_merged")
        .select(
            "listing_id",
            F.col("listing_date").cast("date").alias("listing_date"),
            "brand",
            "model",
            "price_rub",
            "price_usd",
            "price_category",
            "mileage_km",
            "manufacture_year",
            "engine_power",
            "steering_wheel",
            "city_prepositional",
        )
    )

    missing_from_gold = silver.subtract(reconstructed).count()
    assert missing_from_gold == 0, (
        f"{missing_from_gold:,} Silver rows not recoverable from Gold star schema join."
    )
    extra_in_gold = reconstructed.subtract(silver).count()
    assert extra_in_gold == 0, (
        f"{extra_in_gold:,} Gold rows have no corresponding Silver row."
    )

def _t4_reconstruct_silver_text():
    fact             = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    dim_txt          = (spark.read.table(f"{GOLD}.dim_listing_details")
                        .filter(F.col("__END_AT").isNull())
                        .select("listing_id", "text"))
    reconstructed    = fact.join(dim_txt, on="listing_id", how="inner")
    silver_text      = (spark.read.table(f"{SILVER}.listings_text_transformation")
                        .select("listing_id", "text"))
    silver_with_text = silver_text.join(fact, on="listing_id", how="inner")
    missing = silver_with_text.subtract(reconstructed).count()
    assert missing == 0, (
        f"{missing:,} Silver text rows not recoverable from dim_listing_details."
    )


print("=" * 60)
print("  T4 \u2014 Fact \u21d4 Silver Reconciliation")
print("=" * 60)
run_test("JOIN fact + lookup dims reconstructs Silver listings",     _t4_reconstruct_silver_listings)
run_test("JOIN fact + dim_listing_details reconstructs Silver text", _t4_reconstruct_silver_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 — Referential Integrity (No Orphan FK Keys)

# COMMAND ----------

def _t5_no_orphan_dates():
    fact_dates = (spark.read.table(f"{GOLD}.fact_listings")
                  .select("listing_date").distinct()
                  .filter(F.col("listing_date").isNotNull()))
    dim_dates  = spark.read.table(f"{GOLD}.dim_date").select("date_key")
    orphans    = fact_dates.join(
        dim_dates,
        fact_dates["listing_date"] == dim_dates["date_key"],
        how="left_anti"
    ).count()
    assert orphans == 0, (
        f"{orphans:,} fact listing_date values not found in dim_date."
    )

def _t5_no_orphan_price_category_keys():
    fact_keys = (spark.read.table(f"{GOLD}.fact_listings")
                 .select("price_category_key").distinct()
                 .filter(F.col("price_category_key").isNotNull()))
    dim_keys  = spark.read.table(f"{GOLD}.dim_price_category").select("price_category_key")
    orphans   = fact_keys.subtract(dim_keys).count()
    assert orphans == 0, (
        f"{orphans:,} fact price_category_key values not in dim_price_category."
    )

def _t5_no_orphan_steering_keys():
    fact_keys = (spark.read.table(f"{GOLD}.fact_listings")
                 .select("steering_key").distinct()
                 .filter(F.col("steering_key").isNotNull()))
    dim_keys  = spark.read.table(f"{GOLD}.dim_steering").select("steering_key")
    orphans   = fact_keys.subtract(dim_keys).count()
    assert orphans == 0, (
        f"{orphans:,} fact steering_key values not in dim_steering."
    )


print("=" * 60)
print("  T5 \u2014 Referential Integrity")
print("=" * 60)
run_test("no orphan listing_date \u2192 dim_date",                 _t5_no_orphan_dates)
run_test("no orphan price_category_key \u2192 dim_price_category", _t5_no_orphan_price_category_keys)
run_test("no orphan steering_key \u2192 dim_steering",             _t5_no_orphan_steering_keys)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Summary

# COMMAND ----------

passed = sum(1 for r in _results if r[0] == "PASS")
failed = sum(1 for r in _results if r[0] == "FAIL")
errors = sum(1 for r in _results if r[0] == "ERROR")

print()
print("=" * 60)
print("  GOLD LAYER TEST RESULTS")
print("=" * 60)
print(f"  \u2713  PASSED : {passed}")
print(f"  \u2717  FAILED : {failed}")
print(f"  !  ERRORS : {errors}")
print(f"     TOTAL  : {len(_results)}")
print("=" * 60)

if failed > 0 or errors > 0:
    print()
    print("  FAILURES / ERRORS:")
    for r in _results:
        if r[0] in ("FAIL", "ERROR"):
            print(f"  [{r[0]}] {r[1]}")
            print(f"         {r[2]}")
    print("=" * 60)
    raise Exception(f"{failed + errors} test(s) failed — see output above.")
else:
    print()
    print("  ALL TESTS PASSED")
    print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## pytest Wrappers (CI/CD only)
# MAGIC
# MAGIC These functions are discovered by pytest in CI/CD via databricks-connect.
# MAGIC They are not executed when running this notebook directly in Databricks.

# COMMAND ----------

def test_t1_fact_listings_matches_silver(spark):               _t1_fact_matches_silver()
def test_t1_dim_car_current_matches_silver(spark):              _t1_dim_car_matches_silver()
def test_t1_dim_location_current_matches_silver(spark):         _t1_dim_location_matches_silver()
def test_t1_dim_listing_details_current_matches_silver(spark):  _t1_dim_listing_details_matches_silver()
def test_t1_dim_listing_photos_current_matches_silver(spark):   _t1_dim_listing_photos_matches_silver()
def test_t1_dim_date_row_count(spark):                          _t1_dim_date_row_count()
def test_t1_agg_tables_non_empty(spark):                        _t1_agg_tables_non_empty()
def test_t2_fact_no_missing_silver_ids(spark):                  _t2_fact_no_missing_silver_ids()
def test_t2_fact_no_invented_ids(spark):                        _t2_fact_no_invented_ids()
def test_t2_scd2_keys_both_directions(spark):                   _t2_scd2_keys_both_directions()

def test_t3_fact_audit_chain(spark):                            _t3_fact_audit_chain()
def test_t3_scd2_metadata_columns(spark):                       _t3_scd2_metadata_columns()

def test_t4_reconstruct_silver_listings(spark):                 _t4_reconstruct_silver_listings()
def test_t4_reconstruct_silver_text(spark):                     _t4_reconstruct_silver_text()

def test_t5_no_orphan_dates(spark):                             _t5_no_orphan_dates()
def test_t5_no_orphan_price_category_keys(spark):               _t5_no_orphan_price_category_keys()
def test_t5_no_orphan_steering_keys(spark):                     _t5_no_orphan_steering_keys()
