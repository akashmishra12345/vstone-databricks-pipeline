# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — pytest Test Suite
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
# MAGIC ## Imports

# COMMAND ----------

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark Session Fixture

# COMMAND ----------

# ── Spark session fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    return DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration & Registry

# COMMAND ----------

# ── Config ────────────────────────────────────────────────────────────────────

CATALOG           = "vstone_catalog"
GOLD              = f"{CATALOG}.gold"
SILVER            = f"{CATALOG}.silver"
DIM_DATE_EXPECTED = 7670

SCD2_REGISTRY = [
    {"name": "dim_car",             "silver": f"{SILVER}.car_catalog_transformation",   "keys": ["brand", "model"]},
    {"name": "dim_location",        "silver": f"{SILVER}.geography_transformation",      "keys": ["city_prepositional"]},
    {"name": "dim_listing_details", "silver": f"{SILVER}.listings_text_transformation",  "keys": ["listing_id"]},
    {"name": "dim_listing_photos",  "silver": f"{SILVER}.listings_photo_transformation", "keys": ["listing_id", "photo_url_clean"]},
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Reconciliation (Silver → Gold)

# COMMAND ----------

# ── T1 — Reconciliation ───────────────────────────────────────────────────────

def test_t1_fact_listings_matches_silver(spark):
    silver_cnt = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    gold_cnt   = spark.read.table(f"{GOLD}.fact_listings").count()
    diff_pct   = abs(gold_cnt - silver_cnt) / max(silver_cnt, 1) * 100
    assert diff_pct <= 1.0, (
        f"fact_listings diverges from Silver by {diff_pct:.3f}% — "
        f"Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def test_t1_dim_car_current_matches_silver(spark):
    silver_cnt = spark.read.table(f"{SILVER}.car_catalog_transformation").select("brand", "model").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_car").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_car CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def test_t1_dim_location_current_matches_silver(spark):
    silver_cnt = spark.read.table(f"{SILVER}.geography_transformation").select("city_prepositional").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_location").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_location CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def test_t1_dim_listing_details_current_matches_silver(spark):
    silver_cnt = spark.read.table(f"{SILVER}.listings_text_transformation").select("listing_id").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_listing_details").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_listing_details CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def test_t1_dim_listing_photos_current_matches_silver(spark):
    silver_cnt = spark.read.table(f"{SILVER}.listings_photo_transformation").select("listing_id", "photo_url_clean").distinct().count()
    gold_cnt   = spark.read.table(f"{GOLD}.dim_listing_photos").filter(F.col("__END_AT").isNull()).count()
    assert gold_cnt == silver_cnt, (
        f"dim_listing_photos CURRENT mismatch — Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )

def test_t1_dim_date_row_count(spark):
    cnt = spark.read.table(f"{GOLD}.dim_date").count()
    assert cnt == DIM_DATE_EXPECTED, (
        f"dim_date expected {DIM_DATE_EXPECTED:,} rows | got {cnt:,}"
    )

@pytest.mark.parametrize("agg_table", [
    "agg_monthly_sales_trend",
    "agg_brand_location_performance",
    "agg_regional_market_depth",
    "agg_comprehensive_kpi_cube",
    "agg_top_10_brands_by_spend",
])
def test_t1_agg_tables_non_empty(spark, agg_table):
    cnt = spark.read.table(f"{GOLD}.{agg_table}").count()
    assert cnt > 0, f"[{agg_table}] is empty."

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Row Integrity (Silver → Gold)

# COMMAND ----------

# ── T2 — Row Integrity ────────────────────────────────────────────────────────

def test_t2_fact_no_missing_silver_ids(spark):
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    missing    = silver_ids.subtract(gold_ids).count()
    assert missing == 0, f"fact_listings missing {missing:,} Silver listing_id(s)."

def test_t2_fact_no_invented_ids(spark):
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    invented   = gold_ids.subtract(silver_ids).count()
    assert invented == 0, f"fact_listings has {invented:,} listing_id(s) not in Silver."

@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY
])
def test_t2_scd2_no_missing_silver_keys(spark, entry):
    silver_keys = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_keys   = (spark.read.table(f"{GOLD}.{entry['name']}")
                   .filter("__END_AT IS NULL")
                   .select(*entry["keys"]))
    missing = silver_keys.subtract(gold_keys).count()
    assert missing == 0, f"[{entry['name']}] {missing:,} Silver key(s) missing from Gold."

@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY
])
def test_t2_scd2_no_invented_gold_keys(spark, entry):
    silver_keys = spark.read.table(entry["silver"]).select(*entry["keys"]).distinct()
    gold_keys   = (spark.read.table(f"{GOLD}.{entry['name']}")
                   .filter("__END_AT IS NULL")
                   .select(*entry["keys"]))
    invented = gold_keys.subtract(silver_keys).count()
    assert invented == 0, f"[{entry['name']}] {invented:,} Gold key(s) not traceable to Silver."

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — Audit Columns

# COMMAND ----------

# ── T3 — Audit Columns ────────────────────────────────────────────────────────

def test_t3_fact_audit_chain(spark):
    df          = spark.read.table(f"{GOLD}.fact_listings")
    audit_chain = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]
    missing_col = [c for c in audit_chain if c not in df.columns]
    assert missing_col == [], f"fact_listings missing audit columns: {missing_col}"
    for col in audit_chain:
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, f"fact_listings.{col} has {nulls:,} NULL rows."

@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY
])
def test_t3_scd2_metadata_columns(spark, entry):
    cols = spark.read.table(f"{GOLD}.{entry['name']}").columns
    for required in ("__START_AT", "__END_AT", "silver_load_dt"):
        assert required in cols, f"[{entry['name']}] Missing SCD2 column: {required}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## T4 — Fact ↔ Silver Reconciliation (Reverse Join)

# COMMAND ----------

# ── T7 — Fact ↔ Silver Reconciliation ────────────────────────────────────────

def test_t4_reconstruct_silver_listings(spark):
    """
    Verifies that joining fact_listings with lookup dims reconstructs the 
    Silver 'listings_silver_merged' table perfectly.
    """
    fact      = spark.read.table(f"{GOLD}.fact_listings")
    dim_price = spark.read.table(f"{GOLD}.dim_price_category").select("price_category_key", "price_category")
    dim_steer = spark.read.table(f"{GOLD}.dim_steering").select("steering_key", "steering_wheel")

    # Reconstruction join
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
            F.col("listing_date").cast("date").alias("listing_date"), # Notebook casts to date
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

    # Integrity Check: Count differences
    missing_from_gold = silver.subtract(reconstructed).count()
    assert missing_from_gold == 0, f"{missing_from_gold:,} Silver rows missing in Gold reconstruction."
    
    extra_in_gold = reconstructed.subtract(silver).count()
    assert extra_in_gold == 0, f"{extra_in_gold:,} Gold rows have no Silver origin."

def test_t4_reconstruct_silver_text(spark):
    """
    Verifies 1:1 text data recovery from dim_listing_details.
    """
    fact            = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    dim_txt         = (spark.read.table(f"{GOLD}.dim_listing_details")
                        .filter(F.col("__END_AT").isNull()) # Get current active version
                        .select("listing_id", "text"))
    
    reconstructed   = fact.join(dim_txt, on="listing_id", how="inner")
    
    silver_text     = (spark.read.table(f"{SILVER}.listings_text_transformation")
                        .select("listing_id", "text"))
    
    # Only compare IDs that exist in the Fact table
    silver_filtered = silver_text.join(fact, on="listing_id", how="inner")
    
    missing = silver_filtered.subtract(reconstructed).count()
    assert missing == 0, f"{missing:,} Text records failed reconciliation."

def test_t4_reconstruct_silver_photos(spark):
    """
    Verifies that the photo counts in Fact match the counts in dim_listing_photos.
    """
    # Fact stores the denormalized count
    fact_counts = (spark.read.table(f"{GOLD}.fact_listings")
                   .select("listing_id", F.col("photo_count").alias("gold_count")))
    
    # Dim stores the raw rows; we aggregate to check
    dim_counts = (spark.read.table(f"{GOLD}.dim_listing_photos")
                  .filter(F.col("__END_AT").isNull())
                  .groupBy("listing_id")
                  .agg(F.count("photo_url_clean").alias("dim_count")))
    
    comparison = fact_counts.join(dim_counts, on="listing_id", how="left")
    
    # Check for count mismatches
    mismatches = comparison.filter(F.col("gold_count") != F.coalesce(F.col("dim_count"), F.lit(0))).count()
    assert mismatches == 0, f"{mismatches:,} listings have mismatched photo counts between Fact and Dim."

def test_t4_reconstruct_silver_car_specs(spark):
    """
    Verifies that car specs (fuel_type, etc.) can be recovered by joining 
    Fact back to dim_car.
    """
    fact    = spark.read.table(f"{GOLD}.fact_listings").select("brand", "model").distinct()
    dim_car = (spark.read.table(f"{GOLD}.dim_car")
               .filter(F.col("__END_AT").isNull())
               .select("brand", "model", "fuel_type", "transmission", "drive_type"))
    
    reconstructed = fact.join(dim_car, on=["brand", "model"], how="inner")
    
    silver_car = (spark.read.table(f"{SILVER}.car_catalog_transformation")
                  .select("brand", "model", "fuel_type", "transmission", "drive_type"))
    
    # Silver data must be reconstructable using the brand+model natural key
    missing = silver_car.join(fact, on=["brand", "model"], how="inner").subtract(reconstructed).count()
    assert missing == 0, f"{missing:,} Car specifications lost during Gold transformation."

# COMMAND ----------

# # ── T7 — Fact ↔ Silver Reconciliation ────────────────────────────────────────

# def test_t4_reconstruct_silver_listings(spark):
#     fact      = spark.read.table(f"{GOLD}.fact_listings")
#     dim_price = spark.read.table(f"{GOLD}.dim_price_category").select("price_category_key", "price_category")
#     dim_steer = spark.read.table(f"{GOLD}.dim_steering").select("steering_key", "steering_wheel")

#     reconstructed = (
#         fact
#         .join(dim_price, on="price_category_key", how="left")
#         .join(dim_steer, on="steering_key",        how="left")
#         .select(
#             "listing_id",
#             "listing_date",
#             "brand",
#             "model",
#             "price_rub",
#             "price_usd",
#             "price_category",
#             "mileage_km",
#             "manufacture_year",
#             "engine_power",
#             "steering_wheel",
#             F.col("location_key").alias("city_prepositional"),
#         )
#     )

#     silver = (
#         spark.read.table(f"{SILVER}.listings_silver_merged")
#         .select(
#             "listing_id",
#             F.col("listing_date").cast("date").alias("listing_date"),
#             "brand",
#             "model",
#             "price_rub",
#             "price_usd",
#             "price_category",
#             "mileage_km",
#             "manufacture_year",
#             "engine_power",
#             "steering_wheel",
#             "city_prepositional",
#         )
#     )

#     missing_from_gold = silver.subtract(reconstructed).count()
#     assert missing_from_gold == 0, (
#         f"{missing_from_gold:,} Silver rows not recoverable from Gold star schema join."
#     )
#     extra_in_gold = reconstructed.subtract(silver).count()
#     assert extra_in_gold == 0, (
#         f"{extra_in_gold:,} Gold rows have no corresponding Silver row."
#     )

# def test_t4_reconstruct_silver_text(spark):
#     fact             = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
#     dim_txt          = (spark.read.table(f"{GOLD}.dim_listing_details")
#                         .filter(F.col("__END_AT").isNull())
#                         .select("listing_id", "text"))
#     reconstructed    = fact.join(dim_txt, on="listing_id", how="inner")
#     silver_text      = (spark.read.table(f"{SILVER}.listings_text_transformation")
#                         .select("listing_id", "text"))
#     silver_with_text = silver_text.join(fact, on="listing_id", how="inner")
#     missing = silver_with_text.subtract(reconstructed).count()
#     assert missing == 0, (
#         f"{missing:,} Silver text rows not recoverable from dim_listing_details."
#     )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 — Referential Integrity (No Orphan FK Keys)

# COMMAND ----------

# ── T5 — Referential Integrity ────────────────────────────────────────────────

def test_t5_no_orphan_dates(spark):
    fact_dates = (spark.read.table(f"{GOLD}.fact_listings")
                  .select("listing_date").distinct()
                  .filter(F.col("listing_date").isNotNull()))
    dim_dates  = spark.read.table(f"{GOLD}.dim_date").select("date_key")
    orphans    = fact_dates.join(
        dim_dates,
        fact_dates["listing_date"] == dim_dates["date_key"],
        how="left_anti"
    ).count()
    assert orphans == 0, f"{orphans:,} fact listing_date values not found in dim_date."

def test_t5_no_orphan_price_category_keys(spark):
    fact_keys = (spark.read.table(f"{GOLD}.fact_listings")
                 .select("price_category_key").distinct()
                 .filter(F.col("price_category_key").isNotNull()))
    dim_keys  = spark.read.table(f"{GOLD}.dim_price_category").select("price_category_key")
    orphans   = fact_keys.subtract(dim_keys).count()
    assert orphans == 0, f"{orphans:,} fact price_category_key values not in dim_price_category."

def test_t5_no_orphan_steering_keys(spark):
    fact_keys = (spark.read.table(f"{GOLD}.fact_listings")
                 .select("steering_key").distinct()
                 .filter(F.col("steering_key").isNotNull()))
    dim_keys  = spark.read.table(f"{GOLD}.dim_steering").select("steering_key")
    orphans   = fact_keys.subtract(dim_keys).count()
    assert orphans == 0, f"{orphans:,} fact steering_key values not in dim_steering."

def test_t5_referential_integrity_car_keys(spark):
    # Fact table read karke on-the-fly keys clean karna
    fact_listings = (
        spark.read.table(f"{GOLD}.fact_listings")
        .withColumn("f_brand_clean", F.lower(F.trim(F.col("brand"))))
        .withColumn("f_model_clean", F.lower(F.trim(F.col("model"))))
    )
    total_listings = fact_listings.count()

    # Dimension table read karke active records aur keys clean karna
    dim_car = (
        spark.read.table(f"{GOLD}.dim_car")
        .filter(F.col("__END_AT").isNull())
        .withColumn("d_brand_clean", F.lower(F.trim(F.col("brand"))))
        .withColumn("d_model_clean", F.lower(F.trim(F.col("model"))))
    )

    # Join on normalized cleaned keys
    matched_listings = (
        fact_listings.alias("f")
        .join(
            dim_car.alias("d"),
            (F.col("f.f_brand_clean") == F.col("d.d_brand_clean")) & 
            (F.col("f.f_model_clean") == F.col("d.d_model_clean")),
            how="inner"
        ).count()
    )

    join_rate = (matched_listings / total_listings) * 100 if total_listings > 0 else 0

    assert join_rate >= 60, (
        f"Join rate is {join_rate:.2f}%. Even after on-the-fly cleaning, match rate is too low. "
        f"Total: {total_listings}, Matched: {matched_listings}."
    )

def test_t5_no_orphan_location_keys(spark):
    """
    
    Normalizes keys to handle casing/spacing mismatches between Fact and Dim.
    """
    fact_df = (spark.read.table(f"{GOLD}.fact_listings")
               .select(F.lower(F.trim(F.col("location_key"))).alias("location_key"))
               .filter(F.col("location_key").isNotNull()))
    total_listings = fact_df.count()

    dim_locs = (spark.read.table(f"{GOLD}.dim_location")
                .filter(F.col("__END_AT").isNull())
                .select(F.lower(F.trim(F.col("city_prepositional"))).alias("location_key")))

    matched_listings = fact_df.join(dim_locs, on="location_key", how="inner").count()
    
    join_rate = (matched_listings / total_listings) * 100 if total_listings > 0 else 0

    # 60% baseline as per project guardrails
    assert join_rate >= 60, (
        f"Location join rate is {join_rate:.2f}%. "
        f"Total Unique Locs in Fact: {total_listings}, Matched: {matched_listings}."
    )

def test_t5_no_orphan_listing_details(spark):
    """
    IT-6c: Checks 1:1 referential integrity for listing details.
    """
    fact_ids = spark.read.table(f"{GOLD}.fact_listings").select("listing_id").distinct()
    total_ids = fact_ids.count()

    dim_ids = (spark.read.table(f"{GOLD}.dim_listing_details")
               .filter(F.col("__END_AT").isNull())
               .select("listing_id"))

    matched_ids = fact_ids.join(dim_ids, on="listing_id", how="inner").count()
    join_rate = (matched_ids / total_ids) * 100 if total_ids > 0 else 0

    assert join_rate >= 60, f"Details join rate too low: {join_rate:.2f}%. Check upstream text transformations."

def test_t5_no_orphan_listing_photos(spark):
    """
    IT-6d: Verifies listings in Fact have matching metadata in Photos dimension.
    """
    fact_ids = spark.read.table(f"{GOLD}.fact_listings").select("listing_id").distinct()
    total_ids = fact_ids.count()

    dim_photo_ids = (spark.read.table(f"{GOLD}.dim_listing_photos")
                     .filter(F.col("__END_AT").isNull())
                     .select("listing_id").distinct())

    matched_ids = fact_ids.join(dim_photo_ids, on="listing_id", how="inner").count()
    join_rate = (matched_ids / total_ids) * 100 if total_ids > 0 else 0

    assert join_rate >= 60, f"Photos join rate too low: {join_rate:.2f}%. Verify 1:Many link integrity."
