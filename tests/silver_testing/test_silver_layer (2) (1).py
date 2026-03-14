# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer Test Suite
# MAGIC
# MAGIC | Suite | What it checks |
# MAGIC |-------|----------------|
# MAGIC | T1    | Reconciliation — Bronze row count balances Silver + Quarantine |
# MAGIC | T2    | Subset integrity — every Silver PK exists in Bronze (Silver ⊆ Bronze) |
# MAGIC | T3    | Audit columns — present, non-null, correct types |
# MAGIC | T4    | Schema & domain constraints — column types, positive prices, valid dates, geo bounds |
# MAGIC | T5    | Deduplication — no duplicate PKs survive into Silver |
# MAGIC | T6    | Quarantine hygiene — quarantine_reason populated, dt present |
# MAGIC | T7    | Derived columns — price_usd, car_age_years, price_category, brand_std correctness |

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
}

B = lambda t: f"{CONFIG['catalog']}.{CONFIG['bronze']}.{t}"
S = lambda t: f"{CONFIG['catalog']}.{CONFIG['silver']}.{t}"

USD_RATE = 82.5  # Must mirror 08_silver_transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform Helpers
# MAGIC Row-level normalizers mirroring `08_silver_transformation` — used by T1 dedup counts.

# COMMAND ----------

# ── ID casts ──────────────────────────────────────────────────────────────────
def _id_main(col): return F.expr(f"try_cast(`{col}` as long)").cast("string")
def _id_dbl(col):  return F.col(f"`{col}`").cast("double").cast("long").cast("string")

# ── String transforms (must match pandas UDF behaviour in pipeline) ───────────
def _std(col):     # standardize_text: lower + strip; null → "none"
    return F.when(F.col(f"`{col}`").isNull(), F.lit("none")) \
            .otherwise(F.lower(F.trim(F.col(f"`{col}`"))))

def _clean(col):   # clean_text: strip only; null → "None"
    return F.when(F.col(f"`{col}`").isNull(), F.lit("None")) \
            .otherwise(F.trim(F.col(f"`{col}`")))

def _geo(col):     # standardize_geo: strip only; null → "None"
    return F.when(F.col(f"`{col}`").isNull(), F.lit("None")) \
            .otherwise(F.trim(F.col(f"`{col}`")))

# ── Passthrough / simple casts ────────────────────────────────────────────────
def _pass(col):  return F.col(f"`{col}`")
def _dbl(col):   return F.col(f"`{col}`").cast("double")

# ── Numeric / domain transforms ───────────────────────────────────────────────
def _price(col):   return F.expr(f"try_cast(regexp_replace(`{col}`, '[^0-9.]', '') as double)")
def _int2(col):    return F.expr(f"try_cast(try_cast(`{col}` as double) as int)")
def _date(col):
    return F.coalesce(
        F.try_to_timestamp(F.col(f"`{col}`"), F.lit("dd.MM.yyyy")),
        F.try_to_timestamp(F.col(f"`{col}`"), F.lit("yyyy-MM-dd'T'HH:mm:ss'Z'"))
    )
def _eng_vol(col): return F.expr(f"try_cast(regexp_replace(regexp_replace(`{col}`,' л',''),',','.') as double)")
def _eng_pow(col): return F.expr(f"try_cast(regexp_replace(`{col}`,' л.с.','') as int)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registry
# MAGIC Single source of truth for all 5 Silver tables.
# MAGIC
# MAGIC **`pk_bronze_col`** maps each Silver primary-key column back to the raw Bronze column
# MAGIC and the cast function needed to make them comparable — used by T2 subset test.

# COMMAND ----------

REGISTRY = [
    # ── listings_silver_merged ────────────────────────────────────────────────
    {
        "name"         : "listings_silver_merged",
        "silver"       : S("listings_silver_merged"),
        "quarantine"   : S("listings_main_quarantine"),
        "bronze_sources": [
            B("listings_csv_copyinto"),
            B("listings_json_autoloader"),
            B("listings_xml_pyspark"),
            B("listings_csv_dlt"),
        ],
        "primary_key"  : ["listing_id"],
        "audit_cols"   : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        # T1 dedup key — how Bronze IDs are normalised before counting unique keys
        "t1_dedup_exprs": [("id", "listing_id", _id_main)],
        "t1_use_simple_eq": False,
        # T2 — map Silver PK column → (bronze_raw_col, cast_fn)
        # We compare Silver listing_id against Bronze id cast the same way
        "pk_bronze_col": {"listing_id": ("id", _id_main)},
        # T4 domain checks
        "not_null_cols" : ["listing_id", "listing_date", "price_rub"],
        "positive_cols" : ["price_rub"],
        "timestamp_cols": ["listing_date", "silver_load_dt"],
        # T7 derived column verification
        "has_derived"   : True,
    },

    # ── car_catalog_transformation ────────────────────────────────────────────
    {
        "name"         : "car_catalog_transformation",
        "silver"       : S("car_catalog_transformation"),
        "quarantine"   : S("car_catalog_quarantine"),
        "bronze_sources": [B("car_catalog")],
        "primary_key"  : ["brand", "model", "generation"],
        "audit_cols"   : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs": [
            ("Марка",              "brand",           _clean),
            ("Модель",             "model",           _clean),
            ("Поколение",          "generation",      _clean),
            ("Комплектация",       "trim_level",      _clean),
            ("Объём двигателя",    "engine_volume_l", _eng_vol),
            ("Мощность двигателя", "engine_power_hp", _eng_pow),
        ],
        "t1_use_simple_eq": True,
        "pk_bronze_col": {
            "brand"      : ("Марка",     _clean),
            "model"      : ("Модель",    _clean),
            "generation" : ("Поколение", _clean),
        },
        "not_null_cols" : ["brand", "model"],
        "positive_cols" : [],
        "timestamp_cols": ["silver_load_dt"],
        "has_derived"   : False,
    },

    # ── listings_text_transformation ──────────────────────────────────────────
    {
        "name"         : "listings_text_transformation",
        "silver"       : S("listings_text_transformation"),
        "quarantine"   : S("listings_text_quarantine"),
        "bronze_sources": [B("listings_text")],
        "primary_key"  : ["listing_id"],
        "audit_cols"   : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs": [("id", "listing_id", _id_dbl)],
        "t1_use_simple_eq": False,
        "pk_bronze_col": {"listing_id": ("id", _id_dbl)},
        "not_null_cols" : ["listing_id", "text"],
        "positive_cols" : [],
        "timestamp_cols": ["silver_load_dt"],
        "has_derived"   : False,
    },

    # ── listings_photo_transformation ─────────────────────────────────────────
    {
        "name"         : "listings_photo_transformation",
        "silver"       : S("listings_photo_transformation"),
        "quarantine"   : S("listings_photo_quarantine"),
        "bronze_sources": [B("listings_photo")],
        "primary_key"  : ["listing_id"],
        "audit_cols"   : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs": [
            ("id",        "listing_id",      _id_dbl),
            ("photo_url", "photo_url_clean", _std),
        ],
        "t1_use_simple_eq": False,
        "pk_bronze_col": {"listing_id": ("id", _id_dbl)},
        "not_null_cols" : ["listing_id", "photo_url"],
        "positive_cols" : [],
        "timestamp_cols": ["silver_load_dt"],
        "has_derived"   : False,
    },

    # ── geography_transformation ──────────────────────────────────────────────
    {
        "name"         : "geography_transformation",
        "silver"       : S("geography_transformation"),
        "quarantine"   : S("geography_quarantine"),
        "bronze_sources": [B("geo_locations")],
        "primary_key"  : ["city_name"],
        "audit_cols"   : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs": [
            ("name_padesh",   "city_name",         _geo),
            ("greate_padesh", "city_prepositional", _pass),
            ("lat",           "latitude",           _dbl),
            ("lon",           "longitude",          _dbl),
        ],
        "t1_use_simple_eq": False,
        "t1_valid_filter": lambda df: df.filter(
            F.col("latitude").isNotNull()  &
            F.col("longitude").isNotNull() &
            F.col("latitude").between(41, 82) &
            F.col("longitude").between(19, 180)
        ),
        "pk_bronze_col": {"city_name": ("name_padesh", _geo)},
        "not_null_cols" : ["city_name", "latitude", "longitude"],
        "positive_cols" : [],
        "timestamp_cols": ["silver_load_dt"],
        "has_derived"   : False,
    },
]

REGISTRY_PARAMS = [pytest.param(e, id=e["name"]) for e in REGISTRY]

print(f"Registry loaded — {len(REGISTRY)} tables registered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def _union_bronze(spark, sources: list):
    """Unions all Bronze source tables, tolerating missing columns."""
    df = None
    for src in sources:
        b  = spark.read.table(src)
        df = b if df is None else df.unionByName(b, allowMissingColumns=True)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Reconciliation
# MAGIC Bronze row count must balance Silver + Quarantine (allowing for deduplication).

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t1_reconciliation(spark, entry):
    """
    T1 — Bronze row count must equal Silver + Quarantine row count.
    For tables with dedup keys, unique Bronze key count is used instead of raw count.
    """
    bronze_total = sum(spark.read.table(s).count() for s in entry["bronze_sources"])
    silver_cnt   = spark.read.table(entry["silver"]).count()
    quar_cnt     = spark.read.table(entry["quarantine"]).count()
    actual       = silver_cnt + quar_cnt

    if entry.get("t1_use_simple_eq", False):
        dups  = bronze_total - actual
        assert dups >= 0, (
            f"[{entry['name']}] Row count mismatch — "
            f"Raw: {bronze_total:,} | Actual(S+Q): {actual:,} | Dups Dropped: {dups:,}"
        )
    else:
        df_bronze   = _union_bronze(spark, entry["bronze_sources"])
        dedup_exprs = entry.get("t1_dedup_exprs", [])
        df_dedup    = df_bronze.select(
            [tfn(col).alias(alias) for col, alias, tfn in dedup_exprs]
        )
        # Apply geography valid filter before counting if declared
        if entry.get("t1_valid_filter"):
            valid   = entry["t1_valid_filter"](df_dedup).distinct().count()
            invalid = df_dedup.filter(
                ~(
                    F.col("latitude").isNotNull()  &
                    F.col("longitude").isNotNull() &
                    F.col("latitude").between(41, 82) &
                    F.col("longitude").between(19, 180)
                )
            ).distinct().count()
            unique_exp = valid + invalid
        else:
            unique_exp = df_dedup.distinct().count()

        assert actual == unique_exp, (
            f"[{entry['name']}] Reconciliation mismatch — "
            f"Raw: {bronze_total:,} | Unique_Exp: {unique_exp:,} | "
            f"Actual(S+Q): {actual:,} | Dups Dropped: {bronze_total - unique_exp:,}"
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t1_silver_table_non_empty(spark, entry):
    """T1 — Every Silver table must contain at least 1 row."""
    count = spark.read.table(entry["silver"]).count()
    assert count > 0, f"[{entry['name']}] Silver table is empty."


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t1_quarantine_has_rejection_reasons(spark, entry):
    """T1 — Every row in Quarantine must have a non-null quarantine_reason."""
    df = spark.read.table(entry["quarantine"])
    if df.count() > 0 and "quarantine_reason" in df.columns:
        null_reasons = df.filter(F.col("quarantine_reason").isNull()).count()
        assert null_reasons == 0, (
            f"[{entry['name']}] {null_reasons:,} quarantine rows missing quarantine_reason."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Silver PK ⊆ Bronze PK (Subset Integrity)
# MAGIC
# MAGIC **Replaces the row-to-row SHA-256 hash test.**
# MAGIC
# MAGIC Silver is allowed to transform, enrich, and derive new columns from Bronze — row hashes
# MAGIC will naturally differ. What must hold is that every Silver primary key (e.g. `listing_id`)
# MAGIC originates from a real Bronze record — Silver cannot invent new keys.
# MAGIC
# MAGIC This test uses a `left_anti` join: Silver PKs that find **no match** in Bronze are violations.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t2_silver_pks_are_subset_of_bronze(spark, entry):
    """
    T2 — Every Silver primary-key value must exist in at least one Bronze source.
    Silver ⊆ Bronze on primary key(s). Nothing may be invented.

    Strategy:
      1. Extract and cast Bronze PK column(s) using the same cast function as the pipeline.
      2. Extract Silver PK column(s) as-is.
      3. left_anti join Silver onto Bronze on the PK(s).
      4. Assert the anti-join result is empty.
    """
    pk_map     = entry["pk_bronze_col"]          # {silver_col: (bronze_raw_col, cast_fn)}
    silver_pks = entry["primary_key"]             # list of Silver column names

    df_silver = spark.read.table(entry["silver"]).select(*silver_pks).distinct()

    # Build normalised Bronze PK DataFrame from all Bronze sources
    df_bronze_raw = _union_bronze(spark, entry["bronze_sources"])
    bronze_select = [
        pk_map[s_col][1](pk_map[s_col][0]).alias(s_col)
        for s_col in silver_pks
    ]
    df_bronze_pks = df_bronze_raw.select(bronze_select).distinct()

    orphaned = df_silver.join(df_bronze_pks, on=silver_pks, how="left_anti").count()

    assert orphaned == 0, (
        f"[{entry['name']}] {orphaned:,} Silver PK(s) have no matching record in Bronze. "
        f"PKs checked: {silver_pks}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — Audit Columns
# MAGIC Verifies that all audit metadata columns are present, non-null, and correctly typed.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_audit_columns_present(spark, entry):
    """T3 — All declared audit columns must exist in the Silver table schema."""
    df      = spark.read.table(entry["silver"])
    missing = [c for c in entry["audit_cols"] if c not in df.columns]
    assert missing == [], f"[{entry['name']}] Missing audit columns: {missing}"


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_audit_columns_non_null(spark, entry):
    """T3 — All declared audit columns must have zero null values."""
    df = spark.read.table(entry["silver"])
    for col in entry["audit_cols"]:
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['name']}] Audit column '{col}' has {null_cnt:,} NULL rows."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_silver_load_dt_is_timestamp(spark, entry):
    """T3 — silver_load_dt must be TIMESTAMP type, never string or date."""
    df     = spark.read.table(entry["silver"])
    dtypes = dict(df.dtypes)
    if "silver_load_dt" in dtypes:
        assert dtypes["silver_load_dt"].startswith("timestamp"), (
            f"[{entry['name']}] silver_load_dt is '{dtypes['silver_load_dt']}', "
            "expected 'timestamp'."
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_primary_key_non_null(spark, entry):
    """T3 — Primary key columns must never be null."""
    df = spark.read.table(entry["silver"])
    for pk_col in entry["primary_key"]:
        if pk_col in df.columns:
            null_cnt = df.filter(F.col(pk_col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['name']}] Primary key '{pk_col}' has {null_cnt:,} NULL rows."
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T4 — Schema & Domain Constraints
# MAGIC Validates column types, non-null critical columns, positive numeric values, and geo bounds.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t4_critical_columns_not_null(spark, entry):
    """
    T4 — Columns declared in 'not_null_cols' must have zero nulls in Silver.
    These mirror the @dlt.expect constraints in 08_silver_transformation.
    """
    df = spark.read.table(entry["silver"])
    for col in entry.get("not_null_cols", []):
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['name']}] Column '{col}' has {null_cnt:,} NULL rows "
                "(violates DLT expect constraint)."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t4_positive_numeric_columns(spark, entry):
    """
    T4 — Columns in 'positive_cols' must be strictly > 0 in Silver.
    Mirrors the @dlt.expect("positive_price", "price_rub > 0") constraint.
    """
    df = spark.read.table(entry["silver"])
    for col in entry.get("positive_cols", []):
        if col in df.columns:
            bad_cnt = df.filter(
                F.col(col).isNotNull() & (F.col(col) <= 0)
            ).count()
            assert bad_cnt == 0, (
                f"[{entry['name']}] Column '{col}' has {bad_cnt:,} rows with value ≤ 0."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t4_timestamp_columns_are_correct_type(spark, entry):
    """
    T4 — Columns in 'timestamp_cols' must be TIMESTAMP type, not string or date.
    """
    df     = spark.read.table(entry["silver"])
    dtypes = dict(df.dtypes)
    for col in entry.get("timestamp_cols", []):
        if col in dtypes:
            assert dtypes[col].startswith("timestamp"), (
                f"[{entry['name']}] Column '{col}' has type '{dtypes[col]}', "
                "expected 'timestamp'."
            )


def test_t4_geography_russia_bounds(spark):
    """
    T4 — All rows in geography_transformation must be within Russia bounding box
    (lat 41–82N, lon 19–180E). Mirrors the @dlt.expect("russia_bounds") constraint.
    """
    df  = spark.read.table(S("geography_transformation"))
    bad = df.filter(
        ~(
            F.col("latitude").between(41, 82) &
            F.col("longitude").between(19, 180)
        )
    ).count()
    assert bad == 0, (
        f"[geography_transformation] {bad:,} rows outside Russia bounding box "
        "(lat 41-82N, lon 19-180E)."
    )


def test_t4_listing_date_range_is_reasonable(spark):
    """
    T4 — listing_date in listings_silver_merged must fall within a plausible range.
    Any listing dated before 2000 or after today is likely a parse error.
    """
    df  = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("listing_date").isNotNull() &
        (
            (F.year(F.col("listing_date")) < 2000) |
            (F.col("listing_date") > F.current_timestamp())
        )
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows with listing_date outside "
        "plausible range (2000 – now)."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 — Deduplication
# MAGIC Verifies that `dropDuplicates` in the pipeline actually removed all duplicate PKs from Silver.
# MAGIC Tables where multiple rows per PK are expected (e.g. photos) are skipped automatically.

# COMMAND ----------

# Tables where PK uniqueness is NOT expected (one listing can have many photos)
_MULTI_ROW_PK_TABLES = {"listings_photo_transformation", "car_catalog_transformation"}


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t5_no_duplicate_primary_keys_in_silver(spark, entry):
    """
    T5 — Silver must have no duplicate rows on its primary key(s).
    Skipped for tables with intentionally non-unique PKs (photos, catalog).
    """
    if entry["name"] in _MULTI_ROW_PK_TABLES:
        pytest.skip(
            f"[{entry['name']}] PK uniqueness not enforced — skipping dedup check."
        )

    df    = spark.read.table(entry["silver"])
    total = df.count()
    uniq  = df.select(*entry["primary_key"]).distinct().count()

    assert total == uniq, (
        f"[{entry['name']}] Duplicate PKs detected — "
        f"total rows: {total:,}, distinct PKs: {uniq:,}, "
        f"duplicates: {total - uniq:,}. PKs: {entry['primary_key']}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T6 — Quarantine Hygiene
# MAGIC Verifies that every quarantine table is well-formed: reason populated, timestamp present,
# MAGIC and no valid rows accidentally quarantined.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t6_quarantine_reason_non_null(spark, entry):
    """
    T6 — Every row in the quarantine table must carry a non-null quarantine_reason.
    """
    df = spark.read.table(entry["quarantine"])
    if df.count() == 0:
        return  # empty quarantine is fine
    assert "quarantine_reason" in df.columns, (
        f"[{entry['name']}] quarantine table missing 'quarantine_reason' column."
    )
    null_cnt = df.filter(F.col("quarantine_reason").isNull()).count()
    assert null_cnt == 0, (
        f"[{entry['name']}] {null_cnt:,} quarantine rows have NULL quarantine_reason."
    )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t6_quarantine_dt_present_and_non_null(spark, entry):
    """
    T6 — quarantine_dt must exist and be non-null in every quarantine table that has rows.
    """
    df = spark.read.table(entry["quarantine"])
    if df.count() == 0:
        return
    assert "quarantine_dt" in df.columns, (
        f"[{entry['name']}] quarantine table missing 'quarantine_dt' column."
    )
    null_cnt = df.filter(F.col("quarantine_dt").isNull()).count()
    assert null_cnt == 0, (
        f"[{entry['name']}] {null_cnt:,} quarantine rows have NULL quarantine_dt."
    )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t6_silver_and_quarantine_pks_are_disjoint(spark, entry):
    """
    T6 — A PK must not appear in both Silver and Quarantine simultaneously.
    A record should be routed to exactly one destination.
    Skipped for tables where PK uniqueness is not enforced (photos, catalog).
    """
    if entry["name"] in _MULTI_ROW_PK_TABLES:
        pytest.skip(
            f"[{entry['name']}] PK uniqueness not enforced — skipping disjoint check."
        )

    pk_cols   = entry["primary_key"]
    df_silver = spark.read.table(entry["silver"]).select(*pk_cols)
    df_quar   = spark.read.table(entry["quarantine"]).select(*pk_cols)

    overlap = df_silver.join(df_quar, on=pk_cols, how="inner").count()
    assert overlap == 0, (
        f"[{entry['name']}] {overlap:,} PKs appear in both Silver and Quarantine. "
        f"Each record must be routed to exactly one. PKs: {pk_cols}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T7 — Derived Column Correctness (listings_silver_merged)
# MAGIC Validates that enrichment columns computed by the pipeline are mathematically correct
# MAGIC and internally consistent with source columns in the same Silver row.

# COMMAND ----------

def test_t7_price_usd_derived_correctly(spark):
    """
    T7 — price_usd must equal round(price_rub / 82.5, 2) for every non-null row.
    Mirrors: .withColumn("price_usd", F.round(F.col("price_rub") / USD_RATE, 2))
    """
    df = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("price_rub").isNotNull() & F.col("price_usd").isNotNull() &
        (F.abs(F.col("price_usd") - F.round(F.col("price_rub") / USD_RATE, 2)) > 0.01)
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows where price_usd != round(price_rub / {USD_RATE}, 2)."
    )


def test_t7_car_age_years_derived_correctly(spark):
    """
    T7 — car_age_years must equal (2023 - manufacture_year) for every non-null row.
    Mirrors: .withColumn("car_age_years", F.lit(2023) - F.col("manufacture_year"))
    """
    df = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("manufacture_year").isNotNull() & F.col("car_age_years").isNotNull() &
        (F.col("car_age_years") != (F.lit(2023) - F.col("manufacture_year").cast("integer")))
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows where car_age_years != (2023 - manufacture_year)."
    )


def test_t7_price_category_values_are_valid(spark):
    """
    T7 — price_category must only contain the 5 defined labels.
    Mirrors the F.when(...).otherwise("UNKNOWN") logic in _transform_listings.
    """
    valid_cats = {"BUDGET", "MID_RANGE", "PREMIUM", "LUXURY", "UNKNOWN"}
    df = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("price_category").isNotNull() &
        ~F.col("price_category").isin(list(valid_cats))
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows with invalid price_category value "
        f"(allowed: {valid_cats})."
    )


def test_t7_price_category_bucket_boundaries(spark):
    """
    T7 — price_category bucket labels must match the threshold boundaries exactly.
    e.g. price_rub < 300000 → BUDGET; 300000–700000 → MID_RANGE, etc.
    """
    df = spark.read.table(S("listings_silver_merged")).filter(
        F.col("price_rub").isNotNull() & F.col("price_category").isNotNull()
    )

    checks = [
        ("BUDGET",    (F.col("price_rub") < 300_000)),
        ("MID_RANGE", (F.col("price_rub").between(300_000, 700_000))),
        ("PREMIUM",   (F.col("price_rub").between(700_001, 1_500_000))),
        ("LUXURY",    (F.col("price_rub") > 1_500_000)),
    ]
    for label, condition in checks:
        # Rows labelled as <label> that DON'T satisfy the expected price range
        bad = df.filter((F.col("price_category") == label) & ~condition).count()
        assert bad == 0, (
            f"[listings_silver_merged] {bad:,} rows labelled '{label}' "
            "don't satisfy the expected price_rub range."
        )


def test_t7_brand_std_is_uppercase_brand(spark):
    """
    T7 — brand_std must equal upper(trim(brand)) for all non-null rows.
    Mirrors: .withColumn("brand_std", F.upper(F.trim(F.col("brand"))))
    """
    df = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("brand").isNotNull() & F.col("brand_std").isNotNull() &
        (F.col("brand_std") != F.upper(F.trim(F.col("brand"))))
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows where brand_std != upper(trim(brand))."
    )


def test_t7_listing_year_month_match_listing_date(spark):
    """
    T7 — listing_year and listing_month must be consistent with listing_date.
    Mirrors: .withColumn("listing_year", F.date_format("listing_date", "yyyy").cast("integer"))
             .withColumn("listing_month", F.date_format("listing_date", "MM").cast("integer"))
    """
    df = spark.read.table(S("listings_silver_merged")).filter(
        F.col("listing_date").isNotNull()
    )
    bad_year = df.filter(
        F.col("listing_year").isNotNull() &
        (F.col("listing_year") != F.year(F.col("listing_date")))
    ).count()
    bad_month = df.filter(
        F.col("listing_month").isNotNull() &
        (F.col("listing_month") != F.month(F.col("listing_date")))
    ).count()
    assert bad_year == 0, (
        f"[listings_silver_merged] {bad_year:,} rows where listing_year != year(listing_date)."
    )
    assert bad_month == 0, (
        f"[listings_silver_merged] {bad_month:,} rows where listing_month != month(listing_date)."
    )
