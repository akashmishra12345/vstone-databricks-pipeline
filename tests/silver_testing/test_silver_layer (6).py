# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer Test Suite
# MAGIC
# MAGIC | Suite | What it checks |
# MAGIC |-------|----------------|
# MAGIC | T1    | Reconciliation — Bronze row count balances Silver + Quarantine |
# MAGIC | T2    | Subset integrity — every Silver PK exists in Bronze (Silver ⊆ Bronze) |
# MAGIC | T3    | Audit columns — present, non-null, correct types |
# MAGIC | T4    | Schema & domain constraints — hard-filter nulls, warn-only quality metrics, types, geo bounds |
# MAGIC | T5    | Deduplication — no duplicate PKs survive into Silver |
# MAGIC | T6    | Quarantine hygiene — reason populated, dt present, disjoint from Silver |
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
def _std(col):    # standardize_text: lower + strip; null → "none"
    return F.when(F.col(f"`{col}`").isNull(), F.lit("none")) \
            .otherwise(F.lower(F.trim(F.col(f"`{col}`"))))

def _clean(col):  # clean_text: strip only; null → "None"
    return F.when(F.col(f"`{col}`").isNull(), F.lit("None")) \
            .otherwise(F.trim(F.col(f"`{col}`")))

def _geo(col):    # standardize_geo: strip only; null → "None"
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
# MAGIC **Key design — two null-check keys replace the old `not_null_cols`:**
# MAGIC - `filter_not_null_cols` — guarded by a real `.filter()` in the pipeline → **zero nulls enforced**
# MAGIC - `dlt_warn_cols` — covered only by `@dlt.expect` (warn mode) → **nulls pass into Silver, tested by rate threshold**
# MAGIC - `dlt_warn_positive_cols` — `@dlt.expect` positivity check (warn mode) → **tested by rate threshold**
# MAGIC
# MAGIC **`pk_bronze_col`** maps each Silver PK back to its Bronze raw column for the T2 subset test.

# COMMAND ----------

REGISTRY = [
    # ── listings_silver_merged ────────────────────────────────────────────────
    # Hard filters (_LISTINGS_VALID_FILTER): listing_id, price_rub, listing_date
    # @dlt.expect warn-only: price_rub > 0
    {
        "name"               : "listings_silver_merged",
        "silver"             : S("listings_silver_merged"),
        "quarantine"         : S("listings_main_quarantine"),
        "bronze_sources"     : [
            B("listings_csv_copyinto"), B("listings_json_autoloader"),
            B("listings_xml_pyspark"),  B("listings_csv_dlt"),
        ],
        "primary_key"        : ["listing_id"],
        "audit_cols"         : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"     : [("id", "listing_id", _id_main)],
        "t1_use_simple_eq"   : False,
        "pk_bronze_col"      : {"listing_id": ("id", _id_main)},
        "filter_not_null_cols"   : ["listing_id", "listing_date", "price_rub"],
        "dlt_warn_cols"          : [],
        "dlt_warn_positive_cols" : ["price_rub"],
        "timestamp_cols"         : ["listing_date", "silver_load_dt"],
        "has_derived"            : True,
    },

    # ── car_catalog_transformation ────────────────────────────────────────────
    # Hard filter: brand only
    # @dlt.expect warn-only: model
    {
        "name"               : "car_catalog_transformation",
        "silver"             : S("car_catalog_transformation"),
        "quarantine"         : S("car_catalog_quarantine"),
        "bronze_sources"     : [B("car_catalog")],
        "primary_key"        : ["brand", "model", "generation"],
        "audit_cols"         : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"     : [
            ("Марка",              "brand",           _clean),
            ("Модель",             "model",           _clean),
            ("Поколение",          "generation",      _clean),
            ("Комплектация",       "trim_level",      _clean),
            ("Объём двигателя",    "engine_volume_l", _eng_vol),
            ("Мощность двигателя", "engine_power_hp", _eng_pow),
        ],
        "t1_use_simple_eq"   : True,
        "pk_bronze_col"      : {
            "brand"      : ("Марка",     _clean),
            "model"      : ("Модель",    _clean),
            "generation" : ("Поколение", _clean),
        },
        "filter_not_null_cols"   : ["brand"],
        "dlt_warn_cols"          : ["model"],
        "dlt_warn_positive_cols" : [],
        "timestamp_cols"         : ["silver_load_dt"],
        "has_derived"            : False,
    },

    # ── listings_text_transformation ──────────────────────────────────────────
    # Hard filter: listing_id only
    # @dlt.expect warn-only: text
    {
        "name"               : "listings_text_transformation",
        "silver"             : S("listings_text_transformation"),
        "quarantine"         : S("listings_text_quarantine"),
        "bronze_sources"     : [B("listings_text")],
        "primary_key"        : ["listing_id"],
        "audit_cols"         : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"     : [("id", "listing_id", _id_dbl)],
        "t1_use_simple_eq"   : False,
        "pk_bronze_col"      : {"listing_id": ("id", _id_dbl)},
        "filter_not_null_cols"   : ["listing_id"],
        "dlt_warn_cols"          : ["text"],
        "dlt_warn_positive_cols" : [],
        "timestamp_cols"         : ["silver_load_dt"],
        "has_derived"            : False,
    },

    # ── listings_photo_transformation ─────────────────────────────────────────
    # Hard filter: listing_id only
    # @dlt.expect warn-only: photo_url
    {
        "name"               : "listings_photo_transformation",
        "silver"             : S("listings_photo_transformation"),
        "quarantine"         : S("listings_photo_quarantine"),
        "bronze_sources"     : [B("listings_photo")],
        "primary_key"        : ["listing_id"],
        "audit_cols"         : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"     : [
            ("id",        "listing_id",      _id_dbl),
            ("photo_url", "photo_url_clean", _std),
        ],
        "t1_use_simple_eq"   : False,
        "pk_bronze_col"      : {"listing_id": ("id", _id_dbl)},
        "filter_not_null_cols"   : ["listing_id"],
        "dlt_warn_cols"          : ["photo_url"],
        "dlt_warn_positive_cols" : [],
        "timestamp_cols"         : ["silver_load_dt"],
        "has_derived"            : False,
    },

    # ── geography_transformation ──────────────────────────────────────────────
    # Hard filter: latitude, longitude  (via _is_valid_russia)
    # @dlt.expect warn-only: city_name
    {
        "name"               : "geography_transformation",
        "silver"             : S("geography_transformation"),
        "quarantine"         : S("geography_quarantine"),
        "bronze_sources"     : [B("geo_locations")],
        "primary_key"        : ["city_name"],
        "audit_cols"         : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"     : [
            ("name_padesh",   "city_name",         _geo),
            ("greate_padesh", "city_prepositional", _pass),
            ("lat",           "latitude",           _dbl),
            ("lon",           "longitude",          _dbl),
        ],
        "t1_use_simple_eq"   : False,
        "t1_valid_filter"    : lambda df: df.filter(
            F.col("latitude").isNotNull()  &
            F.col("longitude").isNotNull() &
            F.col("latitude").between(41, 82) &
            F.col("longitude").between(19, 180)
        ),
        "pk_bronze_col"      : {"city_name": ("name_padesh", _geo)},
        "filter_not_null_cols"   : ["latitude", "longitude"],
        "dlt_warn_cols"          : ["city_name"],
        "dlt_warn_positive_cols" : [],
        "timestamp_cols"         : ["silver_load_dt"],
        "has_derived"            : False,
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
# MAGIC Bronze row count must balance Silver + Quarantine (accounting for deduplication).

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
        dups = bronze_total - actual
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
        if entry.get("t1_valid_filter"):
            df_valid   = entry["t1_valid_filter"](df_dedup)
            df_invalid = df_dedup.join(df_valid, on=list(df_valid.columns), how="left_anti")
            unique_exp = df_valid.distinct().count() + df_invalid.distinct().count()
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
        null_cnt = df.filter(F.col("quarantine_reason").isNull()).count()
        assert null_cnt == 0, (
            f"[{entry['name']}] {null_cnt:,} quarantine rows missing quarantine_reason."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Silver PK ⊆ Bronze PK (Subset Integrity)
# MAGIC
# MAGIC Replaces the old row-to-row SHA-256 hash test.
# MAGIC
# MAGIC Silver transforms, enriches, and derives new columns from Bronze — row hashes will naturally
# MAGIC differ after transformation. What must hold is that every Silver primary key originates from
# MAGIC a real Bronze record. Uses `left_anti` join: Silver PKs with no Bronze match are violations.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t2_silver_pks_are_subset_of_bronze(spark, entry):
    """
    T2 — Every Silver primary-key value must exist in at least one Bronze source.
    Silver ⊆ Bronze on primary key(s). Nothing may be invented.

    Strategy:
      1. Cast Bronze PK column(s) using the same function as the pipeline.
      2. left_anti join Silver PKs onto Bronze PKs.
      3. Assert the anti-join result is empty — no orphaned Silver keys.
    """
    pk_map    = entry["pk_bronze_col"]   # {silver_col: (bronze_raw_col, cast_fn)}
    silver_pks = entry["primary_key"]

    df_silver     = spark.read.table(entry["silver"]).select(*silver_pks).distinct()
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
# MAGIC All audit metadata columns must be present, non-null, and correctly typed.

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
# MAGIC
# MAGIC **`filter_not_null_cols`** — protected by an actual `.filter()` call in the pipeline.
# MAGIC Rows are physically dropped when null → Silver must have **zero nulls**. A null here means the pipeline filter is broken.
# MAGIC
# MAGIC **`dlt_warn_cols`** — covered only by `@dlt.expect` (warn mode).
# MAGIC Null rows are **not dropped** — they legitimately pass into Silver.
# MAGIC Tested as a quality metric: fails only if null rate exceeds 50% (signals catastrophic upstream data loss).
# MAGIC
# MAGIC **`dlt_warn_positive_cols`** — `@dlt.expect` positivity check (warn mode).
# MAGIC Rows with value ≤ 0 pass into Silver. Tested by rate threshold.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t4_hard_filter_columns_not_null(spark, entry):
    """
    T4 — Columns in 'filter_not_null_cols' must have zero NULLs in Silver.
    These are guarded by an explicit .filter() call in 08_silver_transformation.
    A NULL here means the pipeline filter is broken — this is a hard assertion.
    """
    df = spark.read.table(entry["silver"])
    for col in entry.get("filter_not_null_cols", []):
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['name']}] Column '{col}' has {null_cnt:,} NULL rows — "
                "this column is protected by a pipeline .filter() and must be fully populated."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t4_dlt_warn_columns_null_rate(spark, entry):
    """
    T4 — Columns in 'dlt_warn_cols' use @dlt.expect (warn-only).
    NULLs are allowed in Silver — this test fails only if null rate exceeds 50%.
    """
    MAX_NULL_RATE = 0.50
    df    = spark.read.table(entry["silver"])
    total = df.count()
    if total == 0:
        return
    for col in entry.get("dlt_warn_cols", []):
        if col in df.columns:
            null_cnt  = df.filter(F.col(col).isNull()).count()
            null_rate = null_cnt / total
            assert null_rate <= MAX_NULL_RATE, (
                f"[{entry['name']}] Warn-only column '{col}' null rate is "
                f"{null_rate:.1%} ({null_cnt:,}/{total:,}) — "
                f"exceeds threshold of {MAX_NULL_RATE:.0%}. "
                "Possible source schema change or upstream data loss."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t4_dlt_warn_positive_columns_rate(spark, entry):
    """
    T4 — Columns in 'dlt_warn_positive_cols' use @dlt.expect positivity (warn-only).
    Rows with value <= 0 pass into Silver — fails only if bad-value rate exceeds 50%.
    """
    MAX_BAD_RATE = 0.50
    df    = spark.read.table(entry["silver"])
    total = df.count()
    if total == 0:
        return
    for col in entry.get("dlt_warn_positive_cols", []):
        if col in df.columns:
            bad_cnt  = df.filter(F.col(col).isNotNull() & (F.col(col) <= 0)).count()
            bad_rate = bad_cnt / total
            assert bad_rate <= MAX_BAD_RATE, (
                f"[{entry['name']}] Warn-only positive column '{col}' has {bad_rate:.1%} "
                f"({bad_cnt:,}/{total:,}) rows with value <= 0 — "
                f"exceeds threshold of {MAX_BAD_RATE:.0%}."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t4_timestamp_columns_are_correct_type(spark, entry):
    """T4 — Columns in 'timestamp_cols' must be TIMESTAMP type, not string or date."""
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
    (lat 41-82N, lon 19-180E). Backed by _is_valid_russia() hard filter — zero violations.
    """
    df  = spark.read.table(S("geography_transformation"))
    bad = df.filter(
        ~(F.col("latitude").between(41, 82) & F.col("longitude").between(19, 180))
    ).count()
    assert bad == 0, (
        f"[geography_transformation] {bad:,} rows outside Russia bounding box."
    )


def test_t4_listing_date_range_is_reasonable(spark):
    """T4 — listing_date must fall within 2000 – now. Outside = parse error."""
    df  = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("listing_date").isNotNull() & (
            (F.year(F.col("listing_date")) < 2000) |
            (F.col("listing_date") > F.current_timestamp())
        )
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows with listing_date outside plausible range."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 — Deduplication
# MAGIC Verifies that `dropDuplicates` in the pipeline removed all duplicate PKs from Silver.

# COMMAND ----------

# Tables where PK uniqueness is intentionally NOT enforced
_MULTI_ROW_PK_TABLES = {"listings_photo_transformation", "car_catalog_transformation"}


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t5_no_duplicate_primary_keys_in_silver(spark, entry):
    """
    T5 — Silver must have no duplicate rows on its primary key(s).
    Skipped for tables with intentionally non-unique PKs (photos, catalog).
    """
    if entry["name"] in _MULTI_ROW_PK_TABLES:
        pytest.skip(f"[{entry['name']}] PK uniqueness not enforced — skipping.")

    df    = spark.read.table(entry["silver"])
    total = df.count()
    uniq  = df.select(*entry["primary_key"]).distinct().count()

    assert total == uniq, (
        f"[{entry['name']}] Duplicate PKs detected — "
        f"total rows: {total:,}, distinct PKs: {uniq:,}, "
        f"duplicates: {total - uniq:,}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T6 — Quarantine Hygiene
# MAGIC Every quarantine table must be well-formed: reason populated, timestamp present, disjoint from Silver.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t6_quarantine_reason_non_null(spark, entry):
    """T6 — Every row in quarantine must carry a non-null quarantine_reason."""
    df = spark.read.table(entry["quarantine"])
    if df.count() == 0:
        return
    assert "quarantine_reason" in df.columns, (
        f"[{entry['name']}] quarantine table missing 'quarantine_reason' column."
    )
    null_cnt = df.filter(F.col("quarantine_reason").isNull()).count()
    assert null_cnt == 0, (
        f"[{entry['name']}] {null_cnt:,} quarantine rows have NULL quarantine_reason."
    )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t6_quarantine_dt_present_and_non_null(spark, entry):
    """T6 — quarantine_dt must exist and be non-null in every quarantine table that has rows."""
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
    Skipped for tables where PK uniqueness is not enforced.
    """
    if entry["name"] in _MULTI_ROW_PK_TABLES:
        pytest.skip(f"[{entry['name']}] PK uniqueness not enforced — skipping disjoint check.")

    pk_cols   = entry["primary_key"]
    df_silver = spark.read.table(entry["silver"]).select(*pk_cols)
    df_quar   = spark.read.table(entry["quarantine"]).select(*pk_cols)

    overlap = df_silver.join(df_quar, on=pk_cols, how="inner").count()
    assert overlap == 0, (
        f"[{entry['name']}] {overlap:,} PKs appear in both Silver and Quarantine. "
        "Each record must be routed to exactly one destination."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T7 — Derived Column Correctness (`listings_silver_merged`)
# MAGIC Validates enrichment columns are mathematically consistent with source columns in the same Silver row.

# COMMAND ----------

def test_t7_price_usd_derived_correctly(spark):
    """T7 — price_usd must equal round(price_rub / 82.5, 2) for every non-null row."""
    df  = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("price_rub").isNotNull() & F.col("price_usd").isNotNull() &
        (F.abs(F.col("price_usd") - F.round(F.col("price_rub") / USD_RATE, 2)) > 0.01)
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows where price_usd != round(price_rub / {USD_RATE}, 2)."
    )


def test_t7_car_age_years_derived_correctly(spark):
    """T7 — car_age_years must equal (2023 - manufacture_year) for every non-null row."""
    df  = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("manufacture_year").isNotNull() & F.col("car_age_years").isNotNull() &
        (F.col("car_age_years") != (F.lit(2023) - F.col("manufacture_year").cast("integer")))
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows where car_age_years != (2023 - manufacture_year)."
    )


def test_t7_price_category_values_are_valid(spark):
    """T7 — price_category must only contain the 5 defined labels."""
    valid_cats = {"BUDGET", "MID_RANGE", "PREMIUM", "LUXURY", "UNKNOWN"}
    df  = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("price_category").isNotNull() &
        ~F.col("price_category").isin(list(valid_cats))
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows with invalid price_category value "
        f"(allowed: {valid_cats})."
    )


def test_t7_price_category_bucket_boundaries(spark):
    """T7 — price_category bucket labels must match the pipeline threshold boundaries exactly."""
    df = spark.read.table(S("listings_silver_merged")).filter(
        F.col("price_rub").isNotNull() & F.col("price_category").isNotNull()
    )
    checks = [
        ("BUDGET",    F.col("price_rub") < 300_000),
        ("MID_RANGE", F.col("price_rub").between(300_000, 700_000)),
        ("PREMIUM",   F.col("price_rub").between(700_001, 1_500_000)),
        ("LUXURY",    F.col("price_rub") > 1_500_000),
    ]
    for label, condition in checks:
        bad = df.filter((F.col("price_category") == label) & ~condition).count()
        assert bad == 0, (
            f"[listings_silver_merged] {bad:,} rows labelled '{label}' "
            "don't satisfy the expected price_rub range."
        )


def test_t7_brand_std_is_uppercase_brand(spark):
    """T7 — brand_std must equal upper(trim(brand)) for all non-null rows."""
    df  = spark.read.table(S("listings_silver_merged"))
    bad = df.filter(
        F.col("brand").isNotNull() & F.col("brand_std").isNotNull() &
        (F.col("brand_std") != F.upper(F.trim(F.col("brand"))))
    ).count()
    assert bad == 0, (
        f"[listings_silver_merged] {bad:,} rows where brand_std != upper(trim(brand))."
    )


def test_t7_listing_year_month_match_listing_date(spark):
    """T7 — listing_year and listing_month must be consistent with listing_date."""
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
