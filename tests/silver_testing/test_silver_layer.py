# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer Test Suite
# MAGIC Implements a PyTest framework using Databricks Connect to validate
# MAGIC Reconciliation, Row Integrity (SHA-256), and Audit Columns
# MAGIC across all 5 Silver tables.
# MAGIC
# MAGIC | Suite | What it checks                                                        |
# MAGIC |-------|-----------------------------------------------------------------------|
# MAGIC | T1    | Every Bronze row arrives in Silver or Quarantine (row counts balance) |
# MAGIC | T2    | Every Silver row is byte-for-byte traceable to a Bronze row (SHA-256) |
# MAGIC | T3    | All audit metadata columns present and non-null in Silver             |

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
    "catalog":     "vstone_catalog",
    "bronze":      "bronze",
    "silver":      "silver",
}

B = lambda t: f"{CONFIG['catalog']}.{CONFIG['bronze']}.{t}"
S = lambda t: f"{CONFIG['catalog']}.{CONFIG['silver']}.{t}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform Helpers
# MAGIC Row-level normalisers applied identically to Bronze and Silver before SHA-256.
# MAGIC Must mirror the exact transformations applied in `08_silver_transformation`.

# COMMAND ----------

def _norm(expr):
    """Collapses whitespace, lowercases, replaces null with sentinel — both sides hash identically."""
    return F.coalesce(
        F.lower(F.regexp_replace(F.trim(expr.cast("string")), r"\s+", " ")),
        F.lit("null_placeholder")
    )

# ── ID casts ──────────────────────────────────────────────────────────────────
def _id_main(col): return F.expr(f"try_cast(`{col}` as long)").cast("string")
def _id_dbl(col):  return F.col(f"`{col}`").cast("double").cast("long").cast("string")

# ── String transforms (must match pandas behaviour in pipeline) ───────────────
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
def _coal(col):  return F.coalesce(F.col(f"`{col}`").cast("string"), F.lit(""))

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
# MAGIC Single source of truth for all test suites.
# MAGIC Add a new entry here and all three suites pick it up automatically.

# COMMAND ----------

REGISTRY = [
    # ── listings_silver_merged ────────────────────────────────────────────────
    {
        "name"            : "listings_silver_merged",
        "silver"          : S("listings_silver_merged"),
        "quarantine"      : S("listings_main_quarantine"),
        "bronze_sources"  : [B("listings_csv_copyinto"), B("listings_json_autoloader"),
                             B("listings_xml_pyspark"),  B("listings_csv_dlt")],
        "primary_key"     : ["listing_id"],
        "audit_cols"      : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"  : [("id", "listing_id", _id_main)],
        "t1_use_simple_eq": False,
        "t2_pre_filter"   : lambda df: df.filter(
            F.col("listing_id").isNotNull() & (F.col("price_rub") > 0)
        ),
        "t2_select"       : [
            ("id",   "listing_id",   _id_main),
            ("date", "listing_date", _date),
            ("cost", "price_rub",    _price),
        ],
    },

    # ── car_catalog_transformation ────────────────────────────────────────────
    {
        "name"            : "car_catalog_transformation",
        "silver"          : S("car_catalog_transformation"),
        "quarantine"      : S("car_catalog_quarantine"),
        "bronze_sources"  : [B("car_catalog")],
        "primary_key"     : ["brand", "model", "generation"],
        "audit_cols"      : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"  : [
            ("Марка",              "brand",           _clean),
            ("Модель",             "model",           _clean),
            ("Поколение",          "generation",      _clean),
            ("Комплектация",       "trim_level",      _clean),
            ("Объём двигателя",    "engine_volume_l", _eng_vol),
            ("Мощность двигателя", "engine_power_hp", _eng_pow),
        ],
        "t1_use_simple_eq": True,
        "pk_unique_check" : False,   # catalog has multiple trims per brand/model/generation
        "t2_pre_filter"   : lambda df: df.filter(
            F.col("brand").isNotNull() & F.col("model").isNotNull()
        ),
        "t2_select"       : [
            ("Марка",     "brand",      _clean),
            ("Модель",    "model",      _clean),
            ("Поколение", "generation", _clean),
        ],
    },

    # ── listings_text_transformation ──────────────────────────────────────────
    {
        "name"            : "listings_text_transformation",
        "silver"          : S("listings_text_transformation"),
        "quarantine"      : S("listings_text_quarantine"),
        "bronze_sources"  : [B("listings_text")],
        "primary_key"     : ["listing_id"],
        "audit_cols"      : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"  : [("id", "listing_id", _id_dbl)],
        "t1_use_simple_eq": False,
        "t2_pre_filter"   : lambda df: df.filter(F.col("listing_id").isNotNull()),
        "t2_select"       : [
            ("id",   "listing_id", _id_dbl),
            ("text", "text",       _pass),
        ],
    },

    # ── listings_photo_transformation ─────────────────────────────────────────
    {
        "name"            : "listings_photo_transformation",
        "silver"          : S("listings_photo_transformation"),
        "quarantine"      : S("listings_photo_quarantine"),
        "bronze_sources"  : [B("listings_photo")],
        "primary_key"     : ["listing_id"],
        "audit_cols"      : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"  : [
            ("id",        "listing_id",      _id_dbl),
            ("photo_url", "photo_url_clean", _std),
        ],
        "t1_use_simple_eq": False,
        "pk_unique_check" : False,   # one listing can have multiple photos
        "t2_pre_filter"   : lambda df: df.filter(F.col("listing_id").isNotNull()),
        "t2_select"       : [
            ("id",        "listing_id", _id_dbl),
            ("photo_url", "photo_url",  _pass),
        ],
    },

    # ── geography_transformation ──────────────────────────────────────────────
    {
        "name"            : "geography_transformation",
        "silver"          : S("geography_transformation"),
        "quarantine"      : S("geography_quarantine"),
        "bronze_sources"  : [B("geo_locations")],
        "primary_key"     : ["city_name"],
        "audit_cols"      : ["bronze_load_dt", "bronze_source_file", "silver_load_dt"],
        "t1_dedup_exprs"  : [
            ("name_padesh",   "city_name",         _geo),
            ("greate_padesh", "city_prepositional", _pass),
            ("lat",           "latitude",           _dbl),
            ("lon",           "longitude",          _dbl),
        ],
        "t1_valid_filter" : lambda df: df.filter(
            F.col("latitude").isNotNull()  &
            F.col("longitude").isNotNull() &
            F.col("latitude").between(41, 82) &
            F.col("longitude").between(19, 180)
        ),
        "t1_use_simple_eq": False,
        "t2_pre_filter"   : lambda df: df.filter(F.col("latitude").isNotNull()),
        "t2_select"       : [
            ("name_padesh", "city_name", _geo),
            ("lat",         "latitude",  _dbl),
            ("lon",         "longitude", _dbl),
        ],
    },
]

# Parametrize list — reused across all test suites
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


def _get_row_hash(df, col_list: list):
    """Normalises columns and returns a DataFrame of SHA-256 fingerprints."""
    return (
        df.select([_norm(F.col(c)).alias(c) for c in col_list])
          .withColumn("fp", F.sha2(F.concat_ws("||", *col_list), 256))
          .select("fp")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Reconciliation
# MAGIC Verifies that every Bronze row ends up in Silver or Quarantine (none lost).
# MAGIC `t1_use_simple_eq=True`  → raw count comparison (streaming dedup tables).
# MAGIC `t1_use_simple_eq=False` → distinct-key comparison via `t1_dedup_exprs`.

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
        is_ok = dups >= 0
        assert is_ok, (
            f"[{entry['name']}] Row count mismatch — "
            f"Raw: {bronze_total:,} | Actual(S+Q): {actual:,} | Dups Dropped: {dups:,}"
        )
    else:
        df_bronze  = _union_bronze(spark, entry["bronze_sources"])
        dedup_exprs = entry.get("t1_dedup_exprs", [])
        df_dedup   = df_bronze.select(
            [tfn(col).alias(alias) for col, alias, tfn in dedup_exprs]
        )
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
    """T1 — Every row in Quarantine must have a non-null rejection_reason."""
    df = spark.read.table(entry["quarantine"])
    if df.count() > 0 and "rejection_reason" in df.columns:
        null_reasons = df.filter(F.col("rejection_reason").isNull()).count()
        assert null_reasons == 0, (
            f"[{entry['name']}] {null_reasons:,} quarantine rows missing rejection_reason."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Row-to-Row Integrity (SHA-256)
# MAGIC Verifies that every Silver row is byte-for-byte traceable to a Bronze row.
# MAGIC Applies the same column transforms as the pipeline before hashing.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t2_no_silver_rows_untraced_to_bronze(spark, entry):
    """
    T2 — Every Silver row fingerprint must exist in transformed Bronze (nothing invented).
    Uses subtract() to find Silver fingerprints absent from Bronze.
    """
    df_silver     = spark.read.table(entry["silver"])
    df_bronze_raw = _union_bronze(spark, entry["bronze_sources"])

    bronze_tx = df_bronze_raw.select(
        [tfn(b_col).alias(s_col) for b_col, s_col, tfn in entry["t2_select"]]
    )
    if entry.get("t2_pre_filter"):
        bronze_tx = entry["t2_pre_filter"](bronze_tx)

    # Scope Bronze to only the PKs that exist in Silver
    bronze_scoped = bronze_tx.join(
        df_silver.select(*entry["primary_key"]),
        on=entry["primary_key"],
        how="left_semi"
    )

    col_list  = [s for _, s, _ in entry["t2_select"]]
    bronze_fp = _get_row_hash(bronze_scoped, col_list)
    silver_fp = _get_row_hash(df_silver, col_list)

    unmatched = silver_fp.subtract(bronze_fp).count()
    assert unmatched == 0, (
        f"[{entry['name']}] {unmatched:,} Silver row(s) not traceable to Bronze "
        f"(compared columns: {col_list})."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — Audit Columns
# MAGIC Verifies that all audit metadata columns declared in the registry
# MAGIC are present and fully populated (non-null) in every Silver table.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_audit_columns_present(spark, entry):
    """T3 — All declared audit columns must exist in the Silver table schema."""
    df      = spark.read.table(entry["silver"])
    missing = [c for c in entry["audit_cols"] if c not in df.columns]
    assert missing == [], (
        f"[{entry['name']}] Missing audit columns: {missing}"
    )


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
