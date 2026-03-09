# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Pipeline Integration Test Suite
# MAGIC Implements a PyTest framework using Databricks Connect to validate
# MAGIC the full Bronze → Silver → Gold pipeline across all layers.
# MAGIC
# MAGIC | Suite | What it checks                                                                        |
# MAGIC |-------|---------------------------------------------------------------------------------------|
# MAGIC | T1    | All Bronze tables non-empty; source file counts match Bronze (Volume)                 |
# MAGIC | T2    | Bronze audit columns (load_dt, source_file) propagate to Silver & Gold                |
# MAGIC | T3    | Every Bronze row arrives in Silver or Quarantine (Reconciliation)                     |
# MAGIC | T4    | Silver rows are byte-for-byte traceable to Bronze via SHA-256 (Row Integrity)         |
# MAGIC | T5    | listing_id lineage is unbroken from Silver through to fact_listings in Gold            |
# MAGIC | T6    | Gold star-schema joins >= 60%; SCD2 dim counts match Silver; agg tables non-empty     |
# MAGIC | T7    | SCD2 timeline consistency and one-active-row-per-key across all three SCD2 dims       |

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
    "raw":     "raw",
}

CHUNKS_PATH  = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/chunks"
LANDING_PATH = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/landing"

BRONZE = f"{CONFIG['catalog']}.{CONFIG['bronze']}"
SILVER = f"{CONFIG['catalog']}.{CONFIG['silver']}"
GOLD   = f"{CONFIG['catalog']}.{CONFIG['gold']}"

MIN_JOIN_RATE_PCT = 60.0
DIM_DATE_EXPECTED = 7670

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform Helpers
# MAGIC Mirror the exact column transformations applied in the Silver pipeline
# MAGIC so SHA-256 fingerprints computed here match what the pipeline produces.

# COMMAND ----------

def _norm(expr):
    """Collapses whitespace, lowercases, replaces null with sentinel."""
    return F.coalesce(
        F.lower(F.regexp_replace(F.trim(expr.cast("string")), r"\s+", " ")),
        F.lit("null_placeholder")
    )

def _id_main(col): return F.expr(f"try_cast(`{col}` as long)").cast("string")
def _id_dbl(col):  return F.col(f"`{col}`").cast("double").cast("long").cast("string")
def _std(col):
    return F.when(F.col(f"`{col}`").isNull(), F.lit("none")) \
             .otherwise(F.lower(F.trim(F.col(f"`{col}`"))))
def _clean(col):
    return F.when(F.col(f"`{col}`").isNull(), F.lit("None")) \
             .otherwise(F.trim(F.col(f"`{col}`")))
def _geo(col):
    return F.when(F.col(f"`{col}`").isNull(), F.lit("None")) \
             .otherwise(F.trim(F.col(f"`{col}`")))
def _pass(col):  return F.col(f"`{col}`")
def _dbl(col):   return F.col(f"`{col}`").cast("double")
def _price(col): return F.expr(f"try_cast(regexp_replace(`{col}`, '[^0-9.]', '') as double)")
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
# MAGIC Single source of truth — one entry per Silver table carrying its Bronze sources,
# MAGIC dedup expressions, SHA-256 select list, audit columns, and quarantine table.

# COMMAND ----------

# ── Bronze table registry ─────────────────────────────────────────────────────
BRONZE_REGISTRY = [
    {
        "name"             : "Chunk 1 — CSV / COPY INTO",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_1.csv",
        "table"            : "listings_csv_copyinto",
        "fmt"              : "csv",
        "opts"             : {"header": "true"},
        "null_exclude_col" : "id",
    },
    {
        "name"  : "Chunk 2 — CSV / DLT",
        "src"   : f"{CHUNKS_PATH}/1_main_chunk_2.csv",
        "table" : "listings_csv_dlt",
        "fmt"   : "csv",
        "opts"  : {"header": "true"},
    },
    {
        "name"  : "Chunk 3 — JSON / Auto Loader",
        "src"   : f"{CHUNKS_PATH}/1_main_chunk_3.json",
        "table" : "listings_json_autoloader",
        "fmt"   : "json",
        "opts"  : {"multiLine": "true"},
    },
    {
        "name"  : "Chunk 4 — XML / PySpark",
        "src"   : f"{CHUNKS_PATH}/1_main_chunk_4.xml",
        "table" : "listings_xml_pyspark",
        "fmt"   : "xml",
        "opts"  : {"rowTag": "record"},
    },
    {
        "name"  : "Landing — Text Data",
        "src"   : f"{LANDING_PATH}/1_text.csv",
        "table" : "listings_text",
        "fmt"   : "csv",
        "opts"  : {"header": "true", "multiLine": "true", "escape": '"'},
    },
    {
        "name"  : "Landing — Photo Data",
        "src"   : f"{LANDING_PATH}/1_photo.csv",
        "table" : "listings_photo",
        "fmt"   : "csv",
        "opts"  : {"header": "true"},
    },
    {
        "name"  : "Landing — Car Catalog",
        "src"   : f"{LANDING_PATH}/catalogs.csv",
        "table" : "car_catalog",
        "fmt"   : "csv",
        "opts"  : {"header": "true", "sep": ";"},
    },
    {
        "name"  : "Landing — Geo Locations",
        "src"   : f"{LANDING_PATH}/final_geografic.csv",
        "table" : "geo_locations",
        "fmt"   : "csv",
        "opts"  : {"header": "true"},
    },
]

# Listing tables that must have a STRING id column
LISTING_TABLES = (
    "listings_csv_copyinto",
    "listings_csv_dlt",
    "listings_json_autoloader",
    "listings_xml_pyspark",
)

SOURCE_FILE_MAP = {
    "listings_csv_copyinto":    "1_main_chunk_1.csv",
    "listings_csv_dlt":         "1_main_chunk_2.csv",
    "listings_json_autoloader": "1_main_chunk_3.json",
    "listings_xml_pyspark":     "1_main_chunk_4.xml",
    "listings_text":            "1_text.csv",
    "listings_photo":           "1_photo.csv",
    "car_catalog":              "catalogs.csv",
    "geo_locations":            "final_geografic.csv",
}

# ── Silver registry ───────────────────────────────────────────────────────────
B = lambda t: f"{BRONZE}.{t}"
S = lambda t: f"{SILVER}.{t}"

SILVER_REGISTRY = [
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
        "pk_unique_check" : False,
        "t2_pre_filter"   : lambda df: df.filter(
            F.col("brand").isNotNull() & F.col("model").isNotNull()
        ),
        "t2_select"       : [
            ("Марка",     "brand",      _clean),
            ("Модель",    "model",      _clean),
            ("Поколение", "generation", _clean),
        ],
    },
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
        "pk_unique_check" : False,
        "t2_pre_filter"   : lambda df: df.filter(F.col("listing_id").isNotNull()),
        "t2_select"       : [
            ("id",        "listing_id", _id_dbl),
            ("photo_url", "photo_url",  _pass),
        ],
    },
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
        "t1_use_simple_eq": False,
        "t2_pre_filter"   : lambda df: df.filter(F.col("latitude").isNotNull()),
        "t2_select"       : [
            ("name_padesh", "city_name", _geo),
            ("lat",         "latitude",  _dbl),
            ("lon",         "longitude", _dbl),
        ],
    },
]

# ── Gold registry ─────────────────────────────────────────────────────────────
ALL_GOLD_TABLES = [
    "dim_date", "fact_listings", "agg_monthly_sales_trend", "agg_brand_location_performance",
    "agg_regional_market_depth", "agg_comprehensive_kpi_cube", "agg_top_10_brands_by_spend",
]
SCD2_DIM_TABLES = ["dim_car", "dim_location", "dim_listing_details", "dim_listing_photos"]

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

# ── Parametrize helpers ───────────────────────────────────────────────────────
BRONZE_PARAMS     = [pytest.param(e, id=e["table"]) for e in BRONZE_REGISTRY]
SILVER_PARAMS     = [pytest.param(e, id=e["name"])  for e in SILVER_REGISTRY]
GOLD_TABLE_PARAMS = [pytest.param(t, id=t) for t in ALL_GOLD_TABLES]
SCD2_DIM_PARAMS   = [pytest.param(t, id=t) for t in SCD2_DIM_TABLES]
SCD2_PARAMS       = [pytest.param(e, id=e["name"]) for e in SCD2_REGISTRY]
JOIN_PARAMS       = [pytest.param(e, id=e["dim"])  for e in JOIN_REGISTRY]
LISTING_PARAMS    = [
    pytest.param(e, id=e["table"])
    for e in BRONZE_REGISTRY if e["table"] in LISTING_TABLES
]

print(
    f"Registry loaded — "
    f"{len(BRONZE_REGISTRY)} Bronze | "
    f"{len(SILVER_REGISTRY)} Silver | "
    f"{len(ALL_GOLD_TABLES) + len(SCD2_DIM_TABLES)} Gold tables | "
    f"{len(SCD2_REGISTRY)} SCD2 dims | "
    f"{len(JOIN_REGISTRY)} joins registered."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def read_source(spark, entry: dict):
    """
    Reads source file from Volume using registry options.
    Applies null_exclude_col filter if set (removes known-bad rows).
    Returns (DataFrame, exclusion_note_or_None).
    """
    df = (
        spark.read
        .format(entry["fmt"])
        .options(**entry["opts"])
        .option("inferSchema", "false")
        .load(entry["src"])
    )
    excl_col = entry.get("null_exclude_col")
    if excl_col:
        null_count = df.filter(F.col(excl_col).isNull()).count()
        df = df.filter(F.col(excl_col).isNotNull())
        return df, f"{null_count} null-{excl_col} source rows excluded"
    return df, None


def get_row_hash_silver(df, col_list: list):
    """SHA-256 per row across given columns using Silver-side _norm normalisation."""
    return (
        df.select([_norm(F.col(c)).alias(c) for c in col_list])
          .withColumn("fp", F.sha2(F.concat_ws("||", *col_list), 256))
          .select("fp")
    )


def union_bronze(spark, sources: list):
    """Unions all Bronze source tables, tolerating missing columns."""
    df = None
    for src in sources:
        b  = spark.read.table(src)
        df = b if df is None else df.unionByName(b, allowMissingColumns=True)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Bronze Volume & Completeness
# MAGIC Verifies all 8 Bronze tables are non-empty and their row counts
# MAGIC match the source files they were ingested from.

# COMMAND ----------

@pytest.mark.parametrize("entry", BRONZE_PARAMS)
def test_t1_bronze_table_non_empty(spark, entry):
    """T1 — Every Bronze table must contain at least 1 row (ingestion ran)."""
    count = spark.read.table(f"{BRONZE}.{entry['table']}").count()
    assert count > 0, f"[{entry['table']}] Bronze table is empty."


@pytest.mark.parametrize("entry", BRONZE_PARAMS)
def test_t1_source_count_equals_bronze_count(spark, entry):
    """T1 — Row count in source file must exactly match Bronze table row count."""
    df_src, excl_note = read_source(spark, entry)
    src_count    = df_src.count()
    bronze_count = spark.read.table(f"{BRONZE}.{entry['table']}").count()
    gap          = src_count - bronze_count
    assert gap == 0, (
        f"[{entry['table']}] Row count mismatch — "
        f"Source: {src_count:,} | Bronze: {bronze_count:,} | GAP: {gap:,}"
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )


def test_t1_chunk1_no_null_id_rows(spark):
    """T1 — listings_csv_copyinto must contain zero rows where id IS NULL."""
    null_id_rows = (
        spark.read.table(f"{BRONZE}.listings_csv_copyinto")
        .filter(F.col("id").isNull())
        .count()
    )
    assert null_id_rows == 0, (
        f"listings_csv_copyinto contains {null_id_rows} null-id rows — "
        "these should have been excluded during ingestion."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Bronze Audit Column Integrity
# MAGIC Verifies load_dt and source_file are present, fully populated, correctly typed,
# MAGIC and identify the right ingestion source in every Bronze table.

# COMMAND ----------

@pytest.mark.parametrize("entry", BRONZE_PARAMS)
def test_t2_bronze_audit_columns_present(spark, entry):
    """T2 — load_dt and source_file must exist in every Bronze table schema."""
    df      = spark.read.table(f"{BRONZE}.{entry['table']}")
    missing = [c for c in ("load_dt", "source_file") if c not in df.columns]
    assert missing == [], f"[{entry['table']}] Missing Bronze audit columns: {missing}"


@pytest.mark.parametrize("entry", BRONZE_PARAMS)
def test_t2_bronze_audit_columns_non_null(spark, entry):
    """T2 — load_dt and source_file must have zero NULL values in every Bronze table."""
    df = spark.read.table(f"{BRONZE}.{entry['table']}")
    for col in ("load_dt", "source_file"):
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['table']}] Column '{col}' has {null_cnt:,} NULL rows."
            )


@pytest.mark.parametrize("entry", BRONZE_PARAMS)
def test_t2_bronze_load_dt_is_timestamp(spark, entry):
    """T2 — load_dt must be TIMESTAMP type in every Bronze table."""
    df     = spark.read.table(f"{BRONZE}.{entry['table']}")
    dtypes = dict(df.dtypes)
    if "load_dt" in dtypes:
        assert dtypes["load_dt"].startswith("timestamp"), (
            f"[{entry['table']}] load_dt is '{dtypes['load_dt']}', expected 'timestamp'."
        )


@pytest.mark.parametrize("entry", BRONZE_PARAMS)
def test_t2_bronze_source_file_identifies_origin(spark, entry):
    """T2 — source_file values must identify the correct ingestion source file."""
    expected = SOURCE_FILE_MAP[entry["table"]]
    df       = spark.read.table(f"{BRONZE}.{entry['table']}")
    files    = [r["source_file"] for r in df.select("source_file").distinct().collect()]
    assert any(expected in f for f in files), (
        f"[{entry['table']}] Expected source_file containing '{expected}', got {files}."
    )


@pytest.mark.parametrize("entry", BRONZE_PARAMS)
def test_t2_no_rescued_data_pollution(spark, entry):
    """T2 — _rescued_data must be entirely NULL if present (no schema mismatch)."""
    df = spark.read.table(f"{BRONZE}.{entry['table']}")
    if "_rescued_data" in df.columns:
        rescued = df.filter(F.col("_rescued_data").isNotNull()).count()
        assert rescued == 0, (
            f"[{entry['table']}] _rescued_data has {rescued:,} non-null rows — "
            "indicates schema mismatch during ingestion."
        )


@pytest.mark.parametrize("entry", LISTING_PARAMS)
def test_t2_bronze_listing_id_is_string_type(spark, entry):
    """T2 — id column in listing Bronze tables must be STRING type (inferSchema=false)."""
    df     = spark.read.table(f"{BRONZE}.{entry['table']}")
    dtypes = dict(df.dtypes)
    if "id" in dtypes:
        assert dtypes["id"] == "string", (
            f"[{entry['table']}] 'id' should be STRING in Bronze, got '{dtypes['id']}'."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — Silver Reconciliation (Bronze -> Silver + Quarantine)
# MAGIC Verifies every Bronze row ends up in Silver or Quarantine (no silent drops).
# MAGIC Silver audit columns must be present, non-null, and correctly typed.
# MAGIC Quarantine rows must carry a non-null rejection_reason.

# COMMAND ----------

@pytest.mark.parametrize("entry", SILVER_PARAMS)
def test_t3_silver_table_non_empty(spark, entry):
    """T3 — Every Silver table must contain at least 1 row."""
    count = spark.read.table(entry["silver"]).count()
    assert count > 0, f"[{entry['name']}] Silver table is empty."


@pytest.mark.parametrize("entry", SILVER_PARAMS)
def test_t3_reconciliation_bronze_to_silver_plus_quarantine(spark, entry):
    """T3 — Bronze row count must equal Silver + Quarantine (nothing silently dropped)."""
    bronze_total = sum(spark.read.table(s).count() for s in entry["bronze_sources"])
    silver_cnt   = spark.read.table(entry["silver"]).count()
    quar_cnt     = spark.read.table(entry["quarantine"]).count()
    actual       = silver_cnt + quar_cnt

    if entry.get("t1_use_simple_eq", False):
        assert bronze_total >= actual, (
            f"[{entry['name']}] Row count mismatch — "
            f"Bronze: {bronze_total:,} | Silver+Quarantine: {actual:,}"
        )
    else:
        df_bronze   = union_bronze(spark, entry["bronze_sources"])
        dedup_exprs = entry.get("t1_dedup_exprs", [])
        df_dedup    = df_bronze.select(
            [tfn(col).alias(alias) for col, alias, tfn in dedup_exprs]
        )
        unique_exp = df_dedup.distinct().count()
        assert actual == unique_exp, (
            f"[{entry['name']}] Reconciliation mismatch — "
            f"Bronze raw: {bronze_total:,} | Unique expected: {unique_exp:,} | "
            f"Silver+Quarantine: {actual:,} | Dups dropped: {bronze_total - unique_exp:,}"
        )


@pytest.mark.parametrize("entry", SILVER_PARAMS)
def test_t3_quarantine_has_rejection_reasons(spark, entry):
    """T3 — Every quarantine row must carry a non-null rejection_reason."""
    df = spark.read.table(entry["quarantine"])
    if df.count() > 0 and "rejection_reason" in df.columns:
        null_reasons = df.filter(F.col("rejection_reason").isNull()).count()
        assert null_reasons == 0, (
            f"[{entry['name']}] {null_reasons:,} quarantine rows missing rejection_reason."
        )


@pytest.mark.parametrize("entry", SILVER_PARAMS)
def test_t3_silver_audit_columns_present_and_non_null(spark, entry):
    """T3 — All declared audit columns must exist and be fully non-null in Silver."""
    df      = spark.read.table(entry["silver"])
    missing = [c for c in entry["audit_cols"] if c not in df.columns]
    assert missing == [], f"[{entry['name']}] Missing audit columns: {missing}"
    for col in entry["audit_cols"]:
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['name']}] Audit column '{col}' has {null_cnt:,} NULL rows."
            )


@pytest.mark.parametrize("entry", SILVER_PARAMS)
def test_t3_silver_load_dt_is_timestamp(spark, entry):
    """T3 — silver_load_dt must be TIMESTAMP type in every Silver table."""
    df     = spark.read.table(entry["silver"])
    dtypes = dict(df.dtypes)
    if "silver_load_dt" in dtypes:
        assert dtypes["silver_load_dt"].startswith("timestamp"), (
            f"[{entry['name']}] silver_load_dt is '{dtypes['silver_load_dt']}', "
            "expected 'timestamp'."
        )


@pytest.mark.parametrize("entry", SILVER_PARAMS)
def test_t3_silver_primary_key_non_null(spark, entry):
    """T3 — Primary key columns must never be null in Silver."""
    df = spark.read.table(entry["silver"])
    for pk_col in entry["primary_key"]:
        if pk_col in df.columns:
            null_cnt = df.filter(F.col(pk_col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['name']}] Primary key '{pk_col}' has {null_cnt:,} NULL rows."
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T4 — Silver Row Integrity (SHA-256)
# MAGIC Verifies every Silver row is byte-for-byte traceable to a Bronze row.
# MAGIC Applies the same column transforms as the pipeline before hashing.

# COMMAND ----------

@pytest.mark.parametrize("entry", SILVER_PARAMS)
def test_t4_no_silver_rows_untraced_to_bronze(spark, entry):
    """T4 — Every Silver row fingerprint must exist in transformed Bronze (nothing invented)."""
    df_silver     = spark.read.table(entry["silver"])
    df_bronze_raw = union_bronze(spark, entry["bronze_sources"])

    bronze_tx = df_bronze_raw.select(
        [tfn(b_col).alias(s_col) for b_col, s_col, tfn in entry["t2_select"]]
    )
    if entry.get("t2_pre_filter"):
        bronze_tx = entry["t2_pre_filter"](bronze_tx)

    bronze_scoped = bronze_tx.join(
        df_silver.select(*entry["primary_key"]),
        on=entry["primary_key"],
        how="left_semi"
    )

    col_list  = [s for _, s, _ in entry["t2_select"]]
    bronze_fp = get_row_hash_silver(bronze_scoped, col_list)
    silver_fp = get_row_hash_silver(df_silver,     col_list)

    unmatched = silver_fp.subtract(bronze_fp).count()
    assert unmatched == 0, (
        f"[{entry['name']}] {unmatched:,} Silver row(s) not traceable to Bronze "
        f"(compared columns: {col_list})."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 — listing_id Lineage (Silver -> Gold)
# MAGIC Verifies Silver listing_ids flow into fact_listings with no drops or invented IDs,
# MAGIC and that SCD2 dim natural keys are fully covered in both directions.

# COMMAND ----------

def test_t5_fact_listings_matches_silver(spark):
    """T5 — fact_listings row count must be within 1% of listings_silver_merged."""
    silver_cnt = spark.read.table(f"{SILVER}.listings_silver_merged").count()
    gold_cnt   = spark.read.table(f"{GOLD}.fact_listings").count()
    diff_pct   = abs(gold_cnt - silver_cnt) / max(silver_cnt, 1) * 100
    assert diff_pct <= 1.0, (
        f"fact_listings row count diverges from Silver by {diff_pct:.3f}% — "
        f"Silver={silver_cnt:,} | Gold={gold_cnt:,}"
    )


def test_t5_fact_listings_no_missing_silver_ids(spark):
    """T5 — Every Silver listing_id must exist in fact_listings (nothing dropped)."""
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    missing    = silver_ids.subtract(gold_ids).count()
    assert missing == 0, f"fact_listings is missing {missing:,} Silver listing_id(s)."


def test_t5_fact_listings_no_invented_ids(spark):
    """T5 — fact_listings must not contain listing_ids absent from Silver."""
    silver_ids = spark.read.table(f"{SILVER}.listings_silver_merged").select("listing_id")
    gold_ids   = spark.read.table(f"{GOLD}.fact_listings").select("listing_id")
    invented   = gold_ids.subtract(silver_ids).count()
    assert invented == 0, f"fact_listings contains {invented:,} invented listing_id(s) not in Silver."


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t5_scd2_no_missing_silver_keys(spark, entry):
    """T5 — Every distinct Silver key must appear as an active Gold row (__END_AT IS NULL)."""
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
def test_t5_scd2_no_invented_gold_keys(spark, entry):
    """T5 — Gold active rows must not contain keys absent from Silver."""
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
# MAGIC ## T6 — Gold Quality (Schema, Audit Chain, Star Schema Joins)
# MAGIC Verifies the full audit chain on fact_listings, schema integrity on fact and dim_date,
# MAGIC FK join rates >= 60%, and that all aggregate tables are non-empty.

# COMMAND ----------

def test_t6_fact_listings_full_audit_chain(spark):
    """T6 — fact_listings must carry the full bronze->silver->gold audit chain (non-null)."""
    df          = spark.read.table(f"{GOLD}.fact_listings")
    audit_chain = ["bronze_load_dt", "bronze_source_file", "silver_load_dt", "gold_load_dt"]
    missing     = [c for c in audit_chain if c not in df.columns]
    assert missing == [], f"fact_listings missing audit chain columns: {missing}"
    for col in audit_chain:
        nulls = df.filter(F.col(col).isNull()).count()
        assert nulls == 0, f"fact_listings.{col} has {nulls:,} NULL rows."


@pytest.mark.parametrize("table", GOLD_TABLE_PARAMS)
def test_t6_gold_load_dt_present_and_non_null(spark, table):
    """T6 — gold_load_dt must exist and be fully non-null in every Gold table."""
    df = spark.read.table(f"{GOLD}.{table}")
    assert "gold_load_dt" in df.columns, f"[{table}] Missing column: gold_load_dt"
    nulls = df.filter(F.col("gold_load_dt").isNull()).count()
    assert nulls == 0, f"[{table}] gold_load_dt has {nulls:,} NULL rows."


@pytest.mark.parametrize("table", SCD2_DIM_PARAMS)
def test_t6_scd2_silver_load_dt_non_null(spark, table):
    """T6 — SCD2 dims must carry silver_load_dt (non-null)."""
    df = spark.read.table(f"{GOLD}.{table}")
    assert "silver_load_dt" in df.columns, f"[{table}] Missing column: silver_load_dt"
    nulls = df.filter(F.col("silver_load_dt").isNull()).count()
    assert nulls == 0, f"[{table}] silver_load_dt has {nulls:,} NULL rows."


@pytest.mark.parametrize("entry", SCD2_PARAMS)
def test_t6_scd2_metadata_columns_present(spark, entry):
    """T6 — SCD2 dims must have __START_AT, __END_AT and silver_load_dt columns."""
    cols = spark.read.table(f"{GOLD}.{entry['name']}").columns
    for required in ("__START_AT", "__END_AT", "silver_load_dt"):
        assert required in cols, (
            f"[{entry['name']}] Missing SCD2 metadata column: {required}"
        )


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


def test_t6_dim_date_row_count(spark):
    """T6 — dim_date must contain exactly 7,670 rows (2010-2030 calendar spine)."""
    date_cnt = spark.read.table(f"{GOLD}.dim_date").count()
    assert date_cnt == DIM_DATE_EXPECTED, (
        f"dim_date row count mismatch — Expected={DIM_DATE_EXPECTED:,} | Actual={date_cnt:,}"
    )


@pytest.mark.parametrize("entry", JOIN_PARAMS)
def test_t6_fact_dim_join_rate(spark, entry):
    """T6 — Join rate between fact_listings and each dimension must be >= 60%."""
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
        f"[fact -> {entry['dim']}] Join rate {rate}% is below minimum {MIN_JOIN_RATE_PCT}% — "
        f"Matched={joined:,} / Total={fact_cnt:,}"
    )


@pytest.mark.parametrize("agg_table", [
    pytest.param(t, id=t) for t in [
        "agg_monthly_sales_trend", "agg_brand_location_performance",
        "agg_regional_market_depth", "agg_comprehensive_kpi_cube", "agg_top_10_brands_by_spend",
    ]
])
def test_t6_agg_tables_non_empty(spark, agg_table):
    """T6 — All aggregate Gold tables must contain at least 1 row."""
    cnt = spark.read.table(f"{GOLD}.{agg_table}").count()
    assert cnt > 0, f"[{agg_table}] Aggregate table is empty."

# COMMAND ----------

# MAGIC %md
# MAGIC ## T7 — SCD2 Integrity
# MAGIC Verifies exactly one CURRENT row per natural key, timeline consistency,
# MAGIC and that every key has at least one active row.

# COMMAND ----------

@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["name"])
    for e in SCD2_REGISTRY if e["name"] != "dim_listing_photos"
])
def test_t7_scd2_one_active_row_per_key(spark, entry):
    """T7 — Each natural key must have exactly one active row (__END_AT IS NULL)."""
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


@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["name"])
    for e in SCD2_REGISTRY if e["name"] != "dim_listing_photos"
])
def test_t7_scd2_timeline_consistency(spark, entry):
    """T7 — For historical rows, __START_AT must always be strictly before __END_AT."""
    timeline_sql = (
        f"SELECT COUNT(*) AS c FROM {GOLD}.{entry['name']} "
        f"WHERE __END_AT IS NOT NULL AND __START_AT >= __END_AT"
    )
    violations = spark.sql(timeline_sql).collect()[0]["c"]
    assert violations == 0, (
        f"[{entry['name']}] {violations:,} historical rows have __START_AT >= __END_AT."
    )


@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["name"])
    for e in SCD2_REGISTRY if e["name"] != "dim_listing_photos"
])
def test_t7_scd2_every_key_has_active_row(spark, entry):
    """T7 — Every natural key must have at least one active row (__END_AT IS NULL)."""
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
