"""
test_bronze_layer.py — Bronze Layer Test Suite
===============================================
Implements a PyTest framework using Databricks Connect to validate
Volume & Completeness, Row Integrity (SHA-256), and Schema & Metadata
across all 8 Bronze tables.

| Suite | What it checks                                   |
|-------|--------------------------------------------------|
| T1    | Row count: source file vs Bronze table           |
| T2    | SHA-256 fingerprint match: every row, every col  |
| T3    | Audit columns, _rescued_data, data types         |
"""

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F


# =========================
# FIXTURES & CONFIG
# =========================

@pytest.fixture(scope="session")
def spark():
    """Initializes the Databricks Connect session for the test suite."""
    return DatabricksSession.builder.getOrCreate()


CONFIG = {
    "catalog": "vstone_catalog",
    "bronze":  "bronze",
    "raw":     "raw",
}

CHUNKS_PATH  = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/chunks"
LANDING_PATH = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/landing"

# Registry — single source of truth for all test suites.
# Each entry: name, src, table, fmt, opts, null_exclude_col (optional)
REGISTRY = [
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

# Parametrize list — reused across all test suites
REGISTRY_PARAMS = [pytest.param(e, id=e["table"]) for e in REGISTRY]


# =========================
# HELPER FUNCTIONS
# =========================

def get_table_path(table_name: str) -> str:
    """Returns the fully-qualified Bronze table name."""
    return f"{CONFIG['catalog']}.{CONFIG['bronze']}.{table_name}"


def read_source(spark, entry: dict):
    """
    Reads source file from Volume with registry options.
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
        return df, f"{null_count} null-{excl_col} source rows excluded (bad data in source)"
    return df, None


def get_row_hash(df, columns):
    """
    Normalises each column to trimmed STRING, replaces NULLs with '',
    then hashes all columns together into a SHA-256 fingerprint per row.
    """
    normalised = df.select([
        F.coalesce(F.trim(F.col(c).cast("string")), F.lit("")).alias(c)
        for c in columns
    ])
    return (
        normalised
        .withColumn("row_hash", F.sha2(F.concat_ws("||", *columns), 256))
        .select("row_hash")
    )


# =========================
# T1 — VOLUME & COMPLETENESS
# =========================

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t1_source_count_equals_bronze_count(spark, entry):
    """
    T1 — Row count in source file must exactly match Bronze table row count.
    Source rows with known-bad data (null_exclude_col) are excluded before comparison.
    """
    df_src, excl_note = read_source(spark, entry)
    src_count    = df_src.count()
    bronze_count = spark.read.table(get_table_path(entry["table"])).count()
    gap          = src_count - bronze_count

    assert gap == 0, (
        f"[{entry['table']}] Row count mismatch — "
        f"Source: {src_count:,} | Bronze: {bronze_count:,} | GAP: {gap:,}"
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t1_bronze_table_non_empty(spark, entry):
    """T1 — Every Bronze table must contain at least 1 row."""
    count = spark.read.table(get_table_path(entry["table"])).count()
    assert count > 0, f"[{entry['table']}] Bronze table is empty."


def test_t1_chunk1_no_null_id_rows(spark):
    """
    T1 — Chunk 1 source has 2 null-id rows that must be excluded from Bronze.
    Bronze listings_csv_copyinto must contain zero rows where id IS NULL.
    """
    null_id_rows = (
        spark.read.table(get_table_path("listings_csv_copyinto"))
        .filter(F.col("id").isNull())
        .count()
    )
    assert null_id_rows == 0, (
        f"listings_csv_copyinto contains {null_id_rows} null-id rows — "
        "these should have been excluded during ingestion."
    )


# =========================
# T2 — ROW-TO-ROW INTEGRITY (SHA-256)
# =========================

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t2_no_source_rows_missing_from_bronze(spark, entry):
    """
    T2 — Every source row fingerprint must exist in Bronze (nothing lost).
    Uses subtract() to find fingerprints present in source but absent from Bronze.
    """
    df_src, excl_note = read_source(spark, entry)
    df_brz      = spark.read.table(get_table_path(entry["table"]))
    common_cols = [c for c in df_src.columns if c in df_brz.columns]

    missing = get_row_hash(df_src, common_cols).subtract(
              get_row_hash(df_brz, common_cols)).count()

    assert missing == 0, (
        f"[{entry['table']}] {missing} source row(s) not found in Bronze "
        f"(compared {len(common_cols)} columns: {common_cols})"
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t2_no_extra_rows_invented_in_bronze(spark, entry):
    """
    T2 — No Bronze row fingerprint may be absent from source (Bronze must not invent rows).
    Uses subtract() to find fingerprints present in Bronze but absent from source.
    """
    df_src, excl_note = read_source(spark, entry)
    df_brz      = spark.read.table(get_table_path(entry["table"]))
    common_cols = [c for c in df_src.columns if c in df_brz.columns]

    extra = get_row_hash(df_brz, common_cols).subtract(
            get_row_hash(df_src, common_cols)).count()

    assert extra == 0, (
        f"[{entry['table']}] {extra} Bronze row(s) not traceable to source "
        f"(compared {len(common_cols)} columns)"
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )


# =========================
# T3 — SCHEMA & METADATA
# =========================

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_audit_columns_present(spark, entry):
    """T3 — load_dt and source_file must exist in every Bronze table schema."""
    df      = spark.read.table(get_table_path(entry["table"]))
    missing = [c for c in ("load_dt", "source_file") if c not in df.columns]
    assert missing == [], (
        f"[{entry['table']}] Missing audit columns: {missing}"
    )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_audit_columns_non_null(spark, entry):
    """T3 — load_dt and source_file must have zero null values in every Bronze table."""
    df = spark.read.table(get_table_path(entry["table"]))
    for col in ("load_dt", "source_file"):
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['table']}] Column '{col}' has {null_cnt:,} NULL rows."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_load_dt_is_timestamp(spark, entry):
    """T3 — load_dt must be TIMESTAMP type, never string or date."""
    df     = spark.read.table(get_table_path(entry["table"]))
    dtypes = dict(df.dtypes)
    if "load_dt" in dtypes:
        assert dtypes["load_dt"].startswith("timestamp"), (
            f"[{entry['table']}] load_dt is '{dtypes['load_dt']}', expected 'timestamp'."
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_no_rescued_data_pollution(spark, entry):
    """
    T3 — If _rescued_data column is present it must be entirely NULL.
    Non-null _rescued_data indicates schema mismatch during Auto Loader ingestion.
    """
    df = spark.read.table(get_table_path(entry["table"]))
    if "_rescued_data" in df.columns:
        rescued = df.filter(F.col("_rescued_data").isNotNull()).count()
        assert rescued == 0, (
            f"[{entry['table']}] _rescued_data has {rescued:,} non-null rows — "
            "indicates schema mismatch during ingestion."
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_source_file_identifies_origin(spark, entry):
    """T3 — source_file values must correctly identify the ingestion source file."""
    expected_map = {
        "listings_csv_copyinto":    "chunk1.csv",
        "listings_csv_dlt":         "chunk2.csv",
        "listings_json_autoloader": "chunk3.json",
        "listings_xml_pyspark":     "chunk4.xml",
        "listings_text":            "1_text.csv",
        "listings_photo":           "1_photo.csv",
        "car_catalog":              "catalogs.csv",
        "geo_locations":            "final_geografic.csv",
    }
    expected = expected_map[entry["table"]]
    df    = spark.read.table(get_table_path(entry["table"]))
    files = [r["source_file"] for r in df.select("source_file").distinct().collect()]
    assert any(expected in f for f in files), (
        f"[{entry['table']}] Expected source_file containing '{expected}', got {files}."
    )


@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["table"])
    for e in REGISTRY
    if e["table"] in (
        "listings_csv_copyinto", "listings_csv_dlt",
        "listings_json_autoloader", "listings_xml_pyspark",
    )
])
def test_t3_listing_id_is_string_type(spark, entry):
    """
    T3 — id column in listing Bronze tables must be STRING type.
    COPY INTO and Auto Loader load with inferSchema=false, preserving raw types.
    """
    df     = spark.read.table(get_table_path(entry["table"]))
    dtypes = dict(df.dtypes)
    if "id" in dtypes:
        assert dtypes["id"] == "string", (
            f"[{entry['table']}] 'id' should be STRING in Bronze, got '{dtypes['id']}'."
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_source_file_non_empty_string(spark, entry):
    """T3 — source_file must never be an empty string."""
    df = spark.read.table(get_table_path(entry["table"]))
    if "source_file" in df.columns:
        bad = df.filter(F.trim(F.col("source_file")) == "").count()
        assert bad == 0, (
            f"[{entry['table']}] {bad} rows have empty source_file."
        )
