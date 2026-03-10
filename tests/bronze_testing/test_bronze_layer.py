# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Test Suite
# MAGIC Implements a PyTest framework to validate Volume & Completeness, Row Integrity (SHA-256), and Schema & Metadata across all 8 Bronze tables.
# MAGIC
# MAGIC | Suite | What it checks                                  |
# MAGIC |-------|-------------------------------------------------|
# MAGIC | T1    | Row count: source file vs Bronze table          |
# MAGIC | T2    | SHA-256 fingerprint match: every row, every col |
# MAGIC | T3    | Audit columns, _rescued_data, data types        |

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
    "raw":     "raw",
}

CHUNKS_PATH  = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/chunks"
LANDING_PATH = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/landing"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registry
# MAGIC Single source of truth for all test suites.
# MAGIC Add a new entry here and all three suites pick it up automatically.

# COMMAND ----------

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

# Listing tables that must have a STRING id column
LISTING_TABLES = (
    "listings_csv_copyinto",
    "listings_csv_dlt",
    "listings_json_autoloader",
    "listings_xml_pyspark",
)

# Expected source filename fragment per table
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

print(f"Registry loaded — {len(REGISTRY)} tables registered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def get_table_path(table_name: str) -> str:
    """Returns the fully-qualified Bronze table name."""
    return f"{CONFIG['catalog']}.{CONFIG['bronze']}.{table_name}"


def read_source(spark, entry: dict):
    # Reads source file from Volume with registry options and applies null exclusion if specified.
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
    # Returns a DataFrame of SHA-256 row hashes for the specified columns.
    normalised = df.select([
        F.coalesce(F.trim(F.col(c).cast("string")), F.lit("")).alias(c)
        for c in columns
    ])
    return (
        normalised
        .withColumn("row_hash", F.sha2(F.concat_ws("||", *columns), 256))
        .select("row_hash")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 — Volume & Completeness
# MAGIC Compares row counts between source file and Bronze table.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t1_source_count_equals_bronze_count(spark, entry):
    # Verify source row count matches Bronze table row count, excluding known-bad rows.
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
    # Ensure Bronze table is not empty.
    count = spark.read.table(get_table_path(entry["table"])).count()
    assert count > 0, f"[{entry['table']}] Bronze table is empty."


def test_t1_chunk1_no_null_id_rows(spark):
    # Confirm listings_csv_copyinto Bronze table contains no rows with null id.
    null_id_rows = (
        spark.read.table(get_table_path("listings_csv_copyinto"))
        .filter(F.col("id").isNull())
        .count()
    )
    assert null_id_rows == 0, (
        f"listings_csv_copyinto contains {null_id_rows} null-id rows — "
        "these should have been excluded during ingestion."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2 — Row-to-Row Integrity (SHA-256)
# MAGIC Generates a SHA-256 fingerprint for every row using all columns present in both
# MAGIC source and Bronze. Compares fingerprint sets via subtract — any mismatch means
# MAGIC a row was corrupted, altered, or missing.

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t2_no_source_rows_missing_from_bronze(spark, entry):
    # Ensure every source row fingerprint exists in Bronze (nothing lost).
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
    # Ensure no Bronze row fingerprint is absent from source (Bronze must not invent rows).
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## T3 — Schema & Metadata

# COMMAND ----------

@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_audit_columns_present(spark, entry):
    # Verify load_dt and source_file exist in every Bronze table schema.
    df      = spark.read.table(get_table_path(entry["table"]))
    missing = [c for c in ("load_dt", "source_file") if c not in df.columns]
    assert missing == [], (
        f"[{entry['table']}] Missing audit columns: {missing}"
    )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_audit_columns_non_null(spark, entry):
    # Ensure load_dt and source_file have zero null values in every Bronze table.
    df = spark.read.table(get_table_path(entry["table"]))
    for col in ("load_dt", "source_file"):
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['table']}] Column '{col}' has {null_cnt:,} NULL rows."
            )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_load_dt_is_timestamp(spark, entry):
    # Check that load_dt is TIMESTAMP type, not string or date.
    df     = spark.read.table(get_table_path(entry["table"]))
    dtypes = dict(df.dtypes)
    if "load_dt" in dtypes:
        assert dtypes["load_dt"].startswith("timestamp"), (
            f"[{entry['table']}] load_dt is '{dtypes['load_dt']}', expected 'timestamp'."
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_no_rescued_data_pollution(spark, entry):
    # If _rescued_data column is present, it must be entirely NULL.
    df = spark.read.table(get_table_path(entry["table"]))
    if "_rescued_data" in df.columns:
        rescued = df.filter(F.col("_rescued_data").isNotNull()).count()
        assert rescued == 0, (
            f"[{entry['table']}] _rescued_data has {rescued:,} non-null rows — "
            "indicates schema mismatch during ingestion."
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_source_file_identifies_origin(spark, entry):
    # source_file values must correctly identify the ingestion source file.
    expected = SOURCE_FILE_MAP[entry["table"]]
    df    = spark.read.table(get_table_path(entry["table"]))
    files = [r["source_file"] for r in df.select("source_file").distinct().collect()]
    assert any(expected in f for f in files), (
        f"[{entry['table']}] Expected source_file containing '{expected}', got {files}."
    )


@pytest.mark.parametrize("entry", [
    pytest.param(e, id=e["table"])
    for e in REGISTRY
    if e["table"] in LISTING_TABLES
])
def test_t3_listing_id_is_string_type(spark, entry):
    # id column in listing Bronze tables must be STRING type.
    df     = spark.read.table(get_table_path(entry["table"]))
    dtypes = dict(df.dtypes)
    if "id" in dtypes:
        assert dtypes["id"] == "string", (
            f"[{entry['table']}] 'id' should be STRING in Bronze, got '{dtypes['id']}'."
        )


@pytest.mark.parametrize("entry", REGISTRY_PARAMS)
def test_t3_source_file_non_empty_string(spark, entry):
    # source_file must never be an empty string.
    df = spark.read.table(get_table_path(entry["table"]))
    if "source_file" in df.columns:
        bad = df.filter(F.trim(F.col("source_file")) == "").count()
        assert bad == 0, (
            f"[{entry['table']}] {bad} rows have empty source_file."
        )
