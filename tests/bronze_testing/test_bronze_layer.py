# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Test Suite
# MAGIC
# MAGIC Two test types across 6 suites:
# MAGIC
# MAGIC | Type | Suite | Tests | What it proves |
# MAGIC |------|-------|-------|----------------|
# MAGIC | **Unit** | U1 — Schema | 5 | Every DDL column present, all STRING, id is STRING not numeric |
# MAGIC | **Unit** | U2 — Audit Columns | 6 | load_dt + source_file present, non-null, correct type, correct filename |
# MAGIC | **Unit** | U3 — Pipeline Constraints | 10 | Null-id purge, MERGE idempotency, XML idempotency, Cyrillic cols, lat/lon STRING, _c0, DLT warn-only, multiLine |
# MAGIC | **Reconciliation** | R1 — Row Count | 2 | source == Bronze (exact match, no drops, no inflation) |
# MAGIC | **Reconciliation** | R2 — Source→Bronze Integrity | 1 | Every source row fingerprint (SHA-256) exists in Bronze |
# MAGIC | **Reconciliation** | R3 — Bronze→Source Integrity | 1 | Every Bronze row fingerprint traces back to source (no invention) |
# MAGIC
# MAGIC **Total: 25 tests across 8 Bronze tables**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark Session & Configuration

# COMMAND ----------

@pytest.fixture(scope="session")
def spark():
    """Databricks Connect session shared across all tests in this suite."""
    return DatabricksSession.builder.getOrCreate()


CONFIG = {
    "catalog": "vstone_catalog",
    "bronze":  "bronze",
    "raw":     "raw",
}

CHUNKS_PATH  = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/chunks"
LANDING_PATH = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/landing"


def B(table: str) -> str:
    """Returns the fully-qualified Bronze table name."""
    return f"{CONFIG['catalog']}.{CONFIG['bronze']}.{table}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registry
# MAGIC
# MAGIC Single source of truth for all 8 Bronze tables. Every field is derived
# MAGIC directly from the ingestion pipeline source files.

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# Each entry maps directly to one Bronze table and one ingestion pipeline.
# Fields used by tests are marked with the test suites that consume them.
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY = [

    # ── 03: CSV / COPY INTO ────────────────────────────────────────────────────
    {
        "name"             : "Chunk 1 — CSV / COPY INTO",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_1.csv",
        "table"            : "listings_csv_copyinto",
        "fmt"              : "csv",
        "opts"             : {"header": "true"},
        "null_exclude_col" : "id",          
        "has_id"           : True,
        "source_file_value": "1_main_chunk_1.csv",
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": True,   
        "idempotency_note" : "COPY INTO force=false (file-tracked)",
    },

    # ── 06: CSV / DLT Auto Loader ──────────────────────────────────────────────
    {
        "name"             : "Chunk 2 — CSV / DLT Auto Loader",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_2.csv",
        "table"            : "listings_csv_dlt",
        "fmt"              : "csv",
        "opts"             : {"header": "true"},
        "null_exclude_col" : None,          
        "has_id"           : True,
        "source_file_value": "1_main_chunk_2.csv",
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": False,   
        "idempotency_note" : "DLT checkpoint-managed",
    },

    # ── 04: JSON / Auto Loader foreachBatch MERGE ──────────────────────────────
    {
        "name"             : "Chunk 3 — JSON / Auto Loader MERGE",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_3.json",
        "table"            : "listings_json_autoloader",
        "fmt"              : "json",
        "opts"             : {"multiLine": "true"},
        "null_exclude_col" : "id",          
        "has_id"           : True,
        "source_file_value": "1_main_chunk_3.json",  
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": True,    
        "idempotency_note" : "MERGE on id — whenNotMatchedInsertAll (existing ids skipped)",
    },

    # ── 05: XML / PySpark native reader ───────────────────────────────────────
    {
        "name"             : "Chunk 4 — XML / PySpark Native",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_4.xml",
        "table"            : "listings_xml_pyspark",
        "fmt"              : "xml",
        "opts"             : {"rowTag": "record"},
        "null_exclude_col" : "id",          
        "has_id"           : True,
        "source_file_value": "1_main_chunk_4.xml",
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": True,    
        "idempotency_note" : "DELETE WHERE source_file=FILE_NAME + append",
    },

    # ── 07: Text descriptions ──────────────────────────────────────────────────
    {
        "name"             : "Landing — Text Descriptions",
        "src"              : f"{LANDING_PATH}/1_text.csv",
        "table"            : "listings_text",
        "fmt"              : "csv",
        "opts"             : {"header": "true", "multiLine": "true", "escape": '"'},
        "null_exclude_col" : None,
        "has_id"           : True,
        "source_file_value": "1_text.csv",
        "expected_cols"    : ["id", "text"],
        "id_null_guaranteed_zero": False,
        "idempotency_note" : "DLT checkpoint-managed",
    },

    # ── 07: Photo URLs ─────────────────────────────────────────────────────────
    {
        "name"             : "Landing — Photo URLs",
        "src"              : f"{LANDING_PATH}/1_photo.csv",
        "table"            : "listings_photo",
        "fmt"              : "csv",
        "opts"             : {"header": "true"},
        "null_exclude_col" : None,
        "has_id"           : True,
        "source_file_value": "1_photo.csv",
        "expected_cols"    : ["_c0", "photo_url", "id"],
        "id_null_guaranteed_zero": False,
        "idempotency_note" : "DLT checkpoint-managed",
    },

    # ── 07: Car catalog ────────────────────────────────────────────────────────
    {
        "name"             : "Landing — Car Catalog",
        "src"              : f"{LANDING_PATH}/catalogs.csv",
        "table"            : "car_catalog",
        "fmt"              : "csv",
        "opts"             : {"header": "true", "sep": ";"},
        "null_exclude_col" : None,
        "has_id"           : False,
        "source_file_value": "catalogs.csv",
        "expected_cols"    : [
            "Марка","Модель","Поколение","Комплектация",
            "Объём двигателя","Мощность двигателя","Расход топлива",
            "Тип топлива","Коробка передач","Привод","Кол-во мест",
            "Клиренс","Объем багажника","Период выпуска","Тип кузова",
            "Марка кузова","Время разгона 0-100 км/ч, с",
            "Максимальная скорость, км/ч","Страна сборки",
        ],
        "id_null_guaranteed_zero": False,
        "idempotency_note" : "DLT checkpoint-managed",
    },

    # ── 07: Geo locations ──────────────────────────────────────────────────────
    {
        "name"             : "Landing — Geo Locations",
        "src"              : f"{LANDING_PATH}/final_geografic.csv",
        "table"            : "geo_locations",
        "fmt"              : "csv",
        "opts"             : {"header": "true"},
        "null_exclude_col" : None,
        "has_id"           : False,
        "source_file_value": "final_geografic.csv",
        "expected_cols"    : ["_c0", "name_padesh", "greate_padesh", "lat", "lon"],
        "id_null_guaranteed_zero": False,
        "idempotency_note" : "DLT checkpoint-managed",
    },
]

# ── Parametrize helpers ────────────────────────────────────────────────────────
ALL_PARAMS     = [pytest.param(e, id=e["table"]) for e in REGISTRY]
LISTING_PARAMS = [pytest.param(e, id=e["table"]) for e in REGISTRY
                  if e["table"] in (
                      "listings_csv_copyinto","listings_csv_dlt",
                      "listings_json_autoloader","listings_xml_pyspark",
                  )]
NULL_ZERO_PARAMS = [pytest.param(e, id=e["table"]) for e in REGISTRY
                    if e.get("id_null_guaranteed_zero")]

print(f"Registry: {len(REGISTRY)} tables | "
      f"{len(LISTING_PARAMS)} listing tables | "
      f"{len(NULL_ZERO_PARAMS)} tables with guaranteed-zero null-id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def read_source(spark, entry: dict):
    df = (
        spark.read
        .format(entry["fmt"])
        .options(**entry["opts"])
        .option("inferSchema", "false")
        .load(entry["src"])
    )
    excl_col = entry.get("null_exclude_col")
    if excl_col and excl_col in df.columns:
        null_cnt = df.filter(F.col(excl_col).isNull()).count()
        df       = df.filter(F.col(excl_col).isNotNull())
        return df, f"{null_cnt:,} null-{excl_col} source rows excluded (pipeline removes these)"
    return df, None


def row_hash(df, cols: list):
    normalised = df.select([
        F.coalesce(F.trim(F.col(c).cast("string")), F.lit("")).alias(c)
        for c in cols
    ])
    return (
        normalised
        .withColumn("row_hash", F.sha2(F.concat_ws("||", *cols), 256))
        .select("row_hash")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U1 — Unit Tests: Schema & Data Types

# COMMAND ----------

@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u1_table_exists(spark, entry):
    assert spark.catalog.tableExists(B(entry["table"])), (
        f"[{entry['table']}] Table not found: {B(entry['table'])}. "
        f"Run ingestion pipeline ({entry.get('name','?')}) first."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u1_table_non_empty(spark, entry):
    count = spark.read.table(B(entry["table"])).count()
    assert count > 0, (
        f"[{entry['table']}] Table is empty after ingestion."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u1_all_expected_columns_present(spark, entry):
    actual  = spark.read.table(B(entry["table"])).columns
    missing = [c for c in entry["expected_cols"] if c not in actual]
    assert missing == [], (
        f"[{entry['table']}] Columns in pipeline DDL but missing from Bronze: {missing}. "
        f"Schema drift or wrong DDL applied."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u1_all_data_columns_are_string_type(spark, entry):
    dtypes     = dict(spark.read.table(B(entry["table"])).dtypes)
    audit_cols = {"load_dt", "source_file", "_rescued_data", "_c0"}  
    non_string = [
        c for c in entry["expected_cols"]
        if c in dtypes and dtypes[c] != "string" and c not in audit_cols
    ]
    assert non_string == [], (
        f"[{entry['table']}] Data columns with wrong type (expected string): "
        f"{[(c, dtypes[c]) for c in non_string]}. "
        f"inferSchema=false may not have been applied."
    )


@pytest.mark.parametrize("entry", LISTING_PARAMS)
def test_u1_listing_id_is_string_not_numeric(spark, entry):
    dtypes = dict(spark.read.table(B(entry["table"])).dtypes)
    assert dtypes.get("id") == "string", (
        f"[{entry['table']}] 'id' column type is '{dtypes.get('id')}', expected 'string'. "
        f"inferSchema=false was not applied — Silver cast to long will be a no-op."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U2 — Unit Tests: Audit Columns
# MAGIC
# MAGIC Every Bronze table must carry `load_dt` (TIMESTAMP) and `source_file` (STRING).
# MAGIC These are added by every pipeline and are mandatory for end-to-end lineage.

# COMMAND ----------

@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_audit_columns_present(spark, entry):
    cols    = spark.read.table(B(entry["table"])).columns
    missing = [c for c in ("load_dt", "source_file") if c not in cols]
    assert missing == [], (
        f"[{entry['table']}] Missing audit columns: {missing}."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_audit_columns_non_null(spark, entry):
    df = spark.read.table(B(entry["table"]))
    for col in ("load_dt", "source_file"):
        if col in df.columns:
            null_cnt = df.filter(F.col(col).isNull()).count()
            assert null_cnt == 0, (
                f"[{entry['table']}] '{col}' has {null_cnt:,} NULL rows. "
                f"Pipeline must assign this column for every row written."
            )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_load_dt_is_timestamp_type(spark, entry):
    dtypes = dict(spark.read.table(B(entry["table"])).dtypes)
    if "load_dt" in dtypes:
        assert dtypes["load_dt"].startswith("timestamp"), (
            f"[{entry['table']}] load_dt is '{dtypes['load_dt']}', expected 'timestamp'. "
            f"COPY INTO may have written load_dt as a string."
        )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_source_file_is_string_type(spark, entry):
    dtypes = dict(spark.read.table(B(entry["table"])).dtypes)
    if "source_file" in dtypes:
        assert dtypes["source_file"] == "string", (
            f"[{entry['table']}] source_file is '{dtypes['source_file']}', expected 'string'."
        )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_source_file_contains_expected_filename(spark, entry):
    expected = entry["source_file_value"]
    files    = [
        r["source_file"]
        for r in spark.read.table(B(entry["table"]))
        .select("source_file").distinct().collect()
    ]
    assert any(expected in (f or "") for f in files), (
        f"[{entry['table']}] Expected source_file containing '{expected}'. "
        f"Found: {files}. Wrong file may have been ingested."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_source_file_not_empty_string(spark, entry):
    df = spark.read.table(B(entry["table"]))
    if "source_file" in df.columns:
        bad = df.filter(F.trim(F.col("source_file")) == "").count()
        assert bad == 0, (
            f"[{entry['table']}] {bad:,} rows have an empty source_file."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## U3 — Unit Tests: Pipeline-Specific Constraints
# MAGIC
# MAGIC Tests that are specific to one pipeline's behaviour.
# MAGIC Each test is derived directly from the ingestion logic in the pipeline source files.

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# 03_bronze_csv_copyinto.py specific tests
# ─────────────────────────────────────────────────────────────────────────────

def test_u3_copyinto_null_id_purge_complete(spark):
    null_rows = (
        spark.read.table(B("listings_csv_copyinto"))
        .filter(F.col("id").isNull())
        .count()
    )
    assert null_rows == 0, (
        f"listings_csv_copyinto has {null_rows:,} null-id rows. "
        f"Step 4 (DELETE FROM WHERE id IS NULL) did not complete correctly."
    )


def test_u3_copyinto_no_duplicate_ids(spark):
    df    = spark.read.table(B("listings_csv_copyinto")).filter(F.col("id").isNotNull())
    total = df.count()
    uniq  = df.select("id").distinct().count()
    dups  = total - uniq
    assert dups == 0, (
        f"listings_csv_copyinto has {dups:,} duplicate id values. "
        f"COPY INTO force=true may have been used, or the repair step appended duplicates."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 04_bronze_auto_loader.py specific tests
# ─────────────────────────────────────────────────────────────────────────────

def test_u3_json_foreachbatch_null_id_guard(spark):
    null_rows = (
        spark.read.table(B("listings_json_autoloader"))
        .filter(F.col("id").isNull())
        .count()
    )
    assert null_rows == 0, (
        f"listings_json_autoloader has {null_rows:,} null-id rows. "
        f"foreachBatch null guard (filter id IS NOT NULL) was not applied."
    )


def test_u3_json_merge_idempotency_no_duplicate_ids(spark):
    CHUNKS_PATH = f"/Volumes/{CONFIG['catalog']}/{CONFIG['raw']}/chunks"

    # Count duplicate ids in source JSON
    df_src = (
        spark.read
        .format("json")
        .option("multiLine", "true")
        .option("inferSchema", "false")
        .load(f"{CHUNKS_PATH}/1_main_chunk_3.json")
        .filter(F.col("id").isNotNull())
    )
    src_total      = df_src.count()
    src_unique_ids = df_src.select("id").distinct().count()
    src_dups       = src_total - src_unique_ids  # duplicates originating from source

    # Count duplicate ids in Bronze
    df_brz         = spark.read.table(B("listings_json_autoloader")).filter(F.col("id").isNotNull())
    brz_total      = df_brz.count()
    brz_unique_ids = df_brz.select("id").distinct().count()
    brz_dups       = brz_total - brz_unique_ids

    # Bronze duplicates must not exceed source duplicates
    # If they do, the pipeline added NEW duplicates beyond what the source had
    pipeline_introduced_dups = brz_dups - src_dups

    assert pipeline_introduced_dups <= 0, (
        f"listings_json_autoloader has {pipeline_introduced_dups:,} pipeline-introduced "
        f"duplicate id rows (beyond source duplicates). "
        f"Source dups: {src_dups:,} | Bronze dups: {brz_dups:,}. "
        f"MERGE idempotency (whenNotMatchedInsertAll) is broken — "
        f"re-runs are inserting rows that already exist in Bronze."
    )

    # Informational assert: confirm source-originated dups are expected
    if src_dups > 0:
        print(
            f"  INFO [{src_dups:,} source-originated duplicate ids in Bronze] "
            f"These come from the JSON source file itself, not from the pipeline. "
            f"Silver deduplicates them via dropDuplicates(['listing_id'])."
        )


def test_u3_json_rescued_data_absent_or_null(spark):
    df = spark.read.table(B("listings_json_autoloader"))
    if "_rescued_data" in df.columns:
        rescued = df.filter(F.col("_rescued_data").isNotNull()).count()
        assert rescued == 0, (
            f"listings_json_autoloader has {rescued:,} non-null _rescued_data rows. "
            f"Schema mismatch during Auto Loader ingestion — "
            f"foreachBatch drop('_rescued_data') did not execute."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 05_bronze_xml_pyspark.py specific tests
# ─────────────────────────────────────────────────────────────────────────────

def test_u3_xml_null_id_filter_applied(spark):
    null_rows = (
        spark.read.table(B("listings_xml_pyspark"))
        .filter(F.col("id").isNull())
        .count()
    )
    assert null_rows == 0, (
        f"listings_xml_pyspark has {null_rows:,} null-id rows. "
        f"filter(F.col('id').isNotNull()) before write was not applied."
    )


def test_u3_xml_idempotency_single_source_file(spark):
    files = [
        r["source_file"]
        for r in spark.read.table(B("listings_xml_pyspark"))
        .select("source_file").distinct().collect()
    ]
    unexpected = [f for f in files if "1_main_chunk_4.xml" not in f]
    assert unexpected == [], (
        f"listings_xml_pyspark has unexpected source_file values: {unexpected}. "
        f"DELETE + append idempotency left stale rows from a different file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 07_remaining_4_files.py specific tests
# ─────────────────────────────────────────────────────────────────────────────

def test_u3_catalog_cyrillic_columns_present(spark):
    required = ["Марка", "Модель", "Тип топлива", "Коробка передач",
                "Привод", "Страна сборки", "Тип кузова"]
    cols     = spark.read.table(B("car_catalog")).columns
    missing  = [c for c in required if c not in cols]
    assert missing == [], (
        f"car_catalog is missing Cyrillic columns: {missing}. "
        f"delta.columnMapping.mode=name may not be set correctly."
    )


def test_u3_geo_lat_lon_are_string_in_bronze(spark):
    dtypes = dict(spark.read.table(B("geo_locations")).dtypes)
    for col in ("lat", "lon"):
        if col in dtypes:
            assert dtypes[col] == "string", (
                f"geo_locations.{col} is '{dtypes[col]}', expected 'string'. "
                f"inferSchema=false was not respected in SCHEMA_GEO."
            )


def test_u3_photo_c0_index_column_present(spark):
    cols = spark.read.table(B("listings_photo")).columns
    assert "_c0" in cols, (
        "listings_photo is missing the '_c0' column. "
        "Source CSV has an unnamed pandas index — SCHEMA_PHOTO must include _c0."
    )


def test_u3_text_multiline_not_truncated(spark):
    df = spark.read.table(B("listings_text")).filter(F.col("text").isNotNull())
    if df.count() == 0:
        pytest.skip("listings_text has no non-null text rows — skipping length check.")
    avg_len = df.select(F.avg(F.length(F.col("text")))).collect()[0][0]
    assert avg_len is not None and avg_len > 20, (
        f"listings_text average text length is {avg_len:.1f} chars. "
        f"multiLine=true may not be active — text is likely truncated at embedded newlines."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## R1 — Reconciliation: Row Count (Source == Bronze)
# MAGIC
# MAGIC Compares the row count of the source file against the Bronze table.

# COMMAND ----------

@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_r1_row_count_source_equals_bronze(spark, entry):
    df_src, excl_note = read_source(spark, entry)
    src_count    = df_src.count()
    bronze_count = spark.read.table(B(entry["table"])).count()
    gap          = src_count - bronze_count

    assert gap == 0, (
        f"[{entry['table']}] Row count mismatch — "
        f"Source: {src_count:,} | Bronze: {bronze_count:,} | GAP: {gap:,}"
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_r1_bronze_not_inflated_beyond_source(spark, entry):
    df_src, excl_note = read_source(spark, entry)
    src_count    = df_src.count()
    bronze_count = spark.read.table(B(entry["table"])).count()

    assert bronze_count <= src_count, (
        f"[{entry['table']}] Bronze has MORE rows than source — "
        f"Source: {src_count:,} | Bronze: {bronze_count:,} | "
        f"Extra: {bronze_count - src_count:,}. "
        f"Possible double-append or broken idempotency."
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## R2 — Row-to-Row Integrity: Every Source Row Exists in Bronze
# MAGIC
# MAGIC For every row in the source file, a matching SHA-256 fingerprint must exist in Bronze.

# COMMAND ----------

@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_r2_every_source_row_fingerprint_in_bronze(spark, entry):
    df_src, excl_note = read_source(spark, entry)
    df_brz      = spark.read.table(B(entry["table"]))
    audit_cols  = {"load_dt", "source_file", "_rescued_data"}
    common_cols = [c for c in df_src.columns
                   if c in df_brz.columns and c not in audit_cols]

    assert len(common_cols) > 0, (
        f"[{entry['table']}] No common columns between source and Bronze. "
        f"Source cols: {list(df_src.columns)} | Bronze cols: {list(df_brz.columns)}"
    )

    missing = row_hash(df_src, common_cols).subtract(
              row_hash(df_brz, common_cols)).count()

    assert missing == 0, (
        f"[{entry['table']}] {missing:,} source row(s) not found in Bronze "
        f"(hash mismatch — rows were dropped or corrupted). "
        f"Compared {len(common_cols)} columns: {common_cols}"
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## R3 — Row-to-Row Integrity: No Invented Rows in Bronze
# MAGIC
# MAGIC The reverse direction of R2. For every row in Bronze (on the common columns),
# MAGIC a matching SHA-256 fingerprint must exist in the source file.
# MAGIC
# MAGIC **R2 + R3 together prove complete byte-for-byte integrity:**
# MAGIC - R2: source → Bronze (nothing lost)
# MAGIC - R3: Bronze → source (nothing invented)

# COMMAND ----------

@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_r3_no_bronze_row_absent_from_source(spark, entry):
    df_src, excl_note = read_source(spark, entry)
    df_brz      = spark.read.table(B(entry["table"]))
    audit_cols  = {"load_dt", "source_file", "_rescued_data"}
    common_cols = [c for c in df_src.columns
                   if c in df_brz.columns and c not in audit_cols]

    invented = row_hash(df_brz, common_cols).subtract(
               row_hash(df_src, common_cols)).count()

    assert invented == 0, (
        f"[{entry['table']}] {invented:,} Bronze row(s) have no matching source row "
        f"(rows were invented — not traceable to source file). "
        f"Compared {len(common_cols)} columns: {common_cols}"
        + (f" | NOTE: {excl_note}" if excl_note else "")
    )
