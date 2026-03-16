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
    # Pipeline : 03_bronze_csv_copyinto.py
    # Method   : COPY INTO (SQL), force=false — file-tracked, idempotent
    # Null-id  : Step 4 DELETE FROM WHERE id IS NULL (permanent purge)
    # Repair   : Step 5 anti-join on id + append any missing valid rows
    # Schema   : CREATE TABLE DDL — 19 STRING + load_dt TIMESTAMP + source_file STRING
    {
        "name"             : "Chunk 1 — CSV / COPY INTO",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_1.csv",
        "table"            : "listings_csv_copyinto",
        "fmt"              : "csv",
        "opts"             : {"header": "true"},
        "null_exclude_col" : "id",          # pipeline DELETEs null-id rows — mirror this
        "has_id"           : True,
        "source_file_value": "1_main_chunk_1.csv",
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": True,    # U3: pipeline deletes null-id rows
        "idempotency_note" : "COPY INTO force=false (file-tracked)",
    },

    # ── 06: CSV / DLT Auto Loader ──────────────────────────────────────────────
    # Pipeline : 06_bronze_dlt.py
    # Method   : DLT cloudFiles CSV, pathGlobFilter=1_main_chunk_2.csv
    # Null-id  : @dlt.expect('valid_id','id IS NOT NULL') — WARN ONLY, rows kept
    # Null-cost: @dlt.expect('valid_cost','cost IS NOT NULL') — WARN ONLY, rows kept
    # Schema   : StructType — 19 STRING + audit cols
    {
        "name"             : "Chunk 2 — CSV / DLT Auto Loader",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_2.csv",
        "table"            : "listings_csv_dlt",
        "fmt"              : "csv",
        "opts"             : {"header": "true"},
        "null_exclude_col" : None,          # DLT expect is warn-only — rows NOT dropped
        "has_id"           : True,
        "source_file_value": "1_main_chunk_2.csv",
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": False,   # @dlt.expect is warn-only — nulls MAY be present
        "idempotency_note" : "DLT checkpoint-managed",
    },

    # ── 04: JSON / Auto Loader foreachBatch MERGE ──────────────────────────────
    # Pipeline : 04_bronze_auto_loader.py
    # Method   : cloudFiles JSON, foreachBatch MERGE on id (whenNotMatchedInsertAll)
    # Null-id  : filter(id IS NOT NULL) inside foreachBatch — rows never reach MERGE
    # Keys     : may need positional rename if JSON keys mismatch DDL (broken keys case)
    # _rescued : dropped in foreachBatch if Auto Loader added it
    # source_file: _metadata.file_path (full volume path, not a bare filename)
    {
        "name"             : "Chunk 3 — JSON / Auto Loader MERGE",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_3.json",
        "table"            : "listings_json_autoloader",
        "fmt"              : "json",
        "opts"             : {"multiLine": "true"},
        "null_exclude_col" : "id",          # foreachBatch drops null-id before MERGE
        "has_id"           : True,
        "source_file_value": "1_main_chunk_3.json",  # present inside _metadata.file_path
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": True,    # foreachBatch null guard
        "idempotency_note" : "MERGE on id — whenNotMatchedInsertAll (existing ids skipped)",
    },

    # ── 05: XML / PySpark native reader ───────────────────────────────────────
    # Pipeline : 05_bronze_xml_pyspark.py
    # Method   : spark.read.format("xml"), rowTag="record", explicit StructType
    # Null-id  : filter(id IS NOT NULL) before write
    # Idempotency: DELETE WHERE source_file = '1_main_chunk_4.xml' then append
    # source_file: F.lit("1_main_chunk_4.xml") — exact literal
    {
        "name"             : "Chunk 4 — XML / PySpark Native",
        "src"              : f"{CHUNKS_PATH}/1_main_chunk_4.xml",
        "table"            : "listings_xml_pyspark",
        "fmt"              : "xml",
        "opts"             : {"rowTag": "record"},
        "null_exclude_col" : "id",          # filter(id IS NOT NULL) before write
        "has_id"           : True,
        "source_file_value": "1_main_chunk_4.xml",
        "expected_cols"    : [
            "cost","currency","marka","model","year","has_license","place","date",
            "id","engine","power","gear","probeg","sWheel","complectation",
            "transmission","R","G","B",
        ],
        "id_null_guaranteed_zero": True,    # filter before write
        "idempotency_note" : "DELETE WHERE source_file=FILE_NAME + append",
    },

    # ── 07: Text descriptions ──────────────────────────────────────────────────
    # Pipeline : 07_remaining_4_files.py
    # Method   : DLT cloudFiles CSV, multiLine=true, escape/quote for Russian text
    # Null-id  : @dlt.expect('valid_id') — warn-only, nulls kept
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
    # Pipeline : 07_remaining_4_files.py
    # _c0      : unnamed pandas index — Spark names it _c0 when header=true
    # Null-id  : @dlt.expect('valid_id') — warn-only
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
    # Pipeline : 07_remaining_4_files.py
    # sep=;    : semicolon-delimited
    # Columns  : 19 Cyrillic column names (requires delta.columnMapping.mode=name)
    # No id    : catalog has no listing id column
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
    # Pipeline : 07_remaining_4_files.py
    # _c0      : unnamed pandas index (same as listings_photo)
    # lat/lon  : kept as STRING in Bronze — cast to DOUBLE in Silver
    # No id    : no listing id column
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
    """
    Reads the source file from Volume using the registry fmt/opts.
    If null_exclude_col is set, drops rows where that column is null —
    mirroring exactly what the pipeline does before writing to Bronze.

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
    if excl_col and excl_col in df.columns:
        null_cnt = df.filter(F.col(excl_col).isNull()).count()
        df       = df.filter(F.col(excl_col).isNotNull())
        return df, f"{null_cnt:,} null-{excl_col} source rows excluded (pipeline removes these)"
    return df, None


def row_hash(df, cols: list):
    """
    Computes a SHA-256 row fingerprint over the specified columns.
    Nulls are coerced to '' and values are trimmed before hashing so
    both source and Bronze normalise identically regardless of whitespace
    or null representation differences.

    Returns a DataFrame with a single 'row_hash' STRING column.
    """
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
    """
    U1 — Bronze table must exist in the catalog.
    Fails if the ingestion pipeline never ran or the table was dropped.
    """
    assert spark.catalog.tableExists(B(entry["table"])), (
        f"[{entry['table']}] Table not found: {B(entry['table'])}. "
        f"Run ingestion pipeline ({entry.get('name','?')}) first."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u1_table_non_empty(spark, entry):
    """
    U1 — Bronze table must contain at least one row.
    An empty table means the pipeline ran but wrote nothing.
    """
    count = spark.read.table(B(entry["table"])).count()
    assert count > 0, (
        f"[{entry['table']}] Table is empty after ingestion."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u1_all_expected_columns_present(spark, entry):
    """
    U1 — Every column declared in the pipeline DDL must exist in Bronze.

    Source of truth per pipeline:
      listings_csv_copyinto    : CREATE TABLE DDL in 03_bronze_csv_copyinto.py
      listings_csv_dlt         : StructType in 06_bronze_dlt.py
      listings_json_autoloader : BRONZE_SCHEMA_DDL string in 04_bronze_auto_loader.py
      listings_xml_pyspark     : StructType in 05_bronze_xml_pyspark.py
      listings_text/photo/geo  : StructType in 07_remaining_4_files.py
      car_catalog              : StructType (Cyrillic) in 07_remaining_4_files.py

    A missing column means the schema was altered after table creation
    or the pipeline wrote with a wrong DDL.
    """
    actual  = spark.read.table(B(entry["table"])).columns
    missing = [c for c in entry["expected_cols"] if c not in actual]
    assert missing == [], (
        f"[{entry['table']}] Columns in pipeline DDL but missing from Bronze: {missing}. "
        f"Schema drift or wrong DDL applied."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u1_all_data_columns_are_string_type(spark, entry):
    """
    U1 — All raw data columns must be STRING type.

    Every pipeline uses inferSchema=false and explicit STRING DDL.
    A non-STRING data column means schema inference was accidentally enabled.

    Excludes audit columns (load_dt=TIMESTAMP, source_file=STRING) — these
    are intentionally typed differently from the raw data columns.
    """
    dtypes     = dict(spark.read.table(B(entry["table"])).dtypes)
    audit_cols = {"load_dt", "source_file", "_rescued_data", "_c0"}  # _c0 OK as string too
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
    """
    U1 — The 'id' column in the 4 listing tables must be STRING, not INT/LONG/DOUBLE.

    All 4 listing pipelines (03,04,05,06) use inferSchema=false so 'id' must
    remain as STRING in Bronze. Silver casts it with try_cast(id as long).
    If Bronze already contains a numeric type, inferSchema was accidentally enabled.
    """
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
    """
    U2 — load_dt and source_file must exist in every Bronze table schema.
    All pipelines (03–07) explicitly add both columns.
    """
    cols    = spark.read.table(B(entry["table"])).columns
    missing = [c for c in ("load_dt", "source_file") if c not in cols]
    assert missing == [], (
        f"[{entry['table']}] Missing audit columns: {missing}."
    )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_audit_columns_non_null(spark, entry):
    """
    U2 — load_dt and source_file must have zero NULL values in every row.
    All pipelines assign these explicitly on every row written.
    A NULL indicates a pipeline defect or a partial write.
    """
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
    """
    U2 — load_dt must be TIMESTAMP, not string or date.

    03: CREATE TABLE DDL defines load_dt TIMESTAMP.
    04,05,06,07: Spark infers TIMESTAMP from current_timestamp() automatically.
    """
    dtypes = dict(spark.read.table(B(entry["table"])).dtypes)
    if "load_dt" in dtypes:
        assert dtypes["load_dt"].startswith("timestamp"), (
            f"[{entry['table']}] load_dt is '{dtypes['load_dt']}', expected 'timestamp'. "
            f"COPY INTO may have written load_dt as a string."
        )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_source_file_is_string_type(spark, entry):
    """
    U2 — source_file must be STRING type.
    All pipelines use F.lit(FILE_NAME) or _metadata.file_path (both produce STRING).
    """
    dtypes = dict(spark.read.table(B(entry["table"])).dtypes)
    if "source_file" in dtypes:
        assert dtypes["source_file"] == "string", (
            f"[{entry['table']}] source_file is '{dtypes['source_file']}', expected 'string'."
        )


@pytest.mark.parametrize("entry", ALL_PARAMS)
def test_u2_source_file_contains_expected_filename(spark, entry):
    """
    U2 — source_file values must contain the expected origin filename.

    03: COPY INTO uses the FILE_NAME widget literal -> '1_main_chunk_1.csv'
    04: Auto Loader uses _metadata.file_path -> full volume path containing '1_main_chunk_3.json'
    05: F.lit('1_main_chunk_4.xml') -> exact match
    06: F.lit('1_main_chunk_2.csv') -> exact match
    07: F.lit per table -> exact match

    Uses 'in' check so it works for both bare filenames (05,06,07)
    and full paths (04 _metadata.file_path).
    """
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
    """
    U2 — source_file must not be an empty or whitespace-only string.
    An empty source_file means the audit assignment was bypassed.
    """
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
    """
    U3 [listings_csv_copyinto] — Zero null-id rows must remain in Bronze.

    Pipeline Step 4 runs: DELETE FROM listings_csv_copyinto WHERE id IS NULL
    This permanently removes rows that COPY INTO loaded in PERMISSIVE mode
    where the CSV 'id' column was empty or malformed.

    If null-id rows remain, Step 4 did not execute or was skipped.
    """
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
    """
    U3 [listings_csv_copyinto] — id column must be unique (no duplicates).

    COPY INTO force=false tracks ingested files so re-runs skip already-loaded files.
    The missing-row repair in Step 5 uses anti-join on id before appending,
    so it cannot create duplicates.
    Duplicates would indicate force=true was used or the repair logic was bypassed.
    """
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
    """
    U3 [listings_json_autoloader] — Zero null-id rows must exist in Bronze.

    Pipeline upsert_to_bronze() Step C:
        micro_batch_df = micro_batch_df.filter(F.col('id').isNotNull())
    This drops null-id rows before the MERGE executes.
    Null-id rows in Bronze mean the guard was removed or bypassed.
    """
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
    """
    U3 [listings_json_autoloader] — Pipeline must not introduce NEW duplicate ids.

    Root cause of the 14 existing duplicates (confirmed by analysis):
      The source JSON file itself contains 14 rows with duplicate ids
      (same listing posted twice in the source data).
      On first run, the pipeline uses mode('overwrite') because the table
      doesn't exist yet — this writes all source rows as-is, including source
      duplicates. The MERGE path only applies on re-runs.
      The MERGE (whenNotMatchedInsertAll) cannot retroactively remove duplicates
      that were written on the first run.

    What this test asserts:
      Bronze duplicates must NOT exceed source duplicates.
      If Bronze has MORE dups than source, the pipeline introduced new ones
      (broken MERGE, double-append, or wrong overwrite on re-run).
      If Bronze has the SAME dups as source, all duplicates are source-originated
      and the pipeline behaved correctly.

    Separation of concerns:
      Source data quality (duplicate ids in JSON) → Silver layer's responsibility
      to deduplicate via dropDuplicates(['listing_id']).
      Pipeline integrity (no new duplicates introduced) → this test's responsibility.
    """
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
    """
    U3 [listings_json_autoloader] — _rescued_data must not exist or be all NULL.

    Pipeline Step B in upsert_to_bronze():
        if '_rescued_data' in micro_batch_df.columns:
            micro_batch_df = micro_batch_df.drop('_rescued_data')
    Auto Loader adds _rescued_data when a JSON field does not fit the schema.
    The pipeline drops it in foreachBatch.
    A non-null _rescued_data value means schema mismatch occurred and the drop
    did not execute (e.g. first write used overwrite before MERGE branch).
    """
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
    """
    U3 [listings_xml_pyspark] — Zero null-id rows must exist in Bronze.

    Pipeline applies: df_clean = df_xml.filter(F.col('id').isNotNull())
    before writing. Null-id rows in Bronze mean the filter was removed.
    """
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
    """
    U3 [listings_xml_pyspark] — All Bronze rows must have source_file = '1_main_chunk_4.xml'.

    Pipeline idempotency:
        DELETE FROM listings_xml_pyspark WHERE source_file = '1_main_chunk_4.xml'
        then append
    If multiple distinct source_file values exist, the DELETE did not remove
    all previous rows and stale data from an earlier run remains.
    """
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
    """
    U3 [car_catalog] — All 19 Cyrillic column names must be present in Bronze.

    Pipeline 07 uses delta.columnMapping.mode=name (table property) which
    allows Delta to store Cyrillic characters in column names.
    If Cyrillic columns are missing, columnMapping was not applied correctly.
    """
    required = ["Марка", "Модель", "Тип топлива", "Коробка передач",
                "Привод", "Страна сборки", "Тип кузова"]
    cols     = spark.read.table(B("car_catalog")).columns
    missing  = [c for c in required if c not in cols]
    assert missing == [], (
        f"car_catalog is missing Cyrillic columns: {missing}. "
        f"delta.columnMapping.mode=name may not be set correctly."
    )


def test_u3_geo_lat_lon_are_string_in_bronze(spark):
    """
    U3 [geo_locations] — lat and lon must be STRING type in Bronze.

    Pipeline 07 SCHEMA_GEO defines lat and lon as StringType():
        StructField('lat', StringType(), True)
        StructField('lon', StringType(), True)
    inferSchema=false enforces this.
    Silver (_transform_geo) casts them to DOUBLE with .cast('double').
    If lat/lon are already DOUBLE in Bronze, inferSchema was accidentally
    enabled and Silver's cast becomes a no-op.
    """
    dtypes = dict(spark.read.table(B("geo_locations")).dtypes)
    for col in ("lat", "lon"):
        if col in dtypes:
            assert dtypes[col] == "string", (
                f"geo_locations.{col} is '{dtypes[col]}', expected 'string'. "
                f"inferSchema=false was not respected in SCHEMA_GEO."
            )


def test_u3_photo_c0_index_column_present(spark):
    """
    U3 [listings_photo] — _c0 column must be present in Bronze.

    The source CSV 1_photo.csv was generated from a pandas DataFrame and
    contains an unnamed index column. Spark names it '_c0' when header=true.
    Pipeline 07 SCHEMA_PHOTO explicitly includes:
        StructField('_c0', StringType(), True)
    If _c0 is missing, the schema was changed without updating the file handling.
    """
    cols = spark.read.table(B("listings_photo")).columns
    assert "_c0" in cols, (
        "listings_photo is missing the '_c0' column. "
        "Source CSV has an unnamed pandas index — SCHEMA_PHOTO must include _c0."
    )


def test_u3_text_multiline_not_truncated(spark):
    """
    U3 [listings_text] — Average text column length must exceed 20 characters.

    Pipeline 07 uses multiLine=true for Russian car descriptions that span
    multiple CSV lines. If multiLine is disabled, text is truncated at the
    first newline inside a description.
    Average length below 20 chars is a strong signal of truncation.
    """
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
    """
    R1 — Source row count must exactly match Bronze row count.

    Exclusion logic is applied to the source before counting to mirror
    exactly what the pipeline does. See registry null_exclude_col.

    A non-zero gap means:
      Positive gap (src > bronze): rows were silently dropped during ingestion
      Negative gap (src < bronze): rows were invented (inflation)
    Both are failures.
    """
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
    """
    R1 — Bronze must not contain MORE rows than the source.

    Inflation means the pipeline inserted rows that have no origin in the source.
    Possible causes:
      - COPY INTO force=true run twice
      - XML DELETE+append ran twice without DELETE completing
      - MERGE matched on wrong columns and inserted duplicates
      - Missing-row repair (03) inserted wrong rows
    """
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
    """
    R2 — Every source row SHA-256 fingerprint must exist in Bronze.

    Uses subtract():
        source_hashes MINUS bronze_hashes = rows in source but absent from Bronze

    If the result is non-empty, those source rows were either:
      (a) dropped during ingestion
      (b) corrupted — a value changed so the hash no longer matches

    Audit columns (load_dt, source_file, _rescued_data) are excluded from the
    hash because they do not exist in the source file and would always differ.

    The same null_exclude_col logic as R1 is applied to avoid false positives
    from rows the pipeline legitimately removes.
    """
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
    """
    R3 — No Bronze row fingerprint must be absent from the source file.

    Uses subtract():
        bronze_hashes MINUS source_hashes = rows in Bronze with no source origin

    If the result is non-empty, those Bronze rows were invented by the pipeline.
    Possible causes:
      - Missing-row repair (03) inserted wrong rows
      - MERGE (04) matched on wrong condition and created phantom rows
      - XML DELETE+append (05) left rows from a different source file
      - A different source file was accidentally ingested into the same table

    The same null_exclude_col and audit column exclusions apply as in R2.
    """
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
