# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Bronze Auto Loader | JSON Ingestion
# MAGIC
# MAGIC Ingests `1_main_chunk_3.json` into `vstone_catalog.bronze.listings_json_autoloader`.
# MAGIC
# MAGIC | Design Decision      | Choice                        | Reason |
# MAGIC |----------------------|-------------------------------|--------|
# MAGIC | Schema               | Explicit DDL, inferSchema=false | All columns STRING in Bronze |
# MAGIC | Idempotency          | Checkpoint + MERGE on `id`    | Re-run never duplicates rows |
# MAGIC | Audit columns        | `load_dt`, `source_file`      | Mandatory on every row |
# MAGIC | Null guard           | `id IS NOT NULL`              | Drops phantom empty records |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets & Configuration

# COMMAND ----------

import pyspark.sql.functions as F
from delta.tables import DeltaTable

dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Catalog")
dbutils.widgets.text("raw_schema",      "raw",            "2. Raw Schema")
dbutils.widgets.text("bronze_schema",   "bronze",         "3. Bronze Schema")

CATALOG    = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
BRONZE     = dbutils.widgets.get("bronze_schema")

TARGET_TABLE    = f"{CATALOG}.{BRONZE}.listings_json_autoloader"
CHUNKS_PATH     = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/chunks"
SOURCE_FILE     = f"{CHUNKS_PATH}/1_main_chunk_3.json"
ISOLATED_SRC    = f"{CHUNKS_PATH}/isolated_json_source"
ISOLATED_FILE   = f"{ISOLATED_SRC}/1_main_chunk_3.json"
CHECKPOINT_BASE = f"{CHUNKS_PATH}/streaming_metadata/listings_json"
SCHEMA_LOC      = f"{CHECKPOINT_BASE}/schema"
CHECKPOINT_LOC  = f"{CHECKPOINT_BASE}/chkpt"

print(f"Target table   : {TARGET_TABLE}")
print(f"Source file    : {SOURCE_FILE}")
print(f"Isolated src   : {ISOLATED_SRC}")
print(f"Checkpoint     : {CHECKPOINT_LOC}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Schema DDL
# MAGIC
# MAGIC All 19 source columns as STRING — inferSchema = false.
# MAGIC Type casting happens in Silver, not Bronze.

# COMMAND ----------

BRONZE_SCHEMA_DDL = """
    cost          STRING,
    currency      STRING,
    marka         STRING,
    model         STRING,
    year          STRING,
    has_license   STRING,
    place         STRING,
    date          STRING,
    id            STRING,
    engine        STRING,
    power         STRING,
    gear          STRING,
    probeg        STRING,
    sWheel        STRING,
    complectation STRING,
    transmission  STRING,
    R             STRING,
    G             STRING,
    B             STRING
"""

print("Schema defined — inferSchema = false (all STRING)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-Flight Checks
# MAGIC
# MAGIC Three things must be true before the stream starts:
# MAGIC 1. `1_main_chunk_3.json` exists in the chunks volume
# MAGIC 2. It is staged inside `isolated_json_source/`
# MAGIC 3. The checkpoint is cleared if the isolated file was replaced
# MAGIC
# MAGIC This cell handles all three automatically.

# COMMAND ----------

import os
import shutil

# --- Check 1: Source file must exist ---
source_exists = len(dbutils.fs.ls(CHUNKS_PATH)) > 0
try:
    dbutils.fs.ls(SOURCE_FILE)
    source_exists = True
except Exception:
    source_exists = False

if not source_exists:
    raise FileNotFoundError(
        f"Source file not found: {SOURCE_FILE}\n"
        f"Run 02_data_chunking first to generate the JSON chunk."
    )
print(f"Source file confirmed : {SOURCE_FILE}")

# --- Check 2: Stage file into isolated_json_source ---
dbutils.fs.mkdirs(ISOLATED_SRC)
dbutils.fs.mkdirs(SCHEMA_LOC)
dbutils.fs.mkdirs(CHECKPOINT_LOC)

# Always copy fresh — ensures isolated folder has the latest file
dbutils.fs.cp(SOURCE_FILE, ISOLATED_FILE, recurse=False)
print(f"File staged at        : {ISOLATED_FILE}")

# --- Check 3: Peek at actual JSON keys (read WITHOUT schema first) ---
df_raw_check = spark.read.option("multiLine", "true").json(ISOLATED_FILE)
actual_keys  = df_raw_check.columns
print(f"Actual JSON keys      : {actual_keys}")

# Expected column names that match the DDL
EXPECTED_KEYS = ["cost","currency","marka","model","year","has_license",
                 "place","date","id","engine","power","gear","probeg",
                 "sWheel","complectation","transmission","R","G","B"]

# Build rename map: actual key -> correct name
# Keys match when they ARE the correct names (02_data_chunking ran correctly)
# Keys are wrong when they are first-data-row values (csv_to_json header bug)
keys_match = all(k in actual_keys for k in EXPECTED_KEYS)

if keys_match:
    print("JSON keys match DDL   : YES — no rename needed")
    df_final_check = df_raw_check
else:
    # Keys are broken (e.g. "500000.0", "₽", "FIAT" ...)
    # Map positionally: the JSON always has exactly 19 keys in source column order
    print(f"JSON keys mismatch    : actual keys do not match DDL")
    print(f"Applying positional rename: {actual_keys} -> {EXPECTED_KEYS}")
    if len(actual_keys) != len(EXPECTED_KEYS):
        raise ValueError(
            f"Cannot rename: expected {len(EXPECTED_KEYS)} columns, "
            f"found {len(actual_keys)} in JSON.\n"
            f"Actual keys: {actual_keys}"
        )
    df_renamed = df_raw_check
    for old_key, new_key in zip(actual_keys, EXPECTED_KEYS):
        df_renamed = df_renamed.withColumnRenamed(old_key, new_key)
    df_final_check = df_renamed

row_count = df_final_check.filter(F.col("id").isNotNull()).count()

if row_count == 0:
    raise ValueError(
        f"JSON has 0 valid rows after key resolution.\n"
        f"Actual keys found: {actual_keys}\n"
        f"Check that 1_main_chunk_3.json was produced by 02_data_chunking."
    )

print(f"Pre-flight row count  : {row_count:,} valid rows (id IS NOT NULL)")
print(f"\nPre-flight PASSED — stream is ready to start.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reset Checkpoint
# MAGIC
# MAGIC The checkpoint tracks which files Auto Loader has already processed.
# MAGIC We clear it here so the stream always re-processes the file on this run.
# MAGIC
# MAGIC **Idempotency is guaranteed by the MERGE** — clearing the checkpoint
# MAGIC does NOT cause duplicate rows because MERGE skips existing `id` values.

# COMMAND ----------

dbutils.fs.rm(CHECKPOINT_BASE, recurse=True)
dbutils.fs.mkdirs(SCHEMA_LOC)
dbutils.fs.mkdirs(CHECKPOINT_LOC)
print(f"Checkpoint reset      : {CHECKPOINT_BASE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Idempotent Micro-Batch Function
# MAGIC
# MAGIC - **First run**: table does not exist → created from first batch
# MAGIC - **Re-run**: table exists → MERGE on `id`
# MAGIC   - New `id` → inserted
# MAGIC   - Existing `id` → skipped (no duplicates, no updates)

# COMMAND ----------

EXPECTED_KEYS = ["cost","currency","marka","model","year","has_license",
                 "place","date","id","engine","power","gear","probeg",
                 "sWheel","complectation","transmission","R","G","B"]

def upsert_to_bronze(micro_batch_df, batch_id):
    # Step A: Rename broken keys -> correct column names if needed
    current_cols = [c for c in micro_batch_df.columns
                    if c not in ("load_dt", "source_file", "_rescued_data")]
    needs_rename = any(c not in EXPECTED_KEYS for c in current_cols)

    if needs_rename:
        # Keys are broken — rename positionally to correct DDL names
        data_cols = [c for c in micro_batch_df.columns
                     if c not in ("load_dt", "source_file", "_rescued_data")]
        if len(data_cols) == len(EXPECTED_KEYS):
            for old_col, new_col in zip(data_cols, EXPECTED_KEYS):
                micro_batch_df = micro_batch_df.withColumnRenamed(old_col, new_col)
            print(f"  Batch {batch_id} — keys renamed to correct column names")

    # Step B: Drop _rescued_data if Auto Loader added it (schema mismatch helper col)
    if "_rescued_data" in micro_batch_df.columns:
        micro_batch_df = micro_batch_df.drop("_rescued_data")

    # Step C: Null guard — drop rows where id is null
    micro_batch_df = micro_batch_df.filter(F.col("id").isNotNull())

    batch_count = micro_batch_df.count()
    print(f"  Batch {batch_id} — {batch_count:,} valid rows after null guard")

    if batch_count == 0:
        print(f"  Batch {batch_id} — empty after null guard, skipping.")
        return

    if not spark.catalog.tableExists(TARGET_TABLE):
        print(f"  First run — creating table: {TARGET_TABLE}")
        (micro_batch_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(TARGET_TABLE))
        print(f"  Table created — {batch_count:,} rows written.")
    else:
        print(f"  Table exists — running MERGE on id...")
        target = DeltaTable.forName(spark, TARGET_TABLE)
        (target.alias("t")
            .merge(micro_batch_df.alias("s"), "t.id = s.id")
            .whenNotMatchedInsertAll()
            .execute())
        print(f"  MERGE complete — new ids inserted, existing ids skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loader Read Stream

# COMMAND ----------

# Determine whether JSON keys are correct or broken (set during pre-flight)
# keys_match is set by the pre-flight cell above
if keys_match:
    # Keys already match DDL — read directly with schema
    STREAM_SCHEMA = BRONZE_SCHEMA_DDL
    print("Stream schema         : using BRONZE_SCHEMA_DDL directly (keys match)")
else:
    # Keys are broken (first data row used as header by csv_to_json)
    # Build a DDL using the actual broken key names so Auto Loader can parse them
    broken_cols   = df_raw_check.columns   # actual keys from pre-flight
    STREAM_SCHEMA = ", ".join([f"`{c}` STRING" for c in broken_cols])
    print(f"Stream schema         : using BROKEN key names (will rename in foreachBatch)")
    print(f"  Broken keys: {broken_cols}")

print(f"  inferSchema          : false (explicit DDL)")
print(f"  Audit columns        : load_dt, source_file")
print(f"  Null guard           : id IS NOT NULL")
print(f"  Idempotency          : MERGE on id")

df_stream = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format",                "json")
    .option("cloudFiles.schemaLocation",         SCHEMA_LOC)
    .option("cloudFiles.useIncrementalListing",  "auto")
    .option("pathGlobFilter",                    "*.json")
    .option("multiLine",                         "true")      # JSON array format [{},...{}]
    .schema(STREAM_SCHEMA)                                    # inferSchema = false
    .load(ISOLATED_SRC)
    .withColumn("load_dt",     F.current_timestamp())         # Mandatory audit
    .withColumn("source_file", F.col("_metadata.file_path"))  # Mandatory audit
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Stream

# COMMAND ----------

print(f"Starting stream → {TARGET_TABLE} ...")

query = (df_stream.writeStream
    .foreachBatch(upsert_to_bronze)
    .option("checkpointLocation", CHECKPOINT_LOC)
    .trigger(availableNow=True)
    .start())

query.awaitTermination()
print("Stream complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification & Audit

# COMMAND ----------

if not spark.catalog.tableExists(TARGET_TABLE):
    raise Exception(
        f"Table still not found after stream: {TARGET_TABLE}\n"
        f"Check that upsert_to_bronze fired — look for 'Batch 0' in the output above.\n"
        f"If no batches printed, the isolated_json_source folder was empty."
    )

df_bronze      = spark.table(TARGET_TABLE)
total          = df_bronze.count()
null_id        = df_bronze.filter(F.col("id").isNull()).count()
null_cost      = df_bronze.filter(F.col("cost").isNull()).count()
null_marka     = df_bronze.filter(F.col("marka").isNull()).count()
no_load_dt     = df_bronze.filter(F.col("load_dt").isNull()).count()
no_source_file = df_bronze.filter(F.col("source_file").isNull()).count()

print(f"\n{'='*60}")
print(f"  INGESTION SUMMARY")
print(f"{'='*60}")
print(f"  Table              : {TARGET_TABLE}")
print(f"  Total rows         : {total:,}")
print(f"{'='*60}")
print(f"  NULL CHECKS")
print(f"  null id            : {null_id:,}      (must be 0)")
print(f"  null cost          : {null_cost:,}")
print(f"  null marka         : {null_marka:,}")
print(f"{'='*60}")
print(f"  AUDIT COLUMNS")
print(f"  missing load_dt    : {no_load_dt:,}    (must be 0)")
print(f"  missing source_file: {no_source_file:,} (must be 0)")
print(f"{'='*60}")
print(f"  Schema             : inferSchema = false (explicit DDL)")
print(f"  Idempotency        : MERGE on id")
print(f"  Checkpoint         : {CHECKPOINT_LOC}")
print(f"{'='*60}")

all_pass = (null_id == 0 and no_load_dt == 0 and no_source_file == 0 and total > 0)
if all_pass:
    print(f"  ALL CHECKS PASSED — {total:,} rows ingested cleanly.")
else:
    if total == 0:
        print("  ERROR: 0 rows in table — stream ran but wrote nothing.")
    else:
        print("  WARNING: One or more checks failed — investigate above.")

# COMMAND ----------

display(spark.table(TARGET_TABLE).limit(10))
