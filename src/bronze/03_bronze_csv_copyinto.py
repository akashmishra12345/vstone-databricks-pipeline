# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Bronze: CSV Ingestion via COPY INTO
# MAGIC **Source :** `/Volumes/vstone_catalog/raw/chunks/1_main_chunk_1.csv`
# MAGIC **Target :** `vstone_catalog.bronze.listings_csv_copyinto`
# MAGIC **Method :** `COPY INTO` — idempotent, incremental, file-tracked SQL command
# MAGIC
# MAGIC ### How COPY INTO works in this notebook
# MAGIC ```
# MAGIC Volume (CSV file)
# MAGIC       │
# MAGIC       ▼
# MAGIC ┌─────────────────────────────────────────────┐
# MAGIC │  COPY INTO  (tracks which files it has      │
# MAGIC │  already loaded — re-runs are safe,         │
# MAGIC │  duplicate files are skipped automatically) │
# MAGIC └─────────────────────────────────────────────┘
# MAGIC       │
# MAGIC       ▼
# MAGIC vstone_catalog.bronze.listings_csv_copyinto
# MAGIC       │
# MAGIC       ▼
# MAGIC Null-ID purge  →  Missing-row repair  →  Reconciliation
# MAGIC ```
# MAGIC
# MAGIC ### Steps
# MAGIC | Step | Action |
# MAGIC |------|--------|
# MAGIC | 1 | Widgets & configuration |
# MAGIC | 2 | Create Bronze table with explicit schema |
# MAGIC | 3 | **`COPY INTO`** — ingest CSV into Bronze |
# MAGIC | 4 | Null-ID audit & purge |
# MAGIC | 5 | Missing-row repair (anti-join) |
# MAGIC | 6 | Final reconciliation report |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Widgets & Configuration

# COMMAND ----------

import pyspark.sql.functions as F
from delta.tables import DeltaTable

# ── Widgets ───────────────────────────────────────────────────────────────────
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Catalog")
dbutils.widgets.text("raw_schema",      "raw",            "2. Raw Schema")
dbutils.widgets.text("bronze_schema",   "bronze",         "3. Bronze Schema")
dbutils.widgets.text("file_name",       "1_main_chunk_1.csv", "4. Source File Name")

CATALOG    = dbutils.widgets.get("project_catalog")
RAW        = dbutils.widgets.get("raw_schema")
BRONZE     = dbutils.widgets.get("bronze_schema")
FILE_NAME  = dbutils.widgets.get("file_name")

# Derived paths
FILE_PATH    = f"/Volumes/{CATALOG}/{RAW}/chunks/{FILE_NAME}"
TARGET_TABLE = f"{CATALOG}.{BRONZE}.listings_csv_copyinto"

print(f"Source file  : {FILE_PATH}")
print(f"Target table : {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Create Bronze Table (if not exists)
# MAGIC Explicit DDL — 19 source columns (all `STRING` as ingested from CSV) plus
# MAGIC two audit columns: `load_dt` (when the row landed) and `source_file` (which file it came from).
# MAGIC
# MAGIC `IF NOT EXISTS` makes this cell safe to re-run at any time.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS vstone_catalog.bronze.listings_csv_copyinto (
# MAGIC     cost            STRING,
# MAGIC     currency        STRING,
# MAGIC     marka           STRING,
# MAGIC     model           STRING,
# MAGIC     year            STRING,
# MAGIC     has_license     STRING,
# MAGIC     place           STRING,
# MAGIC     date            STRING,
# MAGIC     id              STRING,
# MAGIC     engine          STRING,
# MAGIC     power           STRING,
# MAGIC     gear            STRING,
# MAGIC     probeg          STRING,
# MAGIC     sWheel          STRING,
# MAGIC     complectation   STRING,
# MAGIC     transmission    STRING,
# MAGIC     R               STRING,
# MAGIC     G               STRING,
# MAGIC     B               STRING,
# MAGIC     load_dt         TIMESTAMP,
# MAGIC     source_file     STRING
# MAGIC )
# MAGIC USING DELTA
# MAGIC TBLPROPERTIES (
# MAGIC     'quality'                    = 'bronze',
# MAGIC     'delta.enableChangeDataFeed' = 'true'
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — COPY INTO
# MAGIC `COPY INTO` is Databricks' idempotent CSV ingestion command:
# MAGIC - Tracks every file it has processed in the table's transaction log
# MAGIC - Re-running this cell on the **same file** is completely safe — already-loaded files are skipped
# MAGIC - `FORCE = FALSE` (default) enforces that safety; set to `TRUE` only if you want to re-ingest
# MAGIC
# MAGIC ### Options used
# MAGIC | Option | Value | Reason |
# MAGIC |--------|-------|--------|
# MAGIC | `FORMAT` | `CSV` | source file type |
# MAGIC | `header` | `true` | first row is column names |
# MAGIC | `inferSchema` | `false` | we own the schema — no surprises |
# MAGIC | `encoding` | `UTF-8` | Cyrillic characters in place/model names |
# MAGIC | `mergeSchema` | `false` | reject schema drift — explicit DDL wins |
# MAGIC | `mode` | `PERMISSIVE` | load all rows, bad values become null |

# COMMAND ----------

copy_sql = f"""
COPY INTO {TARGET_TABLE}
FROM (
    SELECT
        cost,
        currency,
        marka,
        model,
        year,
        has_license,
        place,
        date,
        id,
        engine,
        power,
        gear,
        probeg,
        sWheel,
        complectation,
        transmission,
        R,
        G,
        B,
        current_timestamp() AS load_dt,
        '{FILE_NAME}'        AS source_file
    FROM '{FILE_PATH}'
)
FILEFORMAT = CSV
FORMAT_OPTIONS (
    'header'      = 'true',
    'inferSchema' = 'false',
    'encoding'    = 'UTF-8',
    'mergeSchema' = 'false',
    'mode'        = 'PERMISSIVE'
)
COPY_OPTIONS (
    'force' = 'false'
)
"""

result = spark.sql(copy_sql)
result.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Null-ID Audit & Purge
# MAGIC PERMISSIVE mode loads every row — including rows where the CSV value for `id`
# MAGIC was empty or malformed. These become `NULL` in Bronze.
# MAGIC
# MAGIC Null-ID rows are not real listings. This step identifies and permanently
# MAGIC removes them from Bronze so they do not propagate to Silver.

# COMMAND ----------

df_bronze = spark.table(TARGET_TABLE)
total     = df_bronze.count()
null_id   = df_bronze.filter(F.col("id").isNull()).count()

print(f"Bronze total rows  : {total:,}")
print(f"Rows with null id  : {null_id:,}")

if null_id > 0:
    print("\nNull-id rows (preview before deletion):")
    display(df_bronze.filter(F.col("id").isNull()))
else:
    print("\nNo null-id rows found — table is clean.")

# COMMAND ----------

if null_id > 0:
    spark.sql(f"DELETE FROM {TARGET_TABLE} WHERE id IS NULL")
    after_delete = spark.table(TARGET_TABLE).count()
    print(f"Rows deleted       : {null_id:,}")
    print(f"Bronze rows after  : {after_delete:,}")
else:
    print("Nothing to delete.")
    after_delete = total

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Missing-Row Repair
# MAGIC After purging null-ID rows, verify whether any **valid** source rows are absent
# MAGIC from Bronze entirely (COPY INTO may have skipped rows with parse errors).
# MAGIC
# MAGIC Method: read the source CSV directly, anti-join against Bronze on `id`.
# MAGIC Only rows with a non-null `id` in the source are candidates for re-insertion.

# COMMAND ----------

# Read source CSV directly — no schema imposed, raw strings only
df_source = (
    spark.read
         .option("header",      "true")
         .option("inferSchema", "false")
         .option("encoding",    "UTF-8")
         .csv(FILE_PATH)
)

df_bronze_fresh = spark.table(TARGET_TABLE)
src_count       = df_source.count()
bronze_count    = df_bronze_fresh.count()
gap             = src_count - bronze_count

print(f"Source rows  : {src_count:,}")
print(f"Bronze rows  : {bronze_count:,}")
print(f"Gap          : {gap:,}")

# Anti-join: source ids not present in Bronze at all
df_missing = (
    df_source
        .filter(F.col("id").isNotNull())           # ignore null-id source rows
        .join(
            df_bronze_fresh.select("id"),
            on="id",
            how="left_anti"                         # keep only rows absent from Bronze
        )
)
missing_count = df_missing.count()
print(f"\nValid source rows missing from Bronze: {missing_count:,}")

if missing_count > 0:
    print("\nMissing rows preview:")
    display(df_missing)

# COMMAND ----------

if missing_count > 0:
    print(f"Inserting {missing_count:,} missing row(s) into Bronze...")

    df_insert = (
        df_missing
            .withColumn("load_dt",     F.current_timestamp())
            .withColumn("source_file", F.lit(FILE_NAME))
    )

    (df_insert
        .write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "false")   # schema is fixed — no drift allowed
        .saveAsTable(TARGET_TABLE))

    print("Insert complete.")
else:
    print("No missing rows — Bronze is already complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Final Reconciliation
# MAGIC Confirms Bronze is in the correct final state:
# MAGIC - Row count matches source (excluding null-ID source rows which are malformed)
# MAGIC - Zero null-ID rows remain in Bronze

# COMMAND ----------

df_final    = spark.table(TARGET_TABLE)
final_total = df_final.count()
final_nulls = df_final.filter(F.col("id").isNull()).count()

# Source valid row count = source total minus null-id source rows
src_null_id  = df_source.filter(F.col("id").isNull()).count()
src_valid    = src_count - src_null_id
final_gap    = src_valid - final_total

print(f"\n{'='*55}")
print(f"  FINAL RECONCILIATION")
print(f"{'='*55}")
print(f"  Source total rows        : {src_count:,}")
print(f"  Source null-id rows      : {src_null_id:,}  (excluded — malformed)")
print(f"  Source valid rows        : {src_valid:,}")
print(f"  Bronze rows              : {final_total:,}")
print(f"  Gap (valid - bronze)     : {final_gap:,}  ({'PASSED' if final_gap == 0 else 'MISMATCH'})")
print(f"  Null-id rows in Bronze   : {final_nulls:,}  ({'OK' if final_nulls == 0 else 'MUST BE 0'})")
print(f"  load_dt populated        : {df_final.filter(F.col('load_dt').isNull()).count() == 0}")
print(f"  source_file populated    : {df_final.filter(F.col('source_file').isNull()).count() == 0}")
print(f"{'='*55}")

if final_gap == 0 and final_nulls == 0:
    print("  ✓  ALL CHECKS PASSED")
else:
    print("  ✗  CHECK FAILED — review steps above")
    raise ValueError(
        f"Reconciliation failed: gap={final_gap}, null_id_rows={final_nulls}"
    )
