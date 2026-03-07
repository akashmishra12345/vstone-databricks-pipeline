# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Data Chunking
# MAGIC
# MAGIC Splits `1_main.csv` (1,083,269 rows) from the landing volume into 4 chunks
# MAGIC stored in the chunks volume. Intermediate CSVs are deleted after conversion
# MAGIC so the chunks volume contains **exactly 4 files**.
# MAGIC
# MAGIC | Chunk              | Format | Split | Rows      | Target Notebook      |
# MAGIC |--------------------|--------|-------|-----------|----------------------|
# MAGIC | 1_main_chunk_1.csv | CSV    | 50%   | 541,634   | COPY INTO            |
# MAGIC | 1_main_chunk_2.csv | CSV    | 20%   | 216,653   | DLT                  |
# MAGIC | 1_main_chunk_3.json| JSON   | 20%   | 216,653   | Auto Loader          |
# MAGIC | 1_main_chunk_4.xml | XML    | 10%   | 108,329   | PySpark XML          |
# MAGIC
# MAGIC > chunk_4 absorbs the rounding remainder so sum always equals source row count.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets & Configuration

# COMMAND ----------

dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Catalog")
dbutils.widgets.text("raw_schema",      "raw",            "2. Raw Schema")
dbutils.widgets.text("chunks_volume",   "chunks",         "3. Chunks Volume")
dbutils.widgets.text("landing_volume",  "landing",        "4. Landing Volume")

CATALOG      = dbutils.widgets.get("project_catalog")
RAW_SCHEMA   = dbutils.widgets.get("raw_schema")
CHUNKS_VOL   = dbutils.widgets.get("chunks_volume")
LANDING_VOL  = dbutils.widgets.get("landing_volume")

LANDING_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{LANDING_VOL}"
CHUNKS_PATH  = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{CHUNKS_VOL}"
SOURCE_CSV   = f"{LANDING_PATH}/1_main.csv"
SPLIT_PCT    = [50, 20, 20, 10]   # Must sum to 100

print(f"Landing path : {LANDING_PATH}")
print(f"Chunks path  : {CHUNKS_PATH}")
print(f"Source file  : {SOURCE_CSV}")
print(f"Split %%      : {SPLIT_PCT}  (sum={sum(SPLIT_PCT)})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import os
import csv
import json
import shutil
import importlib
import xml.etree.ElementTree as ET

# Only csv_splitter is imported as a module.
# csv_to_json and csv_to_xml are inlined directly to avoid
# importlib.reload() attribute resolution issues on Databricks clusters.
import src.utils.csv_splitter as csv_splitter
csv_splitter = importlib.reload(csv_splitter)

print(f"csv_splitter loaded : {csv_splitter.__file__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-Flight: Verify Source & Show Expected Split

# COMMAND ----------

if not os.path.exists(SOURCE_CSV):
    raise FileNotFoundError(
        f"Source file not found: {SOURCE_CSV}\n"
        f"Upload 1_main.csv to the landing volume first."
    )

with open(SOURCE_CSV, 'r', encoding='utf-8') as f:
    total_rows = sum(1 for _ in f) - 1   # subtract header row

# Calculate expected rows per chunk (last chunk gets remainder)
expected = []
for i, pct in enumerate(SPLIT_PCT):
    if i == 3:
        expected.append(total_rows - sum(expected))
    else:
        expected.append(int(total_rows * pct / 100))

print(f"Source file  : {SOURCE_CSV}")
print(f"Total rows   : {total_rows:,}")
print()
print(f"  {'Chunk':<24} {'Split':>5}  {'Expected rows':>14}")
print(f"  {'-'*48}")
names = ["chunk_1.csv", "chunk_2.csv", "chunk_3.json", "chunk_4.xml"]
for name, pct, exp in zip(names, SPLIT_PCT, expected):
    print(f"  1_main_{name:<18} {pct:>4}%  {exp:>14,}")
print(f"  {'-'*48}")
print(f"  {'TOTAL':<24} {'100%':>5}  {sum(expected):>14,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Split `1_main.csv` into 4 CSV Chunks
# MAGIC
# MAGIC `csv_splitter` writes the **header row into every chunk** (`has_header=True`).
# MAGIC This ensures `csv.DictReader` in Steps 2 & 3 always sees correct column names.

# COMMAND ----------

print("=" * 60)
print(f"  STEP 1 : Splitting 1_main.csv  [{SPLIT_PCT[0]}% / {SPLIT_PCT[1]}% / {SPLIT_PCT[2]}% / {SPLIT_PCT[3]}%]")
print("=" * 60)

csv_splitter.split_csv(
    input_file  = SOURCE_CSV,
    percentages = SPLIT_PCT,
    output_dir  = CHUNKS_PATH,
    has_header  = True,          # Always explicit — never auto-detect in production
    delimiter   = ","
)

# Confirm all 4 CSV chunks exist after split
print()
for i in range(1, 5):
    chunk_path = f"{CHUNKS_PATH}/1_main_chunk_{i}.csv"
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Split failed — chunk not found: {chunk_path}")
    row_count = sum(1 for _ in open(chunk_path, encoding='utf-8')) - 1
    print(f"  1_main_chunk_{i}.csv  created  {row_count:>10,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Convert `chunk_3.csv` → `chunk_3.json`
# MAGIC
# MAGIC Uses `csv.DictReader` which reads column names from the header row.
# MAGIC Produces a JSON array where every object has keys:
# MAGIC `cost, currency, marka, model, year, has_license, place, date,`
# MAGIC `id, engine, power, gear, probeg, sWheel, complectation, transmission, R, G, B`

# COMMAND ----------

print("=" * 60)
print("  STEP 2 : chunk_3.csv  →  chunk_3.json")
print("=" * 60)

CHUNK_3_CSV  = f"{CHUNKS_PATH}/1_main_chunk_3.csv"
CHUNK_3_JSON = f"{CHUNKS_PATH}/1_main_chunk_3.json"

# Remove stale JSON from any previous run before writing fresh
if os.path.exists(CHUNK_3_JSON):
    os.remove(CHUNK_3_JSON)
    print(f"  Removed stale : {CHUNK_3_JSON}")

with open(CHUNK_3_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data   = list(reader)

if not data:
    raise ValueError(
        f"No records in {CHUNK_3_CSV}.\n"
        f"Re-run Step 1 — csv_splitter must use has_header=True."
    )

with open(CHUNK_3_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Key validation — catch header issues immediately
EXPECTED_KEYS = {"cost", "currency", "marka", "model", "year", "has_license",
                 "place", "date", "id", "engine", "power", "gear", "probeg",
                 "sWheel", "complectation", "transmission", "R", "G", "B"}

actual_keys = set(data[0].keys())
bad_keys    = actual_keys - EXPECTED_KEYS

if bad_keys:
    raise ValueError(
        f"Unexpected JSON keys detected: {bad_keys}\n"
        f"chunk_3.csv header row is missing — re-run Step 1."
    )

print(f"  Records written  : {len(data):,}")
print(f"  Output           : {CHUNK_3_JSON}")
print(f"  Key validation   : PASSED ({len(actual_keys)} columns)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Convert `chunk_4.csv` → `chunk_4.xml`

# COMMAND ----------

print("=" * 60)
print("  STEP 3 : chunk_4.csv  →  chunk_4.xml")
print("=" * 60)

CHUNK_4_CSV = f"{CHUNKS_PATH}/1_main_chunk_4.csv"
CHUNK_4_XML = f"{CHUNKS_PATH}/1_main_chunk_4.xml"

if os.path.exists(CHUNK_4_XML):
    os.remove(CHUNK_4_XML)
    print(f"  Removed stale : {CHUNK_4_XML}")

with open(CHUNK_4_CSV, 'r', encoding='utf-8') as f:
    reader   = csv.DictReader(f)
    xml_rows = list(reader)

if not xml_rows:
    raise ValueError(f"No records in {CHUNK_4_CSV}. Re-run Step 1.")

xml_root = ET.Element("data")
for row in xml_rows:
    record = ET.SubElement(xml_root, "record")
    for key, value in row.items():
        elem      = ET.SubElement(record, key)
        elem.text = value

ET.indent(xml_root, space="  ")
ET.ElementTree(xml_root).write(CHUNK_4_XML, encoding='utf-8', xml_declaration=True)

print(f"  Records written  : {len(xml_rows):,}")
print(f"  Output           : {CHUNK_4_XML}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Cleanup: Remove Intermediate CSVs
# MAGIC
# MAGIC After conversion, `chunk_3.csv` and `chunk_4.csv` are deleted.
# MAGIC The chunks volume must contain **exactly 4 files**:
# MAGIC `chunk_1.csv`, `chunk_2.csv`, `chunk_3.json`, `chunk_4.xml`

# COMMAND ----------

print("=" * 60)
print("  STEP 4 : Cleanup — removing intermediate CSVs")
print("=" * 60)

for f_path in [CHUNK_3_CSV, CHUNK_4_CSV]:
    if os.path.exists(f_path):
        os.remove(f_path)
        print(f"  Deleted : {f_path}")

# List only files (not subdirectories) in chunks root
remaining = sorted([
    f for f in os.listdir(CHUNKS_PATH)
    if os.path.isfile(os.path.join(CHUNKS_PATH, f))
])

print(f"\n  Files in chunks volume after cleanup ({len(remaining)}):")
for fname in remaining:
    fpath  = os.path.join(CHUNKS_PATH, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"    {fname:<30}  {size_mb:>8.2f} MB")

if len(remaining) != 4:
    raise Exception(
        f"Expected exactly 4 files in chunks volume, found {len(remaining)}: {remaining}"
    )
print(f"\n  Chunks volume is clean — exactly 4 files confirmed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Stage JSON for Auto Loader
# MAGIC
# MAGIC Auto Loader reads from `isolated_json_source/` to prevent ghost metadata
# MAGIC from the chunks root polluting the stream state.

# COMMAND ----------

print("=" * 60)
print("  STEP 5 : Staging JSON for Auto Loader")
print("=" * 60)

ISOLATED_SRC  = f"{CHUNKS_PATH}/isolated_json_source/"
ISOLATED_JSON = f"{ISOLATED_SRC}1_main_chunk_3.json"

os.makedirs(ISOLATED_SRC, exist_ok=True)
shutil.copy2(CHUNK_3_JSON, ISOLATED_JSON)

print(f"  Staged at : {ISOLATED_JSON}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

def _count_csv(path):
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in f) - 1

def _count_json(path):
    with open(path, encoding='utf-8') as f:
        return len(json.load(f))

def _count_xml(path):
    return len(ET.parse(path).getroot().findall("record"))

final_files = [
    (f"{CHUNKS_PATH}/1_main_chunk_1.csv",  "50%", "COPY INTO",   _count_csv),
    (f"{CHUNKS_PATH}/1_main_chunk_2.csv",  "20%", "DLT",         _count_csv),
    (f"{CHUNKS_PATH}/1_main_chunk_3.json", "20%", "Auto Loader", _count_json),
    (f"{CHUNKS_PATH}/1_main_chunk_4.xml",  "10%", "PySpark XML", _count_xml),
]

total_check = 0
rows_per_chunk = []

for fpath, pct, target, counter in final_files:
    rc = counter(fpath) if os.path.exists(fpath) else 0
    rows_per_chunk.append(rc)
    total_check += rc

row_match = total_check == total_rows

print(f"\n{'='*65}")
print(f"  CHUNKING COMPLETE — 1_main.csv  [{SPLIT_PCT[0]}% / {SPLIT_PCT[1]}% / {SPLIT_PCT[2]}% / {SPLIT_PCT[3]}%]")
print(f"{'='*65}")
print(f"  Source         : {SOURCE_CSV}")
print(f"  Source rows    : {total_rows:,}")
print()
print(f"  {'File':<30} {'Split':>5}  {'Rows':>10}  {'Target'}")
print(f"  {'-'*62}")

for (fpath, pct, target, _), rc in zip(final_files, rows_per_chunk):
    fname    = os.path.basename(fpath)
    actual_p = rc / total_rows * 100 if total_rows else 0
    print(f"  {fname:<30} {pct:>5}  {rc:>10,}  {target}  ({actual_p:.2f}%)")

print(f"  {'-'*62}")
print(f"  {'TOTAL':<30} {'100%':>5}  {total_check:>10,}")
print()
print(f"  Row integrity  : {'PASSED — all {total_rows:,} rows accounted for'.format(total_rows=total_rows) if row_match else 'FAILED — mismatch detected!'}")
print(f"  Auto Loader    : isolated_json_source/1_main_chunk_3.json  staged")
print(f"{'='*65}")
