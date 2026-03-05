# Databricks notebook source
# MAGIC %md
# MAGIC # Widgets & Configuration

# COMMAND ----------

import subprocess, sys, os, shutil
import pandas as pd

# Setup widgets for dynamic execution
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Target Catalog Name")
dbutils.widgets.text("raw_schema", "raw", "2. Raw Schema Name")
dbutils.widgets.text("landing_volume", "landing", "3. Source Volume (Landing)")
dbutils.widgets.text("chunks_volume", "chunks", "4. Target Volume (Chunks)")

# Fetch values into variables
CATALOG = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
LANDING_VOL = dbutils.widgets.get("landing_volume")
CHUNKS_VOL = dbutils.widgets.get("chunks_volume")

# Dynamic Path Construction using Unity Catalog Volumes
LANDING_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{LANDING_VOL}"
CHUNKS_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{CHUNKS_VOL}"
SOURCE_FILE = f"{LANDING_PATH}/1_main.csv"
UTILITIES = LANDING_PATH  # Scripts are stored in the landing volume

# COMMAND ----------

# MAGIC %md
# MAGIC # Logic for chunking 

# COMMAND ----------

# Section 2: Logic 
print("Checking input file...")
if not os.path.exists(SOURCE_FILE):
    print(f"ERROR: Input CSV not found: {SOURCE_FILE}")
    raise FileNotFoundError(f"Input CSV missing: {SOURCE_FILE}")
if os.path.getsize(SOURCE_FILE) == 0:
    print(f"ERROR: Input CSV is empty: {SOURCE_FILE}")
    raise Exception(f"Input CSV is empty: {SOURCE_FILE}")
print(f"✓ Input CSV exists: {SOURCE_FILE} ({os.path.getsize(SOURCE_FILE):,} bytes)")

print("Running csv_splitter.py...")
result = subprocess.run(
    [
        sys.executable,
        f"{UTILITIES}/csv_splitter.py",
        SOURCE_FILE,
        "--percentages", "50", "20", "20", "10",
        "--output-dir", LANDING_PATH,
        "--header", "auto"
    ],
    capture_output=True, text=True
)
print("csv_splitter.py STDOUT:", result.stdout)
print("csv_splitter.py STDERR:", result.stderr)
if result.returncode != 0:
    print("csv_splitter.py failed:", result.stderr)
    raise Exception(f"csv_splitter.py failed: {result.stderr}")

print("Checking chunk CSV files...")
for i in range(1, 5):
    f = f"{LANDING_PATH}/1_main_chunk_{i}.csv"
    if not os.path.exists(f):
        print(f"ERROR: Missing chunk {f}")
        raise FileNotFoundError(f"Missing chunk: {f}")
    print(f"✓ 1_main_chunk{i}.csv — {os.path.getsize(f):,} bytes")

# STEP 2: Convert Chunk 3 → JSON
print("\nConverting chunk3 to JSON...")
json_input = f"{LANDING_PATH}/1_main_chunk_3.csv"
json_output = f"{LANDING_PATH}/1_main_chunk_3.json"
if not os.path.exists(json_input):
    print(f"ERROR: Input CSV not found: {json_input}")
    raise FileNotFoundError(f"Chunk3 CSV does not exist: {json_input}")
result_json = subprocess.run(
    [sys.executable, f"{UTILITIES}/csv_to_json.py", json_input, json_output],
    capture_output=True, text=True
)
print("csv_to_json.py STDOUT:", result_json.stdout)
if result_json.returncode != 0:
    print("csv_to_json.py failed:", result_json.stderr)
    raise Exception(f"csv_to_json.py failed: {result_json.stderr}")
if not os.path.exists(json_output):
    print(f"ERROR: JSON file was not created: {json_output}")
    raise FileNotFoundError("JSON file not created!")
print(f"✓ 1_main_chunk3.json — {os.path.getsize(json_output):,} bytes")

# STEP 3: Convert Chunk 4 → XML
print("\nPreparing chunk4 for XML conversion...")
chunk4_path = f"{LANDING_PATH}/1_main_chunk_4.csv"
if not os.path.exists(chunk4_path):
    print(f"ERROR: Input CSV for XML not found: {chunk4_path}")
    raise FileNotFoundError(f"Chunk4 CSV does not exist: {chunk4_path}")

df_chunk4 = pd.read_csv(chunk4_path, encoding="utf-8")
df_chunk4.columns = [
    c.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "").replace(".", "_")
    for c in df_chunk4.columns
]
for col_name in df_chunk4.select_dtypes(include='object').columns:
    df_chunk4[col_name] = (
        df_chunk4[col_name]
            .astype(str)
            .str.replace('&', 'and', regex=False)
            .str.replace('<', '', regex=False)
            .str.replace('>', '', regex=False)
    )

clean_chunk4_path = f"{LANDING_PATH}/1_main_chunk4_clean.csv"
df_chunk4.to_csv(clean_chunk4_path, index=False, encoding="utf-8")
print(f"✓ Cleaned chunk4 CSV saved. Columns: {list(df_chunk4.columns)}")

xml_output = f"{LANDING_PATH}/1_main_chunk_4.xml"
result_xml = subprocess.run(
    [sys.executable, f"{UTILITIES}/csv_to_xml.py", clean_chunk4_path, xml_output],
    capture_output=True, text=True
)
print("csv_to_xml.py STDOUT:", result_xml.stdout)
if result_xml.returncode != 0:
    print("csv_to_xml.py failed:", result_xml.stderr)
    raise Exception(f"csv_to_xml.py failed: {result_xml.stderr}")
if not os.path.exists(xml_output):
    print(f"ERROR: XML file was not created: {xml_output}")
    raise FileNotFoundError("XML file not created!")
print(f"✓ 1_main_chunk4.xml — {os.path.getsize(xml_output):,} bytes")

# STEP 4: Organize all chunks into /chunks volume
copy_map = {
    f"{LANDING_PATH}/1_main_chunk_1.csv": f"{CHUNKS_PATH}/1_main_chunk_1.csv",
    f"{LANDING_PATH}/1_main_chunk_2.csv": f"{CHUNKS_PATH}/1_main_chunk_2.csv",
    json_output: f"{CHUNKS_PATH}/1_main_chunk_3.json",
    xml_output: f"{CHUNKS_PATH}/1_main_chunk_4.xml",
}
for src, dst in copy_map.items():
    if not os.path.exists(src):
        print(f"ERROR: Source file missing for copy: {src}")
        raise FileNotFoundError(f"Copy source missing: {src}")
    shutil.copy(src, dst)
    print(f"✓ Copied {os.path.basename(src)} → chunks volume")

# STEP 5: CLEANUP (Delete intermediate files in /landing)
print("\nCleaning up intermediate files in /landing...")
files_to_delete = [
    f"{LANDING_PATH}/1_main_chunk_1.csv",
    f"{LANDING_PATH}/1_main_chunk_2.csv",
    f"{LANDING_PATH}/1_main_chunk_3.csv",
    f"{LANDING_PATH}/1_main_chunk_3.json",
    f"{LANDING_PATH}/1_main_chunk_4.csv",
    f"{LANDING_PATH}/1_main_chunk4_clean.csv",
    f"{LANDING_PATH}/1_main_chunk_4.xml"
]

for file_path in files_to_delete:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   Deleted intermediate: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"   Warning: Could not delete {file_path}: {e}")

# Additional Check: Purge anything with '_chunk_' in landing
for f in os.listdir(LANDING_PATH):
    if "_chunk_" in f and f.endswith(('.csv', '.json', '.xml')):
        try:
            full_p = os.path.join(LANDING_PATH, f)
            if os.path.exists(full_p):
                os.remove(full_p)
                print(f"   Purged leftover: {f}")
        except:
            pass

print("\n✓ Pipeline Complete! Chunks safe in: {CHUNKS_PATH}")
