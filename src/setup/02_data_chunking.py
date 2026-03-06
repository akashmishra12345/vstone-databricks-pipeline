# Databricks notebook source
# MAGIC %md
# MAGIC # Modular Imports & Configuration

# COMMAND ----------

# Notebook: src/notebooks/setup/02_data_chunking.py
import os
import sys
import shutil
import pandas as pd
import importlib

# 1. MODULAR IMPORTS 
import src.utils.csv_to_json as csv_to_json
import src.utils.csv_to_xml as csv_to_xml
import src.utils.csv_splitter as csv_splitter

# Force reload taaki development ke waqt latest changes milte rahein
importlib.reload(csv_to_json)
importlib.reload(csv_to_xml)
importlib.reload(csv_splitter)

# 2. CONFIGURATION VIA WIDGETS
dbutils.widgets.text("project_catalog", "vstone_catalog")
dbutils.widgets.text("raw_schema", "raw")
dbutils.widgets.text("landing_volume", "landing")
dbutils.widgets.text("chunks_volume", "chunks")

CATALOG = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
LANDING_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{dbutils.widgets.get('landing_volume')}"
CHUNKS_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{dbutils.widgets.get('chunks_volume')}"
SOURCE_FILE = f"{LANDING_PATH}/1_main.csv"

print(f"INFO: Pipeline initialized for Catalog: {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 1 - CSV Data Splitting

# COMMAND ----------

# --------------------------------------------------------------------------------------
# STEP 1: SPLIT CSV (Logical Distribution)
# Logic: 50% CSV, 20% CSV (Incremental), 20% JSON, 10% XML
# --------------------------------------------------------------------------------------
print("INFO: Starting Step 1 - Splitting main CSV...")

if not os.path.exists(SOURCE_FILE):
    raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

try:
    csv_splitter.split_csv(
        input_file=SOURCE_FILE,
        percentages=[50, 20, 20, 10],
        output_dir=LANDING_PATH
    )
    print("SUCCESS: Data split into 4 chunks.")
except Exception as e:
    print(f"ERROR: Splitting failed: {e}")
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 2 - JSON Conversion (Chunk 3)

# COMMAND ----------

# --------------------------------------------------------------------------------------
# STEP 2: FORMAT CONVERSIONS (JSON)
# --------------------------------------------------------------------------------------
chunk3_csv = f"{LANDING_PATH}/1_main_chunk_3.csv"
chunk3_json = f"{LANDING_PATH}/1_main_chunk_3.json"

print(f"INFO: Converting {os.path.basename(chunk3_csv)} to JSON...")
try:
    csv_to_json.convert(chunk3_csv, chunk3_json)
    print("SUCCESS: JSON conversion successful.")
except Exception as e:
    print(f"ERROR: JSON conversion failed: {e}")
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 3 - XML Conversion (Chunk 4) 

# COMMAND ----------

# --------------------------------------------------------------------------------------
# STEP 3: FORMAT CONVERSIONS (XML)
# --------------------------------------------------------------------------------------
print("\nINFO: Preparing chunk4 for XML conversion...")
chunk4_csv = f"{LANDING_PATH}/1_main_chunk_4.csv"
chunk4_xml = f"{LANDING_PATH}/1_main_chunk_4.xml"

try:
    # Column cleaning for XML tag compatibility
    df_chunk4 = pd.read_csv(chunk4_csv, encoding="utf-8")
    df_chunk4.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df_chunk4.columns]
    
    # Direct Call to Utility
    csv_to_xml.convert(df_chunk4, chunk4_xml)
    print(f"SUCCESS: XML created: {os.path.basename(chunk4_xml)}")
except Exception as e:
    print(f"ERROR: XML conversion failed: {e}")
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 4 - Consolidate to Chunks Volume

# COMMAND ----------

# --------------------------------------------------------------------------------------
# STEP 4: ORGANIZE CHUNKS INTO /CHUNKS VOLUME
# --------------------------------------------------------------------------------------
print("\nINFO: Consolidating files to Chunks Volume...")

copy_map = {
    f"{LANDING_PATH}/1_main_chunk_1.csv": f"{CHUNKS_PATH}/1_main_chunk_1.csv",
    f"{LANDING_PATH}/1_main_chunk_2.csv": f"{CHUNKS_PATH}/1_main_chunk_2.csv",
    chunk3_json: f"{CHUNKS_PATH}/1_main_chunk_3.json",
    chunk4_xml: f"{CHUNKS_PATH}/1_main_chunk_4.xml",
}

for src, dst in copy_map.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  SUCCESS: Copied {os.path.basename(src)}")
    else:
        print(f"  WARNING: Source missing: {src}")

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 5 - Workspace Cleanup

# COMMAND ----------

# --------------------------------------------------------------------------------------
# STEP 5: CLEANUP (Workspace Hygiene)
# --------------------------------------------------------------------------------------
print("\nINFO: Cleaning up intermediate files in Landing Zone...")

for f in os.listdir(LANDING_PATH):
    if "_chunk_" in f:
        file_to_del = os.path.join(LANDING_PATH, f)
        os.remove(file_to_del)
        print(f"  Deleted: {f}")

print(f"\n✅ PIPELINE SUCCESSFUL: All chunks ready in {CHUNKS_PATH}")
