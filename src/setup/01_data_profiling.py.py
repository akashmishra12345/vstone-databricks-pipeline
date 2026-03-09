# Databricks notebook source
# MAGIC %md
# MAGIC # Imports and Configuration

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import (
    col, count, when, isnull, trim, length, countDistinct, 
    min, max, avg, round as spark_round, desc
)

# Setup widgets for environment-agnostic paths
dbutils.widgets.text("project_catalog", "vstone_catalog", "1. Catalog Name")
dbutils.widgets.text("raw_schema", "raw", "2. Raw Schema")
dbutils.widgets.text("landing_volume", "landing", "3. Landing Volume")

# Fetch values into variables
CATALOG = dbutils.widgets.get("project_catalog")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
LANDING_VOL = dbutils.widgets.get("landing_volume")

# Construct the dynamic landing path using Unity Catalog syntax
LANDING_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{LANDING_VOL}"

# File Manifest with original format and options
FILES = {
    "1_main": {"path": f"{LANDING_PATH}/1_main.csv", "format": "csv", "opts": {"sep": ","}},
    "catalogs": {"path": f"{LANDING_PATH}/catalogs.csv", "format": "csv", "opts": {"sep": ";"}},
    "geolocation": {"path": f"{LANDING_PATH}/final_geografic.csv", "format": "csv", "opts": {"sep": ","}},
    "text": {"path": f"{LANDING_PATH}/1_text.csv", "format": "csv", "opts": {"sep": ",", "multiLine": "true", "escape": '"'}},
    "photos": {"path": f"{LANDING_PATH}/1_photo.csv", "format": "csv", "opts": {"sep": ","}},
}

print(f"INFO: Data Profiling initialized for Landing Path: {LANDING_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Audit Function Definition

# COMMAND ----------

def profile_file(name, path, fmt, options):
    print(f"\n{'='*95}\n DATA AUDIT REPORT: {name.upper()}\n{'='*95}")
    
    # READ: Load all columns as string for initial profiling (string-safe)
    df = (spark.read
          .option("header", True)
          .option("inferSchema", False) 
          .option("encoding", "UTF-8")
          .options(**options)
          .format(fmt)
          .load(path))
    
    total_rows = df.count()
    if total_rows == 0: 
        print(f" Warning: File {name} is empty.")
        return None, 0

    # COMPLETE METRICS AUDIT: Nulls, distincts, completeness for each column
    quality_exprs = []
    for c in df.columns:
        quality_exprs.extend([
            count(when(isnull(col(c)) | (trim(col(c)) == ""), c)).alias(f"{c}_nulls"),
            countDistinct(col(c)).alias(f"{c}_distinct")
        ])
    
    audit_results = df.select(quality_exprs).toPandas().transpose()
    audit_results.columns = ["Value"]
    
    profile_rows = []
    for c in df.columns:
        null_count = audit_results.loc[f"{c}_nulls", "Value"]
        distinct_count = audit_results.loc[f"{c}_distinct", "Value"]
        
        profile_rows.append({
            "Column_Name": c,
            "Data_Type": dict(df.dtypes)[c],
            "Total_Count": total_rows,
            "Null_Count": null_count,
            "Null_Percentage": round((null_count / total_rows) * 100, 2),
            "Distinct_Values": distinct_count,
            "Completeness": f"{round(100 - (null_count/total_rows*100), 2)}%"
        })
    
    pdf_final = pd.DataFrame(profile_rows)
    print(f" Dataset Stats: {total_rows:,} Rows | {len(df.columns)} Columns")
    
    # Display audit results with gradient highlighting for null percentage
    try:
        display(pdf_final.style.background_gradient(cmap='YlOrRd', subset=['Null_Percentage']))
    except:
        display(pdf_final)

    # SMART INTEGRITY CHECK: Primary key candidate, duplicate count, uniqueness ratio
    if name == "catalogs":
        pk_cols = ["Марка", "Модель", "Поколение", "Комплектация"]
        distinct_rows = df.select(pk_cols).distinct().count()
        pk_candidate = "Composite (Catalog Schema)"
    else:
        pk_candidate = "id" if "id" in df.columns else df.columns[0]
        distinct_rows = df.select(pk_candidate).distinct().count()
    
    duplicate_count = total_rows - distinct_rows
    
    integrity_summary = pd.DataFrame([
        {"Metric": "Primary Key Candidate", "Status": pk_candidate},
        {"Metric": "Duplicate Records", "Status": f" {duplicate_count:,}" if duplicate_count > 0 else " 0 Duplicates"},
        {"Metric": "Uniqueness Ratio", "Status": f"{round((distinct_rows/total_rows)*100, 2)}%"}
    ])
    display(integrity_summary)
    
    return df, total_rows

# COMMAND ----------

# MAGIC %md
# MAGIC # Automated Audit Execution

# COMMAND ----------

# --- AUTOMATED AUDIT EXECUTION ---
profiled = {}
for name, cfg in FILES.items():
    try:
        df, rows = profile_file(name, cfg["path"], cfg["format"], cfg["opts"])
        profiled[name] = {"df": df}
    except Exception as e:
        print(f" Error profiling {name}: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 1_MAIN Business Logic KPIs

# COMMAND ----------

# --- 1_MAIN KPI SUMMARY ---
if "1_main" in profiled:
    print("\n" + " " * 30 + " 1_MAIN BUSINESS LOGIC KPIs " + " " * 30)
    df_main = profiled["1_main"]["df"]
    
    # Cast 'cost' column to double for numeric analysis
    df_clean = df_main.withColumn("cost_num", col("cost").cast("double"))
    
    # Display average price and total distinct brands
    display(df_clean.select(
        spark_round(avg("cost_num"), 0).alias("Avg_Price_RUB"),
        countDistinct("marka").alias("Total_Brands")
    ))
