# Databricks notebook source
# MAGIC %md
# MAGIC # Widgets Configuration

# COMMAND ----------

# 1. Widgets fetch values from DABs 'base_parameters'
dbutils.widgets.text("catalog_name", "vstone_catalog")
dbutils.widgets.text("raw_schema", "raw")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")

# 2. Variable Assignment
CATALOG = dbutils.widgets.get("catalog_name")
RAW_SCHEMA = dbutils.widgets.get("raw_schema")
BRONZE = dbutils.widgets.get("bronze_schema")
SILVER = dbutils.widgets.get("silver_schema")
GOLD = dbutils.widgets.get("gold_schema")

# COMMAND ----------

# Modular list for creation loop
ALL_LAYERS = [RAW_SCHEMA, BRONZE, SILVER, GOLD, "security"]

print(f"INFO: DABs parameters received. Initializing {CATALOG}...")

# 3. Execution Logic (Clean & Simple)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")

for layer in ALL_LAYERS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {layer}")
    print(f"SUCCESS: Layer '{layer}' created in {CATALOG}")

# 4. Volume Setup in Raw
for vol in ["landing", "chunks", "checkpoints"]:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {RAW_SCHEMA}.{vol}")
    print(f"SUCCESS: Volume {vol} ready.")
