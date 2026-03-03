# Databricks notebook source
# =============================================================
# BUSINESS RULE: CURRENCY NORMALIZATION (RE-FIXED)
# =============================================================
from pyspark.sql.functions import col, when, round

# 1. Pehle existing silver table ko read karein (variable error se bachne ke liye)
TABLE_NAME = "vstone_catalog.silver.listings_silver_merged"
df_silver_raw = spark.table(TABLE_NAME)

# 2. Applying Normalization (Exchange Rate: 1 USD = 82.5 RUB)
# 3. Applying Price Categorization
df_silver_enriched = df_silver_raw.withColumn(
    "price_usd", 
    round(col("price_rub") / 82.5, 2)
).withColumn(
    "price_category",
    when(col("price_rub") < 300000, "BUDGET")
    .when(col("price_rub").between(300000, 1000000), "MID_RANGE")
    .otherwise("PREMIUM")
)

# 4. Quick Verification
print("💰 CURRENCY NORMALIZATION COMPLETED")
df_silver_enriched.select(
    "listing_id", "price_rub", "price_usd", "price_category"
).show(5)

# 5. Overwrite back to Silver to save the new columns
df_silver_enriched.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLE_NAME)

print(f"✅ Table Enriched and Saved: {TABLE_NAME}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vstone_catalog.silver.listings_silver_merged limit 10;

# COMMAND ----------

# =============================================================
# DAY 5: DELTA LAKE TIME TRAVEL & ACID EVIDENCE (FIXED)
# =============================================================
CATALOG = "vstone_catalog"
SILVER = "silver"
TABLE = f"{CATALOG}.{SILVER}.listings_silver_merged"

# 1. Professional Audit Trail: Extended Transaction History
print("\n📜 STEP 1: EXTENDED DELTA TRANSACTION HISTORY")
display(
    spark.sql(f"DESCRIBE HISTORY {TABLE}")
    .select(
        "version", 
        "timestamp", 
        "userName",        # User details
        "operation",       # Kya kiya (WRITE/UPDATE/RESTORE)
        "operationParameters", # Kya changes huye (filters/predicates)
        "job",             # Job ID agar automatic hai
        "notebook.notebookId" # Notebook reference
    )
    .orderBy("version", ascending=False)
)

# 2. Time Travel: Accessing Version 0 (Corrected Columns)
print("\n🕰️ STEP 2: TIME TRAVEL - ACCESSING VERSION 0")
try:
    # Version 0 access karte waqt hum wahi columns use karenge jo error mein dikh rahe hain
    df_v0 = spark.read.format("delta").option("versionAsOf", 0).table(TABLE)
    print(f"✅ Version 0 Row Count: {df_v0.count():,}")
    
    # Error fix: 'listing_date' ki jagah 'date' aur 'listing_id' ki jagah 'id' use karein
    # Kyunki Version 0 mein purane naam hain
    df_v0.select("id", "date", "cost").show(5) 
except Exception as e:
    print(f"❌ Version 0 access error: {e}")

# 3. ACID Properties: Simulated Corruption & Restoration
print("\n🛡️ STEP 3: ACID DEMONSTRATION")

# A. Simulated Corruption
spark.sql(f"UPDATE {TABLE} SET brand = 'CORRUPTED_DATA' WHERE year = 2020")
corrupt_cnt = spark.sql(f"SELECT COUNT(*) FROM {TABLE} WHERE brand = 'CORRUPTED_DATA'").collect()[0][0]
print(f"⚠️ Rows marked as CORRUPTED: {corrupt_cnt}")

# B. Identify and Restore
current_version = spark.sql(f"DESCRIBE HISTORY {TABLE}").select("version").first()[0]
clean_version = current_version - 1

print(f"🔄 RESTORE: Reverting table to clean Version {clean_version}...")
spark.sql(f"RESTORE TABLE {TABLE} TO VERSION AS OF {clean_version}")

# C. Verification
restored_check = spark.sql(f"SELECT COUNT(*) FROM {TABLE} WHERE brand = 'CORRUPTED_DATA'").collect()[0][0]
print(f"✅ Recovery Successful! Corrupted rows remaining: {restored_check}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Delta Lake Time Travel using SQL
# MAGIC SELECT id, date, cost 
# MAGIC FROM vstone_catalog.silver.listings_silver_merged 
# MAGIC VERSION AS OF 0 
# MAGIC LIMIT 5;
