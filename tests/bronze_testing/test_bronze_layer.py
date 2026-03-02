import pytest
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from databricks.connect import DatabricksSession

@pytest.fixture(scope="session")
def spark():
    """Initialize Databricks Session for CI/CD"""
    return DatabricksSession.builder.getOrCreate()

# =========================
# CONFIGURATION
# =========================
CATALOG = "vstone_catalog"
BRONZE = "bronze"
CHUNKS_PATH = f"/Volumes/{CATALOG}/raw/chunks"
LANDING_PATH = f"/Volumes/{CATALOG}/raw/landing"

# Replicating your reconciliation map exactly
TEST_CONFIG = [
    {"name": "listings_csv_copyinto", "src": f"{CHUNKS_PATH}/1_main_chunk_1.csv", "fmt": "csv", "opts": {"header": "true"}},
    {"name": "listings_csv_dlt", "src": f"{CHUNKS_PATH}/1_main_chunk_2.csv", "fmt": "csv", "opts": {"header": "true"}},
    {"name": "listings_json_autoloader", "src": f"{CHUNKS_PATH}/1_main_chunk_3.json", "fmt": "json", "opts": {"multiLine": "true"}},
    {"name": "listings_xml_pyspark", "src": f"{CHUNKS_PATH}/1_main_chunk_4.xml", "fmt": "xml", "opts": {"rowTag": "record"}},
    {"name": "listings_text_bronze", "src": f"{LANDING_PATH}/1_text.csv", "fmt": "csv", "opts": {"header": "true", "multiLine": "true", "escape": '"'}},
    {"name": "listings_photo_bronze", "src": f"{LANDING_PATH}/1_photo.csv", "fmt": "csv", "opts": {"header": "true"}},
    {"name": "car_catalog_bronze", "src": f"{LANDING_PATH}/catalogs.csv", "fmt": "csv", "opts": {"header": "true", "sep": ";"}},
    {"name": "geo_locations_bronze", "src": f"{LANDING_PATH}/final_geografic.csv", "fmt": "csv", "opts": {"header": "true"}}
]

# =========================
# TESTS
# =========================

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_bronze_table_exists(spark, cfg):
    """Verify Bronze tables exist in Unity Catalog"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    assert spark.catalog.tableExists(table_fullname), f"Table missing: {table_fullname}"

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_volume_reconciliation(spark, cfg):
    """Volume & Completeness Test (Count Check)"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    
    # 1. Get Source Count
    raw_df = spark.read.format(cfg["fmt"]).options(**cfg["opts"]).load(cfg["src"])
    raw_count = raw_df.count()
    
    # 2. Get Bronze Table Count
    bronze_count = spark.table(table_fullname).count()
    
    assert raw_count == bronze_count, f"Count mismatch for {cfg['name']}: Raw {raw_count} != Bronze {bronze_count}"

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_row_level_integrity(spark, cfg):
    """Detailed Integrity Check using Fingerprinting (SHA-256)"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    
    # A. Source Data (Raw) - No inferSchema for string-only comparison
    src_df = spark.read.format(cfg["fmt"]).options(**cfg["opts"]).option("inferSchema", "false").load(cfg["src"])
    
    # B. Bronze Data (Target)
    brz_df = spark.table(table_fullname)
    
    # Identify common columns
    common_cols = [c for c in src_df.columns if c in brz_df.columns]
    assert len(common_cols) > 0, f"No common columns for {cfg['name']}"

    def get_fingerprints(df, columns):
        # Normalization logic: Cast to string, Trim, Handle NULLs
        temp_df = df.select([
            F.coalesce(F.trim(F.col(c).cast("string")), F.lit("")).alias(c) 
            for c in columns
        ])
        return temp_df.withColumn("fp", F.sha2(F.concat_ws("||", *columns), 256)).select("fp")

    src_fp = get_fingerprints(src_df, common_cols)
    brz_fp = get_fingerprints(brz_df, common_cols)

    # Reconciliation using subtract
    missing_in_bronze = src_fp.subtract(brz_fp).count()
    extra_in_bronze = brz_fp.subtract(src_fp).count()
    mismatch_total = missing_in_bronze + extra_in_bronze

    assert mismatch_total == 0, f"Integrity Failure in {cfg['name']}: {missing_in_bronze} missing, {extra_in_bronze} extra rows."

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_metadata_audit_and_schema(spark, cfg):
    """Audit Metadata (load_dt/source_file) and Schema Validation"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    df = spark.table(table_fullname)
    cols = df.columns

    # 1. Russian Character Support for Catalog
    if "car_catalog_bronze" in cfg["name"]:
        russian_cols = [c for c in cols if any(ord(char) > 127 for char in c)]
        assert len(russian_cols) > 0, "No Russian headers found in car_catalog_bronze"

    # 2. Audit Metadata Populated
    expected_audit = ["load_dt", "source_file"]
    for audit_col in expected_audit:
        assert audit_col in cols, f"Missing audit column: {audit_col}"
        null_count = df.filter(F.col(audit_col).isNull()).count()
        assert null_count == 0, f"Audit column {audit_col} has {null_count} nulls in {cfg['name']}"

    # 3. Rescued Data Integrity
    if "_rescued_data" in cols:
        rescued_count = df.filter(F.col("_rescued_data").isNotNull()).count()
        assert rescued_count == 0, f"Schema Integrity Alert: {rescued_count} records in _rescued_data for {cfg['name']}"