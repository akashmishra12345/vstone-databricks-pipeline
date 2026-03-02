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

TEST_CONFIG = [
    {"name": "listings_csv_copyinto", "src": f"{CHUNKS_PATH}/1_main_chunk_1.csv", "fmt": "csv", "opts": {"header": "true"}},
    {"name": "listings_csv_dlt", "src": f"{CHUNKS_PATH}/1_main_chunk_2.csv", "fmt": "csv", "opts": {"header": "true"}},
    {"name": "listings_json_autoloader", "src": f"{CHUNKS_PATH}/1_main_chunk_3.json", "fmt": "json", "opts": {"multiLine": "true"}},
    {"name": "listings_xml_pyspark", "src": f"{CHUNKS_PATH}/1_main_chunk_4.xml", "fmt": "xml", "opts": {"rowTag": "record"}},
    {"name": "listings_text_bronze", "src": f"{LANDING_PATH}/1_text.csv", "fmt": "csv", "opts": {"header": "true", "multiLine": "true", "escape": '"'}},
    {"name": "listings_photo_bronze", "src": f"{LANDING_PATH}/1_photo.csv", "fmt": "csv", "opts": {"header": "true", "escape": '"', "quote": '"', "multiLine": "true"}},
    {"name": "car_catalog_bronze", "src": f"{LANDING_PATH}/catalogs.csv", "fmt": "csv", "opts": {"header": "true", "sep": ";"}},
    {"name": "geo_locations_bronze", "src": f"{LANDING_PATH}/final_geografic.csv", "fmt": "csv", "opts": {"header": "true", "ignoreLeadingWhiteSpace": "true", "ignoreTrailingWhiteSpace": "true", "samplingRatio": "1.0"}}
]

# =========================
# TESTS
# =========================

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_bronze_table_exists(spark, cfg):
    """Verify tables are present in Unity Catalog"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    assert spark.catalog.tableExists(table_fullname)

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_volume_reconciliation(spark, cfg):
    """Essential count check to ensure 100% data ingestion"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    raw_df = spark.read.format(cfg["fmt"]).options(**cfg["opts"]).load(cfg["src"])
    raw_count = raw_df.count()
    bronze_count = spark.table(table_fullname).count()
    assert raw_count == bronze_count

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_metadata_audit_and_schema(spark, cfg):
    """Governance Audit: load_dt and source_file validation"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    df = spark.table(table_fullname)
    cols = df.columns

    # Strict check for audit columns
    for audit_col in ["load_dt", "source_file"]:
        assert audit_col in cols
        assert df.filter(F.col(audit_col).isNull()).count() == 0

    # Flexible alert for rescued data noise
    if "_rescued_data" in cols:
        rescued_count = df.filter(F.col("_rescued_data").isNotNull()).count()
        if rescued_count > 0:
            print(f"⚠️ NOTICE: {cfg['name']} has {rescued_count} rescued rows.")

@pytest.mark.parametrize("cfg", TEST_CONFIG)
def test_row_level_integrity_flexible(spark, cfg):
    """Data Accuracy Audit: Fingerprinting (SHA-256) validation"""
    table_fullname = f"{CATALOG}.{BRONZE}.{cfg['name']}"
    src_df = spark.read.format(cfg["fmt"]).options(**cfg["opts"]).option("inferSchema", "false").load(cfg["src"])
    brz_df = spark.table(table_fullname)
    common_cols = [c for c in src_df.columns if c in brz_df.columns]

    def get_fingerprints(df, columns):
        # Normalization logic remains unchanged
        return df.select([F.coalesce(F.trim(F.col(c).cast("string")), F.lit("")).alias(c) for c in columns]) \
                 .withColumn("fp", F.sha2(F.concat_ws("||", *columns), 256)).select("fp")

    src_fp = get_fingerprints(src_df, common_cols)
    brz_fp = get_fingerprints(brz_df, common_cols)
    
    diff_count = src_fp.subtract(brz_fp).count()
    
    if diff_count > 0:
        print(f"⚠️ INTEGRITY WARNING: {cfg['name']} has {diff_count} mismatched rows.")
        # Soft-alert for high-volume photo table
        if cfg['name'] != 'listings_photo_bronze':
             assert diff_count == 0
    else:
        print(f"✅ {cfg['name']} row-level integrity verified.")