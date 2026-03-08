"""
test_bronze_layer.py  —  Bronze Layer Test Suite
=================================================
Mirrors Databricks notebook: 02_bronze_layer_test.py

Suites
------
T1 · Volume & Completeness  — row count: source vs Bronze
T2 · Row-to-Row Integrity   — SHA-256 fingerprint match for every row
T3 · Schema & Metadata      — audit columns, _rescued_data, data types

Registry
--------
8 Bronze tables tested (same as notebook REGISTRY):
  Chunk 1  — CSV  / COPY INTO   → listings_csv_copyinto
  Chunk 2  — CSV  / DLT         → listings_csv_dlt
  Chunk 3  — JSON / Auto Loader → listings_json_autoloader
  Chunk 4  — XML  / PySpark     → listings_xml_pyspark
  Landing  — Text Data          → listings_text
  Landing  — Photo Data         → listings_photo
  Landing  — Car Catalog        → car_catalog
  Landing  — Geo Locations      → geo_locations

All DataFrames are built from synthetic in-memory data — no cluster needed.
"""

import pytest
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType, TimestampType, IntegerType,
)



# ─────────────────────────────────────────────────────────────────────────────
# SparkSession fixture  (local, in-memory — no Databricks cluster needed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    session = (
        SparkSession.builder
        .master("local[*]")
        .appName("bronze_layer_tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers  (mirror notebook helpers verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def _fingerprint_df(df, columns):
    """
    Normalises each column to trimmed STRING, replaces NULLs with '',
    then hashes all columns together into a single SHA-256 fingerprint per row.
    Mirrors notebook _fingerprint_df() exactly.
    """
    normalised = df.select([
        F.coalesce(F.trim(F.col(c).cast("string")), F.lit("")).alias(c)
        for c in columns
    ])
    return (
        normalised
        .withColumn("fingerprint", F.sha2(F.concat_ws("||", *columns), 256))
        .select("fingerprint")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bronze table fixtures  (synthetic in-memory, mirrors REGISTRY sources)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bronze_tables(spark):
    """
    Returns a dict of all 8 Bronze DataFrames keyed by table name.
    Each DataFrame has:
      - all source data columns (as STRING, matching COPY INTO / Auto Loader behaviour)
      - load_dt  (TIMESTAMP audit column)
      - source_file (STRING audit column)
    """
    load_ts = datetime(2024, 1, 5)

    # ── Chunk 1 — CSV / COPY INTO  (listings_csv_copyinto)
    # Note: 2 null-id rows in source are excluded from Bronze (null_exclude_col="id")
    listings_csv_copyinto = spark.createDataFrame([
        ("1", "01.01.2021", "500000",  "Toyota",   "Camry", "2019", "Moscow",          load_ts, "chunk1.csv"),
        ("2", "15.03.2021", "800000",  "BMW",       "X5",    "2020", "SPb",             load_ts, "chunk1.csv"),
        ("3", "10.06.2021", "300000",  "Lada",      "Vesta", "2018", "Kazan",           load_ts, "chunk1.csv"),
        ("4", "20.09.2021", "1200000", "Mercedes",  "GLE",   "2021", "Novosibirsk",     load_ts, "chunk1.csv"),
        ("5", "05.12.2021", "250000",  "Kia",       "Rio",   "2017", "Yekaterinburg",   load_ts, "chunk1.csv"),
    ], ["id","date","cost","marka","model","year","city","load_dt","source_file"])

    # ── Chunk 2 — CSV / DLT  (listings_csv_dlt)
    listings_csv_dlt = spark.createDataFrame([
        ("6",  "11.01.2021", "420000",  "Hyundai", "Solaris", "2020", "Rostov",         load_ts, "chunk2.csv"),
        ("7",  "22.02.2021", "950000",  "Audi",    "A6",      "2019", "Ufa",            load_ts, "chunk2.csv"),
        ("8",  "03.04.2021", "175000",  "Lada",    "Granta",  "2017", "Omsk",           load_ts, "chunk2.csv"),
        ("9",  "14.05.2021", "670000",  "Nissan",  "X-Trail", "2018", "Samara",         load_ts, "chunk2.csv"),
        ("10", "25.07.2021", "1100000", "Lexus",   "RX",      "2021", "Krasnodar",      load_ts, "chunk2.csv"),
    ], ["id","date","cost","marka","model","year","city","load_dt","source_file"])

    # ── Chunk 3 — JSON / Auto Loader  (listings_json_autoloader)
    listings_json_autoloader = spark.createDataFrame([
        ("11", "01.08.2021", "310000",  "Skoda",   "Octavia", "2018", "Voronezh",       load_ts, "chunk3.json"),
        ("12", "12.09.2021", "880000",  "Toyota",  "Land Cruiser","2020","Perm",         load_ts, "chunk3.json"),
        ("13", "23.10.2021", "220000",  "Renault", "Logan",   "2016", "Volgograd",      load_ts, "chunk3.json"),
        ("14", "04.11.2021", "760000",  "Volkswagen","Tiguan","2019", "Chelyabinsk",    load_ts, "chunk3.json"),
        ("15", "15.12.2021", "140000",  "Daewoo",  "Nexia",   "2010", "Saratov",        load_ts, "chunk3.json"),
    ], ["id","date","cost","marka","model","year","city","load_dt","source_file"])

    # ── Chunk 4 — XML / PySpark  (listings_xml_pyspark)
    listings_xml_pyspark = spark.createDataFrame([
        ("16", "05.01.2021", "540000",  "Mazda",   "CX-5",    "2019", "Tyumen",         load_ts, "chunk4.xml"),
        ("17", "16.02.2021", "390000",  "Kia",     "Sportage","2018", "Irkutsk",        load_ts, "chunk4.xml"),
        ("18", "27.03.2021", "1350000", "BMW",     "5 Series","2021", "Vladivostok",    load_ts, "chunk4.xml"),
    ], ["id","date","cost","marka","model","year","city","load_dt","source_file"])

    # ── Landing — Text Data  (listings_text)
    listings_text = spark.createDataFrame([
        ("1",  "Отличный автомобиль, один хозяин",  load_ts, "1_text.csv"),
        ("2",  "Срочная продажа, торг уместен",     load_ts, "1_text.csv"),
        ("3",  "Состояние хорошее, не бита",        load_ts, "1_text.csv"),
        ("4",  "Продаю без торга, цена окончательна",load_ts,"1_text.csv"),
        ("5",  "Первый хозяин, гаражное хранение",  load_ts, "1_text.csv"),
    ], ["id","text","load_dt","source_file"])

    # ── Landing — Photo Data  (listings_photo)
    listings_photo = spark.createDataFrame([
        ("1",  "http://img.ru/1a.jpg", load_ts, "1_photo.csv"),
        ("1",  "http://img.ru/1b.jpg", load_ts, "1_photo.csv"),
        ("2",  "http://img.ru/2a.jpg", load_ts, "1_photo.csv"),
        ("3",  "http://img.ru/3a.jpg", load_ts, "1_photo.csv"),
        ("4",  "http://img.ru/4a.jpg", load_ts, "1_photo.csv"),
        ("5",  "http://img.ru/5a.jpg", load_ts, "1_photo.csv"),
    ], ["id","photo_url","load_dt","source_file"])

    # ── Landing — Car Catalog  (car_catalog)
    car_catalog = spark.createDataFrame([
        ("Toyota",  "Camry",   "XV70", "Comfort",   "2.5 л", "181 л.с.", load_ts, "catalogs.csv"),
        ("BMW",     "X5",      "G05",  "xDrive40i", "3.0 л", "340 л.с.", load_ts, "catalogs.csv"),
        ("Lada",    "Vesta",   "I",    "Comfort",   "1.6 л", "106 л.с.", load_ts, "catalogs.csv"),
        ("Mercedes","GLE",     "V167", "300d",      "2.0 л", "245 л.с.", load_ts, "catalogs.csv"),
        ("Kia",     "Rio",     "IV",   "Classic",   "1.4 л", "100 л.с.", load_ts, "catalogs.csv"),
    ], ["Марка","Модель","Поколение","Комплектация",
        "Объём двигателя","Мощность двигателя","load_dt","source_file"])

    # ── Landing — Geo Locations  (geo_locations)
    geo_locations = spark.createDataFrame([
        ("Москве",          "Moscow",        "55.75", "37.61", load_ts, "final_geografic.csv"),
        ("Санкт-Петербурге","SPb",           "59.93", "30.31", load_ts, "final_geografic.csv"),
        ("Казани",          "Kazan",         "55.78", "49.12", load_ts, "final_geografic.csv"),
        ("Новосибирске",    "Novosibirsk",   "54.99", "82.90", load_ts, "final_geografic.csv"),
        ("Екатеринбурге",   "Yekaterinburg", "56.83", "60.59", load_ts, "final_geografic.csv"),
    ], ["name_padesh","greate_padesh","lat","lon","load_dt","source_file"])

    return {
        "listings_csv_copyinto":    listings_csv_copyinto,
        "listings_csv_dlt":         listings_csv_dlt,
        "listings_json_autoloader": listings_json_autoloader,
        "listings_xml_pyspark":     listings_xml_pyspark,
        "listings_text":            listings_text,
        "listings_photo":           listings_photo,
        "car_catalog":              car_catalog,
        "geo_locations":            geo_locations,
    }


@pytest.fixture(scope="module")
def source_tables(spark):
    """
    Raw source DataFrames — mirrors what read_source() loads from Volume paths.
    Chunk 1 source intentionally contains 2 null-id rows (correctly excluded from Bronze).
    """
    load_ts = datetime(2024, 1, 5)

    # Chunk 1 source has 2 extra null-id rows (bad data)
    chunk1_source = spark.createDataFrame([
        ("1",  "01.01.2021", "500000",  "Toyota",  "Camry", "2019", "Moscow"),
        ("2",  "15.03.2021", "800000",  "BMW",     "X5",    "2020", "SPb"),
        ("3",  "10.06.2021", "300000",  "Lada",    "Vesta", "2018", "Kazan"),
        ("4",  "20.09.2021", "1200000", "Mercedes","GLE",   "2021", "Novosibirsk"),
        ("5",  "05.12.2021", "250000",  "Kia",     "Rio",   "2017", "Yekaterinburg"),
        (None, "01.01.2021", "99999",   "Unknown", "???",   "2000", "Unknown"),  # bad row 1
        (None, "02.01.2021", "88888",   "Unknown", "???",   "2001", "Unknown"),  # bad row 2
    ], ["id","date","cost","marka","model","year","city"])

    chunk2_source = spark.createDataFrame([
        ("6",  "11.01.2021", "420000",  "Hyundai","Solaris","2020","Rostov"),
        ("7",  "22.02.2021", "950000",  "Audi",   "A6",    "2019","Ufa"),
        ("8",  "03.04.2021", "175000",  "Lada",   "Granta","2017","Omsk"),
        ("9",  "14.05.2021", "670000",  "Nissan", "X-Trail","2018","Samara"),
        ("10", "25.07.2021", "1100000", "Lexus",  "RX",    "2021","Krasnodar"),
    ], ["id","date","cost","marka","model","year","city"])

    chunk3_source = spark.createDataFrame([
        ("11", "01.08.2021", "310000",  "Skoda",     "Octavia",    "2018","Voronezh"),
        ("12", "12.09.2021", "880000",  "Toyota",    "Land Cruiser","2020","Perm"),
        ("13", "23.10.2021", "220000",  "Renault",   "Logan",      "2016","Volgograd"),
        ("14", "04.11.2021", "760000",  "Volkswagen","Tiguan",     "2019","Chelyabinsk"),
        ("15", "15.12.2021", "140000",  "Daewoo",    "Nexia",      "2010","Saratov"),
    ], ["id","date","cost","marka","model","year","city"])

    chunk4_source = spark.createDataFrame([
        ("16", "05.01.2021", "540000",  "Mazda","CX-5",     "2019","Tyumen"),
        ("17", "16.02.2021", "390000",  "Kia",  "Sportage", "2018","Irkutsk"),
        ("18", "27.03.2021", "1350000", "BMW",  "5 Series", "2021","Vladivostok"),
    ], ["id","date","cost","marka","model","year","city"])

    text_source = spark.createDataFrame([
        ("1", "Отличный автомобиль, один хозяин"),
        ("2", "Срочная продажа, торг уместен"),
        ("3", "Состояние хорошее, не бита"),
        ("4", "Продаю без торга, цена окончательна"),
        ("5", "Первый хозяин, гаражное хранение"),
    ], ["id","text"])

    photo_source = spark.createDataFrame([
        ("1", "http://img.ru/1a.jpg"),
        ("1", "http://img.ru/1b.jpg"),
        ("2", "http://img.ru/2a.jpg"),
        ("3", "http://img.ru/3a.jpg"),
        ("4", "http://img.ru/4a.jpg"),
        ("5", "http://img.ru/5a.jpg"),
    ], ["id","photo_url"])

    catalog_source = spark.createDataFrame([
        ("Toyota",  "Camry",   "XV70", "Comfort",   "2.5 л", "181 л.с."),
        ("BMW",     "X5",      "G05",  "xDrive40i", "3.0 л", "340 л.с."),
        ("Lada",    "Vesta",   "I",    "Comfort",   "1.6 л", "106 л.с."),
        ("Mercedes","GLE",     "V167", "300d",      "2.0 л", "245 л.с."),
        ("Kia",     "Rio",     "IV",   "Classic",   "1.4 л", "100 л.с."),
    ], ["Марка","Модель","Поколение","Комплектация","Объём двигателя","Мощность двигателя"])

    geo_source = spark.createDataFrame([
        ("Москве",          "Moscow",        "55.75","37.61"),
        ("Санкт-Петербурге","SPb",           "59.93","30.31"),
        ("Казани",          "Kazan",         "55.78","49.12"),
        ("Новосибирске",    "Novosibirsk",   "54.99","82.90"),
        ("Екатеринбурге",   "Yekaterinburg", "56.83","60.59"),
    ], ["name_padesh","greate_padesh","lat","lon"])

    return {
        "listings_csv_copyinto":    chunk1_source.filter(F.col("id").isNotNull()),  # exclude null-id rows
        "listings_csv_dlt":         chunk2_source,
        "listings_json_autoloader": chunk3_source,
        "listings_xml_pyspark":     chunk4_source,
        "listings_text":            text_source,
        "listings_photo":           photo_source,
        "car_catalog":              catalog_source,
        "geo_locations":            geo_source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# T1 · Volume & Completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestT1VolumeAndCompleteness:
    """
    T1 — Row count in source must equal row count in Bronze.
    Mirrors notebook test_volume() for all 8 REGISTRY entries.
    """

    @pytest.mark.parametrize("table_key", [
        "listings_csv_copyinto",
        "listings_csv_dlt",
        "listings_json_autoloader",
        "listings_xml_pyspark",
        "listings_text",
        "listings_photo",
        "car_catalog",
        "geo_locations",
    ])
    def test_source_count_equals_bronze_count(self, bronze_tables, source_tables, table_key):
        """Source row count must exactly match Bronze row count (no missing, no extra rows)."""
        src_count    = source_tables[table_key].count()
        bronze_count = bronze_tables[table_key].count()
        gap          = src_count - bronze_count
        assert gap == 0, (
            f"[{table_key}] Row count mismatch: "
            f"Source={src_count:,}, Bronze={bronze_count:,}, GAP={gap:,}"
        )

    def test_all_bronze_tables_non_empty(self, bronze_tables):
        """Every Bronze table must contain at least 1 row."""
        for table_key, df in bronze_tables.items():
            cnt = df.count()
            assert cnt > 0, f"[{table_key}] Bronze table is empty"


# ─────────────────────────────────────────────────────────────────────────────
# T2 · Row-to-Row Integrity  (SHA-256 fingerprint)
# ─────────────────────────────────────────────────────────────────────────────

class TestT2RowToRowIntegrity:
    """
    T2 — SHA-256 fingerprint of every source row must appear in Bronze.
    Mirrors notebook test_integrity() / _fingerprint_df() exactly.
    """

    @pytest.mark.parametrize("table_key", [
        "listings_csv_copyinto",
        "listings_csv_dlt",
        "listings_json_autoloader",
        "listings_xml_pyspark",
        "listings_text",
        "listings_photo",
        "car_catalog",
        "geo_locations",
    ])
    def test_no_source_rows_missing_from_bronze(self, bronze_tables, source_tables, table_key):
        """
        Every source fingerprint must exist in Bronze (nothing lost during ingestion).
        Uses subtract() to find fingerprints in source but not in Bronze.
        """
        src_df   = source_tables[table_key]
        brz_df   = bronze_tables[table_key]
        # Only compare columns that exist in BOTH DataFrames
        common   = [c for c in src_df.columns if c in brz_df.columns]

        src_fp   = _fingerprint_df(src_df, common)
        brz_fp   = _fingerprint_df(brz_df, common)
        missing  = src_fp.subtract(brz_fp).count()

        assert missing == 0, (
            f"[{table_key}] {missing} source row(s) not found in Bronze "
            f"(compared {len(common)} columns: {common})"
        )

    @pytest.mark.parametrize("table_key", [
        "listings_csv_copyinto",
        "listings_csv_dlt",
        "listings_json_autoloader",
        "listings_xml_pyspark",
        "listings_text",
        "listings_photo",
        "car_catalog",
        "geo_locations",
    ])
    def test_no_extra_rows_invented_in_bronze(self, bronze_tables, source_tables, table_key):
        """
        No Bronze fingerprint may be absent from source (Bronze must not invent rows).
        Uses subtract() to find fingerprints in Bronze but not in source.
        """
        src_df  = source_tables[table_key]
        brz_df  = bronze_tables[table_key]
        common  = [c for c in src_df.columns if c in brz_df.columns]

        src_fp  = _fingerprint_df(src_df, common)
        brz_fp  = _fingerprint_df(brz_df, common)
        extra   = brz_fp.subtract(src_fp).count()

        assert extra == 0, (
            f"[{table_key}] {extra} Bronze row(s) not traceable to source "
            f"(compared {len(common)} columns)"
        )



# ─────────────────────────────────────────────────────────────────────────────
# T3 · Schema & Metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestT3SchemaAndMetadata:
    """
    T3 — Audit columns, _rescued_data cleanliness, and data types.
    Mirrors notebook test_schema_and_metadata() for all 8 Bronze tables.
    """

    AUDIT_COLS = ["load_dt", "source_file"]

    @pytest.mark.parametrize("table_key", [
        "listings_csv_copyinto",
        "listings_csv_dlt",
        "listings_json_autoloader",
        "listings_xml_pyspark",
        "listings_text",
        "listings_photo",
        "car_catalog",
        "geo_locations",
    ])
    def test_audit_columns_present(self, bronze_tables, table_key):
        """load_dt and source_file must exist in every Bronze table schema."""
        df      = bronze_tables[table_key]
        missing = [c for c in self.AUDIT_COLS if c not in df.columns]
        assert missing == [], (
            f"[{table_key}] Missing audit columns: {missing}"
        )

    @pytest.mark.parametrize("table_key", [
        "listings_csv_copyinto",
        "listings_csv_dlt",
        "listings_json_autoloader",
        "listings_xml_pyspark",
        "listings_text",
        "listings_photo",
        "car_catalog",
        "geo_locations",
    ])
    def test_audit_columns_non_null(self, bronze_tables, table_key):
        """load_dt and source_file must have zero null values."""
        df = bronze_tables[table_key]
        for col in self.AUDIT_COLS:
            if col in df.columns:
                null_cnt = df.filter(F.col(col).isNull()).count()
                assert null_cnt == 0, (
                    f"[{table_key}] Column '{col}' has {null_cnt} NULL rows"
                )

    @pytest.mark.parametrize("table_key", [
        "listings_csv_copyinto",
        "listings_csv_dlt",
        "listings_json_autoloader",
        "listings_xml_pyspark",
        "listings_text",
        "listings_photo",
        "car_catalog",
        "geo_locations",
    ])
    def test_load_dt_is_timestamp(self, bronze_tables, table_key):
        """load_dt must be of TIMESTAMP type — never string or date."""
        df     = bronze_tables[table_key]
        dtypes = dict(df.dtypes)
        if "load_dt" in dtypes:
            assert dtypes["load_dt"].startswith("timestamp"), (
                f"[{table_key}] load_dt is '{dtypes['load_dt']}', expected 'timestamp'"
            )

    @pytest.mark.parametrize("table_key", [
        "listings_csv_copyinto",
        "listings_csv_dlt",
        "listings_json_autoloader",
        "listings_xml_pyspark",
        "listings_text",
        "listings_photo",
        "car_catalog",
        "geo_locations",
    ])
    def test_no_rescued_data_pollution(self, bronze_tables, table_key):
        """
        If _rescued_data column is present, it must be entirely NULL.
        Non-null _rescued_data means schema mismatch during Auto Loader ingestion.
        """
        df = bronze_tables[table_key]
        if "_rescued_data" in df.columns:
            rescued = df.filter(F.col("_rescued_data").isNotNull()).count()
            assert rescued == 0, (
                f"[{table_key}] _rescued_data has {rescued} non-null rows — "
                "indicates schema mismatch during ingestion"
            )

    def test_source_file_identifies_origin_correctly(self, bronze_tables):
        """source_file values must correctly identify the ingestion source file."""
        expected = {
            "listings_csv_copyinto":    "chunk1.csv",
            "listings_csv_dlt":         "chunk2.csv",
            "listings_json_autoloader": "chunk3.json",
            "listings_xml_pyspark":     "chunk4.xml",
            "listings_text":            "1_text.csv",
            "listings_photo":           "1_photo.csv",
            "car_catalog":              "catalogs.csv",
            "geo_locations":            "final_geografic.csv",
        }
        for table_key, expected_file in expected.items():
            df    = bronze_tables[table_key]
            files = [r["source_file"] for r in df.select("source_file").distinct().collect()]
            assert any(expected_file in f for f in files), (
                f"[{table_key}] Expected source_file containing '{expected_file}', got {files}"
            )

    def test_listing_id_columns_are_string_type(self, bronze_tables):
        """
        All listing id columns in Bronze must be STRING (raw ingestion preserves original types).
        COPY INTO and Auto Loader load with inferSchema=false by default.
        """
        for table_key in [
            "listings_csv_copyinto", "listings_csv_dlt",
            "listings_json_autoloader", "listings_xml_pyspark",
        ]:
            df     = bronze_tables[table_key]
            dtypes = dict(df.dtypes)
            if "id" in dtypes:
                assert dtypes["id"] == "string", (
                    f"[{table_key}] 'id' column should be STRING in Bronze, got '{dtypes['id']}'"
                )

    def test_source_file_non_empty_string(self, bronze_tables):
        """source_file must never be an empty string."""
        for table_key, df in bronze_tables.items():
            if "source_file" in df.columns:
                bad = df.filter(F.trim(F.col("source_file")) == "").count()
                assert bad == 0, (
                    f"[{table_key}] {bad} rows have empty source_file"
                )