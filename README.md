#  VStone — Car Market Analytics Platform

> A production-grade Databricks Lakehouse pipeline that ingests, transforms, and surfaces **1M+ used-car listings** from the Russian market into actionable business intelligence dashboards — built on the **Medallion Architecture** (Bronze → Silver → Gold) with full CI/CD, SHA-256 data integrity testing, and Unity Catalog governance.

---

##  Architecture Overview

```
Raw Files (CSV / JSON / XML / Text / Photos / Geo / catalog)
        │
        ▼
┌────────────────────────────────────────────────────┐
│                  DATABRICKS                        │
│  ┌─────────────────────────────────────────────┐   │
│  │  Unity Catalog  (vstone_catalog)            │   │
│  └─────────────────────────────────────────────┘   │
│  ┌──────────────┐                                  │
│  │ Bronze Layer │  ← COPY INTO, Auto Loader,        │
│  │ (Raw Ingest) │    DLT, PySpark XML               │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ Silver Layer │  ← DLT Streaming, Cleanse,        │
│  │ (Conformed)  │    Merge, SCD2, USD Conversion    │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │  Gold Layer  │  ← DLT Aggregations, Star Schema, │
│  │  (Analytics) │    KPI Cubes, RLS / CLS           │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  Databricks SQL Dashboards                         │
└────────────────────────────────────────────────────┘
        │
        ▼
External: GitHub (CI/CD) │ Kaggle (Source Data)
```

---

##  Data Sources

| File | Entity | Format | Ingestion Method | Bronze Table |
|------|--------|--------|-----------------|--------------|
| `1_main_chunk_1.csv` | Car Listings (50%) | CSV | COPY INTO | `listings_csv_copyinto` |
| `1_main_chunk_2.csv` | Car Listings (20%) | CSV | DLT Auto Loader | `listings_csv_dlt` |
| `1_main_chunk_3.json` | Car Listings (20%) | JSON | Auto Loader (streaming) | `listings_json_autoloader` |
| `1_main_chunk_4.xml` | Car Listings (10%) | XML | PySpark native XML | `listings_xml_pyspark` |
| `1_text.csv` | Listing Descriptions | CSV | DLT Auto Loader | `listings_text` |
| `1_photo.csv` | Photo URLs | CSV | DLT Auto Loader | `listings_photo` |
| `catalogs.csv` | Car Catalog (specs) | CSV (semicolon) | DLT Auto Loader | `car_catalog` |
| `final_geografic.csv` | City Coordinates | CSV | DLT Auto Loader | `geo_locations` |

---

##  Bronze Layer

- Ingests raw data using **COPY INTO** (idempotent), **Auto Loader** (streaming), **PySpark XML**, and **DLT pipelines**
- Every row receives `source_file` and `load_dt` audit columns
- Tables tagged with `quality`, `source`, `description` in Unity Catalog
- **Change Data Feed** enabled on all Bronze tables for downstream CDC
- Null-ID audit and purge on COPY INTO path — malformed rows logged, not silently dropped
- Supports **first-time full load** and **incremental load** via `COPY_OPTIONS('force'='false')`

---

##  Silver Layer

Built with **Databricks Delta Live Tables (DLT)**. All 5 Silver tables stream from Bronze and apply data quality rules, type casting, and deduplication.

**Key transformations:**

| Table | Source | Key Transformations |
|-------|--------|---------------------|
| `listings_silver_merged` | 4 Bronze listing tables | Union, cast, USD conversion (÷ 82.5), price_category, dedup by `listing_id` |
| `listings_text_transformation` | `listings_text` | ID cast, dedup by `listing_id` |
| `listings_photo_transformation` | `listings_photo` | URL standardize (lowercase), dedup by `listing_id + photo_url_clean` |
| `car_catalog_transformation` | `car_catalog` | Cyrillic → English column rename (19 cols), numeric unit stripping |
| `geography_transformation` | `geo_locations` | lat/lon cast to DOUBLE, Russia bounding box validation (41–82°N, 19–180°E) |

**DLT Quality Constraints on `listings_silver_merged`:**

| Constraint | Rule | Action |
|-----------|------|--------|
| `valid_listing_id` | `listing_id IS NOT NULL` | DROP ROW |
| `valid_price` | `price_rub IS NOT NULL` | DROP ROW |
| `valid_date` | `listing_date IS NOT NULL` | DROP ROW |
| `positive_price` | `price_rub > 0` | DROP ROW |

Every rejected row is routed to a paired **quarantine table** (`listings_main_quarantine`, `car_catalog_quarantine`, etc.) with a `quarantine_reason` column.

**Pandas UDFs used:**
- `standardize_text` — lowercase + strip (brand, model, photo URL)
- `clean_text` — strip only (Cyrillic catalog text)
- `standardize_geo` — strip for city names

---

##  Gold Layer — Star Schema

All Gold tables live in `vstone_catalog.gold` and are managed by DLT.

### Dimension Tables (SCD Type 2)

| Table | Natural Key | Tracked Attributes |
|-------|-------------|-------------------|
| `dim_car` | `brand + model` | generation, trim, engine specs |
| `dim_location` | `city_prepositional` | lat, lon, city name |
| `dim_listing_details` | `listing_id` | Russian text description |
| `dim_listing_photos` | `listing_id + photo_url_clean` | photo URL changes |
| `dim_date` | `date_key` | Static 2010–2030 calendar (7,671 rows) |

> **SCD Type 2:** When an attribute changes, a new row is inserted. `__END_AT IS NULL` = current active record.

### Fact Table

**`fact_listings`** — one row per car listing, full star schema with 4 FK references

| Column | Description |
|--------|-------------|
| `listing_id` | Natural PK, FK → dim_listing_details / dim_listing_photos |
| `listing_date` | FK → dim_date.date_key |
| `brand`, `model` | FK → dim_car |
| `location_key` | FK → dim_location.city_prepositional |
| `price_rub`, `price_usd` | Raw & converted financials |
| `price_category` | BUDGET / MID_RANGE / PREMIUM / LUXURY |
| `car_age_at_listing` | Gold-level derived: `listing_year - manufacture_year` |
| `is_high_mileage` | `mileage_km > 100,000` |
| `price_per_hp_usd` | `price_usd / engine_power` |
| `photo_count` | Denormalized from `dim_listing_photos` |
| `bronze_load_dt`, `silver_load_dt`, `gold_load_dt` | Full audit chain |

### Aggregation Tables (5 tables)

| Table | Business Question |
|-------|-----------------|
| `agg_monthly_sales_trend` | How is listing volume and revenue trending month over month? |
| `agg_brand_location_performance` | Which brands dominate which cities? |
| `agg_regional_market_depth` | How deep is inventory by city, fuel type, and price segment? |
| `agg_comprehensive_kpi_cube` | Multi-dimensional KPI: brand × model × year × fuel × mileage |
| `agg_top_10_brands_by_spend` | Which 10 brands hold the most cumulative USD market value? |

---

##  Testing

All tests use **SHA-256 row hashing** for byte-level data integrity verification. Test suite built with `pytest` + `databricks-connect`.

### Bronze Tests (`tests/bronze_testing/test_bronze_layer.py`)

| Suite | Test | What it checks |
|-------|------|----------------|
| T1 | Volume & Completeness | Source file row count = Bronze table row count |
| T1 | Non-empty | Bronze table has > 0 rows |
| T1 | Null-ID purge | `listings_csv_copyinto` has zero null-ID rows |
| T2 | SHA-256 (source → Bronze) | Every source row fingerprint exists in Bronze |
| T2 | SHA-256 (Bronze → source) | No Bronze rows invented beyond source |
| T3 | Audit columns | `load_dt` and `source_file` present and non-null |
| T3 | Type check | `load_dt` is TIMESTAMP type |
| T3 | Rescued data | `_rescued_data` entirely NULL (no schema mismatch) |
| T3 | Source file identity | `source_file` correctly identifies origin file |

### Silver Tests (`tests/silver_testing/test_silver_layer.py`)

| Suite | Test | What it checks |
|-------|------|----------------|
| T1 | Reconciliation | Bronze rows = Silver + Quarantine (no silent drops) |
| T1 | Non-empty | Silver table has > 0 rows |
| T1 | Quarantine reasons | Every quarantine row has a non-null `rejection_reason` |
| T2 | SHA-256 integrity | Every Silver row traceable byte-for-byte to Bronze |
| T3 | Audit columns | `bronze_load_dt`, `bronze_source_file`, `silver_load_dt` present and non-null |
| T3 | Type check | `silver_load_dt` is TIMESTAMP type |
| T3 | PK non-null | Primary key columns have zero NULL values |

### Gold Tests (`tests/gold_testing/test_gold_layer.py`)

| Suite | Test | What it checks |
|-------|------|----------------|
| T1 | Reconciliation | `fact_listings` within 1% of `listings_silver_merged` |
| T1 | SCD2 CURRENT counts | Active dim rows match Silver distinct keys |
| T1 | `dim_date` rows | Exactly 7,671 rows (2010–2030 calendar) |
| T1 | Agg non-empty | All 5 aggregate tables have > 0 rows |
| T2 | listing_id lineage | Every Silver `listing_id` present in `fact_listings` |
| T2 | No invented IDs | `fact_listings` contains no IDs absent from Silver |
| T2 | SCD2 key coverage | Silver keys ↔ Gold active rows (both directions) |
| T3 | `gold_load_dt` | Present and non-null on all Gold tables |
| T3 | SCD2 metadata | `__START_AT`, `__END_AT`, `silver_load_dt` on all SCD2 dims |
| T3 | Full audit chain | `bronze_load_dt → silver_load_dt → gold_load_dt` on `fact_listings` |
| T4 | Schema integrity | All FK, derived, financial, and audit columns present |
| T5 | Star schema join rate | All FK joins ≥ 60% |
| T6 | SCD2 integrity | One active row per key, `__START_AT < __END_AT`, no orphan keys |

### Integration Tests (`tests/integration_test/test_e2e_pipeline.py`)

End-to-end suite (T1–T7) covering the full Bronze → Silver → Gold chain, including SHA-256 cross-layer lineage, SCD2 timeline consistency, and FK join quality — all in a single pytest run.

---

## 🔁 CI/CD

GitHub Actions runs automatically on every push to `dev` branch:

```
.github/workflows/
    └── cicd.yml
            ├── validate-bundle       ← Databricks Asset Bundle validation
            ├── run-bronze-tests      ← pytest Bronze layer (needs: validate-bundle)
            ├── run-silver-tests      ← pytest Silver layer (needs: validate-bundle)
            ├── run-gold-tests        ← pytest Gold layer (needs: validate-bundle)
            ├── run-integration-test  ← pytest E2E (needs: bronze + silver + gold)
            ├── deploy-and-run        ← DAB deploy + run end_to_end_pipeline (needs: integration)
            └── notify                ← Email to akash.r.mishra@v4c.ai on success / failure
```

**Required GitHub Secrets:**

| Secret | Value |
|--------|-------|
| `DATABRICKS_HOST` | Your Databricks workspace URL |
| `DATABRICKS_TOKEN` | Databricks personal access token |
| `MAIL_USERNAME` | Gmail address for notifications |
| `MAIL_PASSWORD` | Gmail App Password (16-char) |

---

##  Governance

**Row-Level Security (RLS)** — `vstone_catalog.security.rls_brand_market_data`

| Group | Brands Visible |
|-------|---------------|
| `admins` | All brands (unrestricted) |
| `premium_users` | `bmw`, `mercedes-benz`, `lexus` |
| `toyota_users` | `toyota` only |
| `honda_users` | `honda` only |
| *(others)* | No rows (deny by default) |

**Column-Level Security (CLS)** — `vstone_catalog.security.cls_brand_market_data`

| Column | Visible to | Hidden for |
|--------|-----------|------------|
| `total_market_value_usd` | `admins`, `finance_group` | All others → `**REDACTED**` |
| `avg_price_usd` | `admins`, `finance_group` | All others → `**REDACTED**` |
| `price_rub`, `price_usd`, `price_per_hp_usd` | `admins`, `finance_group` | All others → `**REDACTED**` |
| `color_r`, `color_g`, `color_b` | `admins` only | All others → `NULL` |

All tables registered in Unity Catalog under `vstone_catalog` with three-level namespace: `catalog.schema.table`. Audit columns (`source_file`, `load_dt`) on every Bronze table.

---

## Performance Optimization

Notebook `src/gold/11_performance_optimization.ipynb` benchmarks two strategies on `fact_listings` (5 queries × 3 runs each):

| Strategy | Configuration | Best For |
|----------|---------------|---------|
| **Liquid Clustering** | `CLUSTER BY (brand, listing_year, location_key, fuel_type)` | Unpredictable multi-column ad-hoc filters |
| **Partitioned + Z-Order** | `PARTITIONED BY (fuel_type)` + `ZORDER BY (brand, listing_year)` | Predictable single-column WHERE clauses |

Also includes **MERGE INTO** demo for late-arriving data corrections (500 price revisions + 10 new inserts).

Churn metrics computed: `agg_stale_inventory` (brand+model+location combos inactive >180 days) and `agg_brand_churn_metrics` (brand-level churn score %).

---

##  Dashboards

**Dashboard 1 — VStone: Car Market Analytics**
- KPI Counters: Total Listings, Market Value (USD), Avg Price
- Top 10 Brands by Market Value (pie chart)
- Monthly Sales Trend by Brand (bar chart)
- Market Value Growth Over Time (area chart)

**Dashboard 2 — Market Intelligence & Performance**
- Executive KPI counters (market value, listings, avg price)
- Top 10 Brands by Total Listings (horizontal bar)
- Price vs Volume Brand Bubble Chart (scatter)
- Top 15 Cities by Inventory Volume (bar)
- Fuel Type Mix by City — Top 10 Cities (area)

---

##  Repository Structure

```
vstone-databricks-pipeline/
├── .github/
│   └── workflows/
│       └── cicd.yml                    # CI/CD pipeline
├── resources/                          # Databricks Asset Bundle YAMLs
│   ├── variables.yml                   # Catalog / schema / warehouse vars
│   ├── end_to_end_pipeline.yml         # Master orchestration job
│   ├── bronze_chunk4_dlt.yml
│   ├── copy_into.yml
│   ├── json_xml_dlt.yml
│   ├── remaining_files.yml
│   ├── silver_transformation.yml
│   ├── gold_layer.yml
│   └── dashboards.yml
├── src/
│   ├── setup/
│   │   ├── 00_catalog_setup.py         # Catalog + schema + volume creation
│   │   ├── 01_data_profiling.py        # Source file audit & KPIs
│   │   └── 02_data_chunking.py         # Split 1_main.csv → 4 chunks (50/20/20/10)
│   ├── bronze/
│   │   ├── 03_bronze_csv_copyinto.py   # COPY INTO ingestion (Chunk 1)
│   │   ├── 04_bronze_auto_loader.py    # JSON Auto Loader streaming (Chunk 3)
│   │   ├── 05_bronze_xml_pyspark.py    # PySpark native XML (Chunk 4)
│   │   ├── 06_bronze_dlt.py            # DLT CSV ingestion (Chunk 2)
│   │   └── 07_remaining_4_files.py     # DLT: text, photo, catalog, geo
│   ├── silver/
│   │   ├── 08_silver_transformation.py # DLT Silver pipeline (all 5 tables)
│   │   └── 09_business_and_delta_timetravel.ipynb  # ACID + Time Travel demo
│   ├── gold/
│   │   ├── 10_gold_layer_dlt.py        # DLT Gold Star Schema + 5 aggregates
│   │   └── 11_performance_optimization.ipynb       # Liquid vs Z-Order benchmarks
│   ├── Governance/
│   │   ├── 11_RLS_row_level_security.ipynb
│   │   └── 11_CLS_column_level_security.ipynb
│   ├── Dashboard/
│   │   ├── Car_Market_Analytics.lvdash.json
│   │   └── Market_Intelligence_Performance.lvdash.json
│   ├── Usage Analysis/
│   │   └── 13_dbu_usage_analysis.ipynb # DBU cost analysis (system.billing.*)
│   └── utils/
│       ├── csv_splitter.py             # 50/20/20/10 chunking utility
│       ├── csv_to_json.py              # Chunk 3 converter
│       └── csv_to_xml.py              # Chunk 4 converter
├── tests/
│   ├── bronze_testing/
│   │   └── test_bronze_layer.py        # T1 Volume, T2 SHA-256, T3 Schema (8 tables)
│   ├── silver_testing/
│   │   └── test_silver_layer.py        # T1 Reconciliation, T2 SHA-256, T3 Audit (5 tables)
│   ├── gold_testing/
│   │   └── test_gold_layer.py          # T1-T6 Gold quality (fact + 4 dims + 5 aggs)
│   ├── integration_test/
│   │   └── test_e2e_pipeline.py        # T1-T7 End-to-End full pipeline
│   └── data_chunking_test/
│       └── 01_data_chunking_test.ipynb # Chunk integrity + format validation
├── databricks.yml                      # DAB root config (dev / test / prod targets)
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Setup & Running

### Prerequisites

- Azure Databricks workspace with **Unity Catalog** enabled
- Cluster runtime: **Databricks Runtime 13.3 LTS** or later
- Databricks CLI + Asset Bundles installed
- Python **3.10+** for local pytest runs

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/vstone-databricks-pipeline.git
cd vstone-databricks-pipeline
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Databricks CLI

```bash
databricks configure --token
# Enter your workspace URL and personal access token
```

### 4. Run Catalog Setup

Open `src/setup/00_catalog_setup.py` in Databricks and run. This creates:

```
vstone_catalog
  ├── raw       (schema + volumes: landing, chunks, checkpoints)
  ├── bronze
  ├── silver
  ├── gold
  └── security
```

### 5. Upload Source Files to Landing Volume

```
/Volumes/vstone_catalog/raw/landing/
  ├── 1_main.csv
  ├── 1_text.csv
  ├── 1_photo.csv
  ├── catalogs.csv
  └── final_geografic.csv
```

### 6. Run Data Chunking

Run `src/setup/02_data_chunking.py` to split `1_main.csv` into 4 format chunks (50/20/20/10).

### 7. Deploy with DAB

```bash
databricks bundle deploy --target dev
databricks bundle run end_to_end_pipeline --target dev
```

### 8. Run Tests

```bash
# Run all Bronze tests
pytest tests/bronze_testing/test_bronze_layer.py -v

# Run all Silver tests
pytest tests/silver_testing/test_silver_layer.py -v

# Run all Gold tests
pytest tests/gold_testing/test_gold_layer.py -v

# Run full end-to-end integration suite
pytest tests/integration_test/test_e2e_pipeline.py -v
```

---

##  Naming Conventions

| Layer | Pattern | Example |
|-------|---------|---------|
| Bronze | `listings_{format}_{method}` | `listings_csv_copyinto` |
| Silver | `{entity}_transformation` | `car_catalog_transformation` |
| Silver (merged) | `listings_silver_merged` | `listings_silver_merged` |
| Gold Dims | `dim_{entity}` | `dim_location` |
| Gold Fact | `fact_listings` | `fact_listings` |
| Gold Aggs | `agg_{description}` | `agg_top_10_brands_by_spend` |
| Security Views | `rls_{entity}` / `cls_{entity}` | `rls_brand_market_data` |
| Notebooks | `{nn}_{name}.py` | `08_silver_transformation.py` |
| Tests | `test_{layer}_layer.py` | `test_bronze_layer.py` |

---

##  Tech Stack

| Category | Tool / Technology |
|----------|------------------|
| Platform | Azure Databricks (Community / Free Edition) |
| Catalog | Unity Catalog (`vstone_catalog`) |
| Storage | Delta Lake |
| Pipeline | Delta Live Tables (DLT) |
| Ingestion | COPY INTO, Auto Loader, PySpark XML, DLT |
| Language | Python, PySpark, SQL |
| Testing | pytest + Databricks Connect, SHA-256 row hashing |
| Security | Unity Catalog RLS + CLS (Views) |
| Deployment | Databricks Asset Bundles (DAB) |
| CI/CD | GitHub Actions |
| Dashboards | Databricks SQL Dashboards (.lvdash.json) |
| Cost Analysis | `system.billing.usage` + `system.billing.list_prices` |
| Notifications | Gmail SMTP via `dawidd6/action-send-mail@v3` |

---

##  Key Metrics

| Metric | Value |
|--------|-------|
| Total source listings | ~1,083,237 |
| Silver merged rows | 1,083,237 |
| Photo records | 7,948,322 |
| Car catalog entries | 117,991 |
| Geography records | 1,462 |
| `dim_date` rows | 7,671 (2010–2030) |
| USD conversion rate | 82.5 RUB/USD (Feb 2023 historical) |
| Minimum FK join rate | 60% (all dims) |
| Test assertions | 144+ across all suites |

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit using the convention below
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request against `dev`

**Commit convention:**

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature or notebook |
| `fix:` | Bug fix |
| `test:` | Adding or updating tests |
| `docs:` | README or documentation |
| `ci:` | CI/CD workflow changes |

---

##  License

MIT License — Copyright (c) 2026 **Akash Mishra**

This license applies to all source code, notebooks, SQL queries, PySpark transformations, Delta Live Tables pipelines, pytest test suites, and architecture assets contained in this repository. Raw data files stored in Databricks Volumes are excluded.

---

##  Contact

| | |
|---|---|
| **Author** | Akash Mishra |
| **Email** | akash.r.mishra@v4c.ai |
| **GitHub** | github.com/your-username/vstone-databricks-pipeline |