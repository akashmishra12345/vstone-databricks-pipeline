#  VStone — Car Market Analytics Platform

> A full-stack Databricks Lakehouse pipeline that ingests, transforms, and surfaces 1M+ used-car listings from the Russian market into actionable business intelligence dashboards.

---

##  Visual Overview

```
Raw Files (CSV / JSON / XML / Text / Photos)
        │
        ▼
┌──────────────────┐
│   Bronze Layer   │  ← COPY INTO, Auto Loader, DLT
│  (Raw Ingestion) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Silver Layer   │  ← Cleanse, Merge, SCD2, USD Conversion
│  (Conformed)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Gold Layer    │  ← DLT Aggregations, KPI Cubes, Row/Column Level Security
│  (Analytics)     │
└────────┬─────────┘
         │
         ▼
  Databricks Dashboard   
```

---

##  Features

- **Multi-format Ingestion:** Handles CSV, JSON, XML, plain text, and binary photo metadata in a unified Bronze layer using COPY INTO, Auto Loader, and PySpark.
- **Intelligent Merging:** Silver layer deduplicates and merges 1,083,237 listings across formats with a configurable 60% join-rate threshold.
- **SCD Type 2 History:** Full change history on all Silver dimension tables using Delta Lake's `__START_AT` / `__END_AT` pattern.
- **DLT Gold Aggregations:** Five pre-built Gold aggregation tables covering brand performance, regional depth, monthly trends, and KPI cubes — all powered by Delta Live Tables.
- **Row & Column Level Security:** Unity Catalog RLS/CLS protecting brand-specific data (Toyota, Honda, Premium users) with group-based access control.
- **BI Dashboards:** Two production Databricks dashboards — Car Market Analytics and Market Intelligence & Performance (5 pages, 15+ widgets).
- **Full Test Coverage:** 144 pytest assertions across Bronze, Silver, Gold, and Security layers.

---

##  Tech Stack

| Category | Tool / Language |
|---|---|
| Platform | Databricks (Unity Catalog) |
| Storage | Delta Lake |
| Pipeline | Delta Live Tables (DLT) |
| Language | Python, SQL, PySpark |
| Ingestion | COPY INTO, Auto Loader, PySpark, dlt|
| Testing | pytest |
| Security | Unity Catalog RLS / CLS |
| Dashboards | Databricks SQL Dashboards |

---

##  Getting Started

### Prerequisites

- Databricks workspace with **Unity Catalog** enabled
- Cluster runtime: **Databricks Runtime 13.3 LTS** or later
- Access to `system.billing.*` tables (account admin required)
- Python **3.10+** for running pytest locally

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/vstone.git
cd vstone

# 2. Install Python test dependencies
pip install -r requirements.txt

# 3. Upload notebooks to Databricks
#    (via Databricks CLI or manually via UI)
databricks workspace import_dir notebooks/ /Users/you@email.com/vstone
```

### Configuration

Create a `.env` file or set the following Databricks notebook widgets before running:

```bash
# Catalog & Schema
CATALOG=vstone_catalog
BRONZE_SCHEMA=bronze
SILVER_SCHEMA=silver
GOLD_SCHEMA=gold

# Landing volume paths
LANDING_PATH=/Volumes/vstone_catalog/raw/landing/
CHUNKS_PATH=/Volumes/vstone_catalog/raw/chunks/

# Currency conversion
USD_RATE=82.5

```

---

##  Usage Examples

### Run the full Bronze → Silver → Gold pipeline

```bash
# In Databricks, open folder in order:
setup
bronze
silver
gold
dashboard
```

### Run the test suite locally

```bash
# Run all 144 tests
pytest vstone_tests/ -v

# Run only Silver layer tests
pytest vstone_tests/test_silver.py -v

# Run only security tests
pytest vstone_tests/test_security.py -v
```


### Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---
 Email | akashmishraa202@gmail.com |
---
