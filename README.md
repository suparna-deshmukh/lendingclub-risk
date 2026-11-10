# LendingClub Risk — Loan-Book Risk Analysis

SQL-first analysis of the LendingClub loan book (2007 – Q4 2018): EDA on 100K+ loans, borrower segmentation into **5 risk profiles** with K-Means, and an explainable **logistic-regression default model** evaluated per segment.

**Author:** Suparna Deshmukh
**Stack:** Python · DuckDB (SQL) · scikit-learn · pandas · matplotlib
**Full report:** [report_lendingclub.pdf](report_lendingclub.pdf) · [SQL queries used](sql_queries_used.pdf)

## The question

LendingClub's raw loan file hides a spread of borrower risk. This project answers three questions in order:

1. **Where does default risk live in the data?** — ten SQL EDA queries over grade, DTI, income, utilization and more.
2. **Can borrowers be grouped into meaningful risk segments?** — RFM-style features + K-Means (k=5), segments labelled by *actual* default rate.
3. **Can a simple, explainable model predict defaults and rank segments correctly?** — logistic regression in a sklearn pipeline, evaluated overall and per segment.

## Pipeline

Four scripts, run in order (each is self-contained and heavily commented):

| # | Script | What it does |
|---|--------|--------------|
| 1 | `01_load_data.py` | Reads the 2M-row CSV in 100K chunks (no full in-memory load), keeps loans with a clear outcome (`Fully Paid` / `Charged Off` / `Default`), builds the `is_default` label, draws a reproducible 100,000-row sample (seed 42), saves it as Parquet, and loads it into a DuckDB database. |
| 2 | `02_eda.py` | Runs 10 SQL queries against DuckDB — overall default rate, default rate by grade / home ownership / employment length / revolving utilization / inquiries / income decile, interest rate by grade, DTI of defaulters vs non-defaulters, loan amount by purpose. Every query is exported as CSV + a chart into `outputs/eda/`. |
| 3 | `03_segmentation.py` | Engineers R/F/M features via SQL, standardizes them, runs K-Means with k=5, labels each segment by its realized default rate, and writes `segment_profiles.csv` plus three figures. |
| 4 | `04_model.py` | Trains a `Pipeline` (scaler + one-hot encoding + logistic regression) on an 80/20 split; evaluates ROC-AUC, PR-AUC, confusion matrix, ROC and precision-recall curves; computes **per-segment** metrics; charts the top coefficients; saves the model to `models/logreg.pkl`. |

## Data

The raw file `accepted_2007_to_2018Q4.csv` (≈2M loans, ≈1.6 GB) is **not committed** — it is far above git's practical file limit. It is available from LendingClub's official datasets portal (https://www.lendingclub.com/datasets). Everything needed to reproduce the analysis is committed:

| File | What it is |
|------|------------|
| `data/loans_100k.parquet` | The 100K-row labelled sample (output of script 1) |
| `data/lc.duckdb` | DuckDB database with the sample loaded (query it directly with SQL) |
| `data/loans_segmented.parquet` | Sample + segment labels (output of script 3) |

## Results (from committed `outputs/`)

- **Five risk profiles** spanning realized default rates of **15.4 % → 26.7 %** (`outputs/segment_profiles.csv`):
  - Segment 1 — 30.0 % of loans, 15.35 % default rate, avg loan $11,360
  - Segment 5 — 15.8 % of loans, 26.73 % default rate, avg loan $19,927, highest mean DTI (20.6)
- **Per-segment ROC-AUC of 0.72 – 0.75**, and mean predicted probability ordering tracks the segments exactly (38 % → 53 %), i.e. the model and the clusters agree on which borrowers are risky (`outputs/per_segment_metrics.csv`).
- Ten EDA cuts with CSVs and charts in `outputs/eda/`; segment and model figures in `outputs/figures/`.

## How to run

```bash
# Dependencies (Python 3.9+)
pip install pandas duckdb scikit-learn matplotlib seaborn numpy

# If you want to regenerate the sample from the raw CSV:
#   1. download accepted_2007_to_2018Q4.csv from the LendingClub datasets portal
#   2. drop it into data/ and update the CONFIGURATION paths at the top of 01_load_data.py
python 01_load_data.py
python 02_eda.py
python 03_segmentation.py
python 04_model.py
```

Skipping script 1 is fine — `data/lc.duckdb` already contains the sample, so scripts 2–4 run against it directly.

> **Note:** scripts carry absolute paths in a `CONFIGURATION` block at the top — point them at your copy of the folder before running.

## Repository layout

```
lendingclub-risk/
├── 01_load_data.py          # chunked CSV → 100K sample → DuckDB
├── 02_eda.py                # 10 SQL EDA queries → CSVs + charts
├── 03_segmentation.py       # R/F/M features → K-Means k=5 → segment profiles
├── 04_model.py              # logistic regression pipeline + per-segment eval
├── data/                    # 100K parquet sample + DuckDB db (raw CSV: see above)
├── models/logreg.pkl        # trained model
├── outputs/
│   ├── eda/                 # per-query CSVs + charts
│   ├── figures/             # segmentation + model figures
│   ├── segment_profiles.csv
│   └── per_segment_metrics.csv
├── report_lendingclub.pdf   # full write-up
└── sql_queries_used.pdf     # every SQL query, with intent
```
