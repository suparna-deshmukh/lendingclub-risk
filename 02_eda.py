# ============================================================
# 02_eda.py
# LendingClub Risk Project — Exploratory Data Analysis
# ============================================================
# This script:
#   - Connects to the DuckDB database
#   - Runs 10 SQL queries to understand the data
#   - Prints each result as a readable table
#   - Saves each result as a CSV file
#   - Creates one bar chart per query
# ============================================================

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH       = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/lc.duckdb"
OUTPUT_CSV    = "/Users/suparna_deshmukh/projects/lendingclub-risk/outputs/eda"
OUTPUT_CHARTS = "/Users/suparna_deshmukh/projects/lendingclub-risk/outputs/eda/charts"

# Create output folders if they don't exist
os.makedirs(OUTPUT_CSV,    exist_ok=True)
os.makedirs(OUTPUT_CHARTS, exist_ok=True)

# Connect to DuckDB
con = duckdb.connect(DB_PATH)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_query(name, sql):
    """Run a SQL query, print the result, and return a DataFrame."""
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)
    df = con.execute(sql).df()
    print(df.to_string(index=False))
    return df

def save_csv(df, filename):
    """Save a DataFrame to a CSV file."""
    path = os.path.join(OUTPUT_CSV, filename)
    df.to_csv(path, index=False)
    print(f"  → Saved CSV: {path}")

def save_chart(fig, filename):
    """Save a matplotlib figure to the charts folder."""
    path = os.path.join(OUTPUT_CHARTS, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → Saved chart: {path}")

# A consistent style for all charts
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   1.2,
    "font.size":        11,
})

# ============================================================
# QUERY 1 — Overall Default Rate
# ============================================================

df1 = run_query(
    "Query 1: Overall Default Rate",
    """
    SELECT
        COUNT(*)                                          AS total_loans,
        SUM(is_default)                                   AS total_defaults,
        ROUND(100.0 * SUM(is_default) / COUNT(*), 2)     AS default_rate_pct
    FROM loans
    """
)
save_csv(df1, "01_overall_default_rate.csv")

fig, ax = plt.subplots(figsize=(5, 4))
categories = ["Non-Default", "Default"]
values     = [
    df1["total_loans"][0] - df1["total_defaults"][0],
    df1["total_defaults"][0]
]
bars = ax.bar(categories, values, color=["#4CAF50", "#F44336"], width=0.5)
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
ax.set_title("Overall Loan Outcomes", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Loans")
ax.set_ylim(0, max(values) * 1.15)
save_chart(fig, "01_overall_default_rate.png")

# ============================================================
# QUERY 2 — Default Rate by Grade
# ============================================================

df2 = run_query(
    "Query 2: Default Rate by Grade (A–G)",
    """
    SELECT
        grade,
        COUNT(*)                                          AS total_loans,
        SUM(is_default)                                   AS defaults,
        ROUND(100.0 * SUM(is_default) / COUNT(*), 2)     AS default_rate_pct
    FROM loans
    GROUP BY grade
    ORDER BY default_rate_pct DESC
    """
)
save_csv(df2, "02_default_rate_by_grade.csv")

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d32f2f", "#e57373", "#ff8a65", "#ffb74d", "#81c784", "#4db6ac", "#4fc3f7"]
bars = ax.bar(df2["grade"], df2["default_rate_pct"], color=colors[:len(df2)], width=0.6)
ax.bar_label(bars, fmt="{:.1f}%", padding=4, fontsize=10)
ax.set_title("Default Rate by Loan Grade", fontsize=14, fontweight="bold")
ax.set_xlabel("Loan Grade")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, df2["default_rate_pct"].max() * 1.2)
save_chart(fig, "02_default_rate_by_grade.png")

# ============================================================
# QUERY 3 — Default Rate by Home Ownership
# ============================================================

df3 = run_query(
    "Query 3: Default Rate by Home Ownership",
    """
    SELECT
        home_ownership,
        COUNT(*)                                          AS total_loans,
        SUM(is_default)                                   AS defaults,
        ROUND(100.0 * SUM(is_default) / COUNT(*), 2)     AS default_rate_pct
    FROM loans
    GROUP BY home_ownership
    ORDER BY default_rate_pct DESC
    """
)
save_csv(df3, "03_default_rate_by_home_ownership.csv")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df3["home_ownership"], df3["default_rate_pct"], color="#5C85D6", width=0.6)
ax.bar_label(bars, fmt="{:.1f}%", padding=4, fontsize=10)
ax.set_title("Default Rate by Home Ownership", fontsize=14, fontweight="bold")
ax.set_xlabel("Home Ownership")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, df3["default_rate_pct"].max() * 1.2)
save_chart(fig, "03_default_rate_by_home_ownership.png")

# ============================================================
# QUERY 4 — Avg and Median Interest Rate by Grade
# ============================================================

df4 = run_query(
    "Query 4: Average & Median Interest Rate by Grade",
    """
    SELECT
        grade,
        ROUND(AVG(int_rate), 2)                           AS avg_int_rate,
        ROUND(MEDIAN(int_rate), 2)                        AS median_int_rate,
        COUNT(*)                                          AS total_loans
    FROM loans
    GROUP BY grade
    ORDER BY grade
    """
)
save_csv(df4, "04_interest_rate_by_grade.csv")

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(df4))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], df4["avg_int_rate"],    width, label="Average",  color="#42A5F5")
bars2 = ax.bar([i + width/2 for i in x], df4["median_int_rate"], width, label="Median",   color="#AB47BC")
ax.bar_label(bars1, fmt="{:.1f}%", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="{:.1f}%", padding=3, fontsize=9)
ax.set_title("Interest Rate by Grade (Average vs Median)", fontsize=14, fontweight="bold")
ax.set_xlabel("Loan Grade")
ax.set_ylabel("Interest Rate (%)")
ax.set_xticks(list(x))
ax.set_xticklabels(df4["grade"])
ax.legend()
ax.set_ylim(0, df4["avg_int_rate"].max() * 1.25)
save_chart(fig, "04_interest_rate_by_grade.png")

# ============================================================
# QUERY 5 — Loan Amount by Purpose
# ============================================================

df5 = run_query(
    "Query 5: Loan Count & Average Amount by Purpose",
    """
    SELECT
        purpose,
        COUNT(*)                                          AS total_loans,
        ROUND(AVG(loan_amnt), 0)                          AS avg_loan_amount
    FROM loans
    GROUP BY purpose
    ORDER BY avg_loan_amount DESC
    """
)
save_csv(df5, "05_loan_amount_by_purpose.csv")

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(df5["purpose"], df5["avg_loan_amount"], color="#26A69A")
ax.bar_label(bars, fmt="${:,.0f}", padding=4, fontsize=9)
ax.set_title("Average Loan Amount by Purpose", fontsize=14, fontweight="bold")
ax.set_xlabel("Average Loan Amount ($)")
ax.set_ylabel("Purpose")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
save_chart(fig, "05_loan_amount_by_purpose.png")

# ============================================================
# QUERY 6 — Default Rate by Employment Length Bucket
# ============================================================

df6 = run_query(
    "Query 6: Default Rate by Employment Length",
    """
    SELECT
        CASE
            WHEN emp_length IN ('< 1 year', '1 year')         THEN '0–1 year'
            WHEN emp_length IN ('2 years', '3 years')         THEN '1–3 years'
            WHEN emp_length IN ('4 years', '5 years')         THEN '3–5 years'
            WHEN emp_length IN ('6 years', '7 years',
                                '8 years', '9 years',
                                '10+ years')                  THEN '5+ years'
            ELSE 'Unknown'
        END                                                   AS emp_bucket,
        COUNT(*)                                              AS total_loans,
        SUM(is_default)                                       AS defaults,
        ROUND(100.0 * SUM(is_default) / COUNT(*), 2)         AS default_rate_pct
    FROM loans
    GROUP BY emp_bucket
    ORDER BY
        CASE emp_bucket
            WHEN '0–1 year'  THEN 1
            WHEN '1–3 years' THEN 2
            WHEN '3–5 years' THEN 3
            WHEN '5+ years'  THEN 4
            ELSE 5
        END
    """
)
save_csv(df6, "06_default_rate_by_emp_length.csv")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df6["emp_bucket"], df6["default_rate_pct"], color="#FF7043", width=0.6)
ax.bar_label(bars, fmt="{:.1f}%", padding=4, fontsize=10)
ax.set_title("Default Rate by Employment Length", fontsize=14, fontweight="bold")
ax.set_xlabel("Employment Length")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, df6["default_rate_pct"].max() * 1.2)
save_chart(fig, "06_default_rate_by_emp_length.png")

# ============================================================
# QUERY 7 — DTI: Defaulters vs Non-Defaulters
# ============================================================

df7 = run_query(
    "Query 7: Average DTI — Defaulters vs Non-Defaulters",
    """
    SELECT
        CASE WHEN is_default = 1 THEN 'Defaulted' ELSE 'Fully Paid' END  AS outcome,
        ROUND(AVG(dti), 2)                                                AS avg_dti,
        ROUND(MEDIAN(dti), 2)                                             AS median_dti,
        COUNT(*)                                                          AS total_loans
    FROM loans
    GROUP BY is_default
    ORDER BY is_default DESC
    """
)
save_csv(df7, "07_dti_defaulters_vs_nondefaulters.csv")

fig, ax = plt.subplots(figsize=(6, 5))
x = range(len(df7))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], df7["avg_dti"],    width, label="Average DTI",  color="#EF5350")
bars2 = ax.bar([i + width/2 for i in x], df7["median_dti"], width, label="Median DTI",   color="#EC407A")
ax.bar_label(bars1, fmt="{:.1f}", padding=3, fontsize=10)
ax.bar_label(bars2, fmt="{:.1f}", padding=3, fontsize=10)
ax.set_title("Debt-to-Income Ratio:\nDefaulters vs Non-Defaulters", fontsize=14, fontweight="bold")
ax.set_ylabel("DTI")
ax.set_xticks(list(x))
ax.set_xticklabels(df7["outcome"])
ax.legend()
ax.set_ylim(0, df7["avg_dti"].max() * 1.3)
save_chart(fig, "07_dti_defaulters_vs_nondefaulters.png")

# ============================================================
# QUERY 8 — Revolving Utilization Buckets vs Default Rate
# ============================================================

df8 = run_query(
    "Query 8: Default Rate by Revolving Utilization Bucket",
    """
    SELECT
        CASE
            WHEN CAST(revol_util AS DOUBLE) < 20  THEN '0–20%'
            WHEN CAST(revol_util AS DOUBLE) < 40  THEN '20–40%'
            WHEN CAST(revol_util AS DOUBLE) < 60  THEN '40–60%'
            WHEN CAST(revol_util AS DOUBLE) < 80  THEN '60–80%'
            ELSE '80%+'
        END                                                   AS revol_bucket,
        COUNT(*)                                              AS total_loans,
        SUM(is_default)                                       AS defaults,
        ROUND(100.0 * SUM(is_default) / COUNT(*), 2)         AS default_rate_pct
    FROM loans
    WHERE revol_util IS NOT NULL AND revol_util != 'nan'
    GROUP BY revol_bucket
    ORDER BY
        CASE revol_bucket
            WHEN '0–20%'   THEN 1
            WHEN '20–40%'  THEN 2
            WHEN '40–60%'  THEN 3
            WHEN '60–80%'  THEN 4
            ELSE 5
        END
    """
)
save_csv(df8, "08_default_rate_by_revol_util.csv")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df8["revol_bucket"], df8["default_rate_pct"], color="#7E57C2", width=0.6)
ax.bar_label(bars, fmt="{:.1f}%", padding=4, fontsize=10)
ax.set_title("Default Rate by Revolving Credit Utilization", fontsize=14, fontweight="bold")
ax.set_xlabel("Revolving Utilization Bucket")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, df8["default_rate_pct"].max() * 1.2)
save_chart(fig, "08_default_rate_by_revol_util.png")

# ============================================================
# QUERY 9 — Credit Inquiries in Last 6 Months vs Default Rate
# ============================================================

df9 = run_query(
    "Query 9: Default Rate by # Credit Inquiries (Last 6 Months)",
    """
    SELECT
        CAST(inq_last_6mths AS INTEGER)                       AS inquiries,
        COUNT(*)                                              AS total_loans,
        SUM(is_default)                                       AS defaults,
        ROUND(100.0 * SUM(is_default) / COUNT(*), 2)         AS default_rate_pct
    FROM loans
    WHERE inq_last_6mths IS NOT NULL
      AND CAST(inq_last_6mths AS INTEGER) <= 10
    GROUP BY inquiries
    ORDER BY inquiries
    """
)
save_csv(df9, "09_default_rate_by_inquiries.csv")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    df9["inquiries"].astype(str),
    df9["default_rate_pct"],
    color="#26C6DA",
    width=0.6
)
ax.bar_label(bars, fmt="{:.1f}%", padding=4, fontsize=9)
ax.set_title("Default Rate by # Credit Inquiries (Last 6 Months)", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Inquiries")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, df9["default_rate_pct"].max() * 1.2)
save_chart(fig, "09_default_rate_by_inquiries.png")

# ============================================================
# QUERY 10 — Annual Income Deciles vs Default Rate
# ============================================================

df10 = run_query(
    "Query 10: Default Rate by Annual Income Decile",
    """
    WITH deciles AS (
        SELECT
            is_default,
            annual_inc,
            NTILE(10) OVER (ORDER BY annual_inc)              AS decile
        FROM loans
        WHERE annual_inc IS NOT NULL
    ),
    decile_bounds AS (
        SELECT
            decile,
            COUNT(*)                                          AS total_loans,
            SUM(is_default)                                   AS defaults,
            ROUND(100.0 * SUM(is_default) / COUNT(*), 2)     AS default_rate_pct,
            ROUND(MIN(annual_inc), 0)                         AS min_income,
            ROUND(MAX(annual_inc), 0)                         AS max_income
        FROM deciles
        GROUP BY decile
    )
    SELECT
        decile,
        '$' || CAST(CAST(min_income AS INTEGER) AS VARCHAR)
            || '–$' || CAST(CAST(max_income AS INTEGER) AS VARCHAR) AS income_range,
        total_loans,
        defaults,
        default_rate_pct
    FROM decile_bounds
    ORDER BY decile
    """
)
save_csv(df10, "10_default_rate_by_income_decile.csv")

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(
    [f"D{d}" for d in df10["decile"]],
    df10["default_rate_pct"],
    color="#66BB6A",
    width=0.6
)
ax.bar_label(bars, fmt="{:.1f}%", padding=4, fontsize=9)
ax.set_title("Default Rate by Annual Income Decile\n(D1 = Lowest Income, D10 = Highest Income)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Income Decile")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, df10["default_rate_pct"].max() * 1.2)

# Add income range labels on x-axis as a second line
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(range(len(df10)))
ax2.set_xticklabels(df10["income_range"], rotation=35, ha="left", fontsize=7)
ax2.set_xlabel("Income Range ($)", fontsize=10)

save_chart(fig, "10_default_rate_by_income_decile.png")

# ============================================================
# DONE
# ============================================================

con.close()

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"  CSVs saved to:   {OUTPUT_CSV}")
print(f"  Charts saved to: {OUTPUT_CHARTS}")
print("=" * 60)