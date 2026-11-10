# ============================================================
# 03_segmentation.py
# LendingClub Risk Project — Borrower Segmentation
# ============================================================
# This script:
#   Step 1 — Pulls and engineers R, F, M features via SQL
#   Step 2 — Scales features and runs K-Means (k=5)
#   Step 3 — Labels segments by actual default rate
#   Step 4 — Builds segment profiles and saves outputs
#   Step 5 — Creates 3 visualizations
# ============================================================

import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import warnings
warnings.filterwarnings("ignore")   # suppress minor sklearn warnings

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH        = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/lc.duckdb"
OUT_PARQUET    = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/loans_segmented.parquet"
OUT_PROFILES   = "/Users/suparna_deshmukh/projects/lendingclub-risk/outputs/segment_profiles.csv"
OUT_FIGURES    = "/Users/suparna_deshmukh/projects/lendingclub-risk/outputs/figures"

os.makedirs(os.path.dirname(OUT_PROFILES), exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   1.2,
    "font.size":        11,
})

# ============================================================
# STEP 1 — Feature Engineering via SQL
# ============================================================

print("=" * 60)
print("  Step 1/4 — Feature engineering via SQL...")
print("=" * 60)

con = duckdb.connect(DB_PATH)

# First compute the 99th percentile of the leverage ratio
# so we can cap outliers
pct99_row = con.execute("""
    SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY
        CAST(loan_amnt AS DOUBLE) / NULLIF(CAST(annual_inc AS DOUBLE), 0)
    ) AS pct99
    FROM loans
    WHERE annual_inc IS NOT NULL
      AND CAST(annual_inc AS DOUBLE) > 0
""").fetchone()

pct99_leverage = round(float(pct99_row[0]), 6)
print(f"  99th percentile of leverage ratio: {pct99_leverage:.4f}")

# Pull all features, applying the RFM definitions
df = con.execute(f"""
    SELECT
        -- R: months since last delinquency (NULL = no delinquency → safer → set to 999)
        COALESCE(CAST(mths_since_last_delinq AS DOUBLE), 999)   AS R,

        -- F: inquiries + delinquencies in recent periods
        COALESCE(CAST(inq_last_6mths AS DOUBLE), 0)
          + COALESCE(CAST(delinq_2yrs AS DOUBLE), 0)            AS F,

        -- M: leverage ratio, capped at 99th percentile
        LEAST(
            COALESCE(CAST(loan_amnt AS DOUBLE), 0)
              / NULLIF(COALESCE(CAST(annual_inc AS DOUBLE), 1), 0),
            {pct99_leverage}
        )                                                        AS M,

        -- Extra columns for profiling
        is_default,
        grade,
        CAST(dti        AS DOUBLE)                              AS dti,
        CAST(int_rate   AS DOUBLE)                              AS int_rate,
        CAST(loan_amnt  AS DOUBLE)                              AS loan_amnt,
        CAST(annual_inc AS DOUBLE)                              AS annual_inc,
        purpose

    FROM loans
    WHERE annual_inc IS NOT NULL
      AND CAST(annual_inc AS DOUBLE) > 0
""").df()

con.close()

print(f"  Rows loaded: {len(df):,}")
print(f"  Columns: {list(df.columns)}")

# ============================================================
# STEP 2 — K-Means Clustering
# ============================================================

print("\n" + "=" * 60)
print("  Step 2/4 — Scaling features and running K-Means (k=5)...")
print("=" * 60)

features = ["R", "F", "M"]
X = df[features].values

# StandardScaler transforms each feature so it has
# mean=0 and standard deviation=1. This stops any single
# feature from dominating just because of its scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-Means with k=5 clusters
# n_init=20 means it tries 20 different random starting points
# and picks the best result — more reliable than the default
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
kmeans.fit(X_scaled)

# Attach raw cluster labels (0–4) back to the DataFrame
df["raw_cluster"] = kmeans.labels_

print("  K-Means complete.")
print("  Raw cluster sizes:")
print(df["raw_cluster"].value_counts().sort_index().to_string())

# ============================================================
# STEP 3 — Re-label Clusters by Actual Default Rate
#           Segment 1 = safest, Segment 5 = riskiest
# ============================================================

print("\n" + "=" * 60)
print("  Step 3/4 — Labelling segments by default rate...")
print("=" * 60)

# Compute actual default rate per raw cluster
cluster_default_rates = (
    df.groupby("raw_cluster")["is_default"]
    .mean()
    .reset_index()
    .rename(columns={"is_default": "default_rate"})
    .sort_values("default_rate")          # sort ascending: lowest risk first
    .reset_index(drop=True)
)

# Build a mapping: raw cluster number → segment label 1–5
cluster_default_rates["segment"] = range(1, 6)
cluster_to_segment = dict(
    zip(cluster_default_rates["raw_cluster"],
        cluster_default_rates["segment"])
)

df["segment"] = df["raw_cluster"].map(cluster_to_segment)

print("  Segment mapping (raw cluster → segment):")
print(cluster_default_rates.to_string(index=False))

# ============================================================
# STEP 4 — Segment Profiles
# ============================================================

print("\n" + "=" * 60)
print("  Step 4/4 — Building segment profiles...")
print("=" * 60)

total_rows = len(df)

profiles = []
for seg in sorted(df["segment"].unique()):
    chunk = df[df["segment"] == seg]

    # Most common grade in this segment
    top_grade = chunk["grade"].value_counts().idxmax()

    profiles.append({
        "segment":          seg,
        "count":            len(chunk),
        "pct_of_total":     round(100.0 * len(chunk) / total_rows, 1),
        "mean_R":           round(chunk["R"].mean(), 1),
        "mean_F":           round(chunk["F"].mean(), 2),
        "mean_M":           round(chunk["M"].mean(), 4),
        "mean_dti":         round(chunk["dti"].mean(), 2),
        "top_grade":        top_grade,
        "default_rate_pct": round(chunk["is_default"].mean() * 100, 2),
        "mean_loan_amnt":   round(chunk["loan_amnt"].mean(), 0),
    })

profile_df = pd.DataFrame(profiles)

print("\n  SEGMENT PROFILES:")
print(profile_df.to_string(index=False))

# Save outputs
df.to_parquet(OUT_PARQUET, index=False)
print(f"\n  → Full dataset saved: {OUT_PARQUET}")

profile_df.to_csv(OUT_PROFILES, index=False)
print(f"  → Segment profiles saved: {OUT_PROFILES}")

# ============================================================
# STEP 5 — Visualizations
# ============================================================

print("\n" + "=" * 60)
print("  Step 5/4 — Creating charts...")
print("=" * 60)

SEG_COLORS = ["#4CAF50", "#8BC34A", "#FFC107", "#FF7043", "#F44336"]
seg_labels  = [f"Seg {i}" for i in profile_df["segment"]]

# ----------------------------------------------------------
# Chart 1 — Default Rate per Segment
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    seg_labels,
    profile_df["default_rate_pct"],
    color=SEG_COLORS,
    width=0.6,
    edgecolor="white"
)
ax.bar_label(bars, fmt="{:.1f}%", padding=5, fontsize=11, fontweight="bold")
ax.set_title("Default Rate by Risk Segment\n(Segment 1 = Safest, Segment 5 = Riskiest)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Risk Segment", fontsize=12)
ax.set_ylabel("Default Rate (%)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, profile_df["default_rate_pct"].max() * 1.22)

path1 = os.path.join(OUT_FIGURES, "01_default_rate_by_segment.png")
fig.savefig(path1, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  → Chart saved: {path1}")

# ----------------------------------------------------------
# Chart 2 — Average Loan Amount per Segment
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    seg_labels,
    profile_df["mean_loan_amnt"],
    color=SEG_COLORS,
    width=0.6,
    edgecolor="white"
)
ax.bar_label(
    bars,
    labels=[f"${v:,.0f}" for v in profile_df["mean_loan_amnt"]],
    padding=5, fontsize=10, fontweight="bold"
)
ax.set_title("Average Loan Amount by Risk Segment",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Risk Segment", fontsize=12)
ax.set_ylabel("Average Loan Amount ($)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_ylim(0, profile_df["mean_loan_amnt"].max() * 1.2)

path2 = os.path.join(OUT_FIGURES, "02_avg_loan_amount_by_segment.png")
fig.savefig(path2, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  → Chart saved: {path2}")

# ----------------------------------------------------------
# Chart 3 — Grade Distribution (stacked bar) per Segment
# ----------------------------------------------------------

# Count each grade within each segment, then pivot to wide format
grade_dist = (
    df.groupby(["segment", "grade"])
    .size()
    .reset_index(name="count")
)
grade_pivot = (
    grade_dist
    .pivot(index="segment", columns="grade", values="count")
    .fillna(0)
)

# Convert to row percentages so the stacked bars all reach 100%
grade_pct = grade_pivot.div(grade_pivot.sum(axis=1), axis=0) * 100

# Ensure grades are ordered A → G
all_grades    = ["A", "B", "C", "D", "E", "F", "G"]
present_grades = [g for g in all_grades if g in grade_pct.columns]
grade_pct      = grade_pct[present_grades]

grade_colors = ["#1565C0", "#42A5F5", "#26A69A", "#FFA726", "#EF5350", "#AB47BC", "#8D6E63"]

fig, ax = plt.subplots(figsize=(9, 6))
bottom = np.zeros(len(grade_pct))

for i, grade in enumerate(present_grades):
    values = grade_pct[grade].values
    bars = ax.bar(
        [f"Seg {s}" for s in grade_pct.index],
        values,
        bottom=bottom,
        label=f"Grade {grade}",
        color=grade_colors[i % len(grade_colors)],
        edgecolor="white",
        linewidth=0.5
    )
    # Only add a label if the slice is big enough to read
    for j, (val, bot) in enumerate(zip(values, bottom)):
        if val > 5:
            ax.text(
                j, bot + val / 2,
                f"{val:.0f}%",
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold"
            )
    bottom += values

ax.set_title("Grade Distribution Within Each Risk Segment",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Risk Segment", fontsize=12)
ax.set_ylabel("Share of Segment (%)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 110)
ax.legend(title="Loan Grade", bbox_to_anchor=(1.01, 1), loc="upper left")

path3 = os.path.join(OUT_FIGURES, "03_grade_distribution_by_segment.png")
fig.savefig(path3, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  → Chart saved: {path3}")

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"  Segmented dataset : {OUT_PARQUET}")
print(f"  Segment profiles  : {OUT_PROFILES}")
print(f"  Charts            : {OUT_FIGURES}/")
print("=" * 60)