# ============================================================
# 04_model.py
# LendingClub Risk Project — Logistic Regression Model
# ============================================================
# Step 1 — Load and prepare features
# Step 2 — Train/test split
# Step 3 — Build a sklearn Pipeline
# Step 4 — Evaluate (ROC-AUC, PR-AUC, confusion matrix, curves)
# Step 5 — Per-segment evaluation
# Step 6 — Top coefficients chart
# Step 7 — Save the model
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split
from sklearn.pipeline           import Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

PARQUET_PATH     = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/loans_segmented.parquet"
OUT_FIGURES      = "/Users/suparna_deshmukh/projects/lendingclub-risk/outputs/figures"
OUT_SEG_METRICS  = "/Users/suparna_deshmukh/projects/lendingclub-risk/outputs/per_segment_metrics.csv"
OUT_MODEL_DIR    = "/Users/suparna_deshmukh/projects/lendingclub-risk/models"
OUT_MODEL_PATH   = os.path.join(OUT_MODEL_DIR, "logreg.pkl")

os.makedirs(OUT_FIGURES,     exist_ok=True)
os.makedirs(OUT_MODEL_DIR,   exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   1.2,
    "font.size":        11,
})

# ============================================================
# STEP 1 — Load and Prepare Features
# ============================================================

print("=" * 60)
print("  Step 1/7 — Loading data and preparing features...")
print("=" * 60)

NUM_COLS = [
    "loan_amnt", "int_rate", "dti", "annual_inc",
    "revol_util", "inq_last_6mths", "delinq_2yrs",
    "open_acc", "total_acc",
]
CAT_COLS = [
    "grade", "home_ownership", "verification_status",
    "purpose", "term",
]
TARGET   = "is_default"
SEGMENT  = "segment"

# Load the parquet file
df_raw = pd.read_parquet(PARQUET_PATH)

# The segmented parquet only kept RFM columns, so we merge
# the missing columns back in from the original 100k file
ORIG_PATH = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/loans_100k.parquet"
df_orig   = pd.read_parquet(ORIG_PATH)
missing   = ["revol_util", "inq_last_6mths", "delinq_2yrs",
             "open_acc", "total_acc", "home_ownership",
             "verification_status", "term"]
df_raw    = df_raw.join(df_orig[missing])
print(f"  Rows loaded: {len(df_raw):,}")

# Keep only the columns we need
keep_cols = NUM_COLS + CAT_COLS + [TARGET, SEGMENT]
df = df_raw[keep_cols].copy()

# Convert numerical columns to actual numbers
# (they may have been stored as strings from the earlier type-fix)
for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert target to integer
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("Int64")

# Drop rows with any missing values in our chosen columns
before = len(df)
df = df.dropna(subset=NUM_COLS + CAT_COLS + [TARGET])
after  = len(df)
print(f"  Rows after dropping nulls: {after:,}  (dropped {before - after:,})")
print(f"  Default rate in clean data: {df[TARGET].mean()*100:.2f}%")

# ============================================================
# STEP 2 — Train / Test Split
# ============================================================

print("\n" + "=" * 60)
print("  Step 2/7 — Train/test split (80/20, stratified)...")
print("=" * 60)

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET].astype(int)
segments = df[SEGMENT]

X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
    X, y, segments,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print(f"  Training rows : {len(X_train):,}")
print(f"  Test rows     : {len(X_test):,}")
print(f"  Train default rate: {y_train.mean()*100:.2f}%")
print(f"  Test  default rate: {y_test.mean()*100:.2f}%")

# ============================================================
# STEP 3 — Build the Pipeline
# ============================================================

print("\n" + "=" * 60)
print("  Step 3/7 — Building and fitting the pipeline...")
print("=" * 60)

# ColumnTransformer applies different transformations to
# different column groups at the same time
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(),                            NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False),          CAT_COLS),
    ]
)

# Pipeline chains preprocessing and model into one object
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model",        LogisticRegression(
                         max_iter=1000,
                         class_weight="balanced",
                         random_state=42,
                     )),
])

pipeline.fit(X_train, y_train)
print("  Pipeline fitted successfully.")

# ============================================================
# STEP 4 — Evaluate the Model
# ============================================================

print("\n" + "=" * 60)
print("  Step 4/7 — Evaluating model performance...")
print("=" * 60)

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]   # probability of default
y_pred       = pipeline.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc  = average_precision_score(y_test, y_pred_proba)

print(f"\n  ROC-AUC Score : {roc_auc:.4f}")
print(f"  PR-AUC Score  : {pr_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fully Paid", "Default"]))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted: Paid", "Predicted: Default"],
    yticklabels=["Actual: Paid",    "Actual: Default"],
    linewidths=0.5, ax=ax
)
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Actual Label",    fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)
path_cm = os.path.join(OUT_FIGURES, "confusion_matrix.png")
fig.savefig(path_cm, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\n  → Saved: {path_cm}")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#1565C0", lw=2,
        label=f"Logistic Regression (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1.5, linestyle="--",
        label="Random Classifier (AUC = 0.500)")
ax.fill_between(fpr, tpr, alpha=0.08, color="#1565C0")
ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Recall)",            fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
path_roc = os.path.join(OUT_FIGURES, "roc_curve.png")
fig.savefig(path_roc, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  → Saved: {path_roc}")

# --- Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
baseline = y_test.mean()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, color="#C62828", lw=2,
        label=f"Logistic Regression (PR-AUC = {pr_auc:.3f})")
ax.axhline(baseline, color="#aaaaaa", lw=1.5, linestyle="--",
           label=f"Random Classifier (baseline = {baseline:.3f})")
ax.fill_between(recall, precision, alpha=0.08, color="#C62828")
ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Recall",    fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
path_pr = os.path.join(OUT_FIGURES, "pr_curve.png")
fig.savefig(path_pr, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  → Saved: {path_pr}")

# ============================================================
# STEP 5 — Per-Segment Evaluation
# ============================================================

print("\n" + "=" * 60)
print("  Step 5/7 — Per-segment evaluation...")
print("=" * 60)

seg_records = []
for seg in sorted(seg_test.unique()):
    mask           = seg_test == seg
    y_seg          = y_test[mask]
    proba_seg      = y_pred_proba[mask]

    # Need at least both classes present to compute AUC
    if y_seg.nunique() < 2:
        auc_seg = float("nan")
    else:
        auc_seg = roc_auc_score(y_seg, proba_seg)

    seg_records.append({
        "segment":              int(seg),
        "test_rows":            int(mask.sum()),
        "actual_default_rate":  round(float(y_seg.mean()) * 100, 2),
        "mean_pred_proba":      round(float(proba_seg.mean()) * 100, 2),
        "roc_auc":              round(float(auc_seg), 4),
    })

seg_df = pd.DataFrame(seg_records)
print(seg_df.to_string(index=False))
seg_df.to_csv(OUT_SEG_METRICS, index=False)
print(f"\n  → Saved: {OUT_SEG_METRICS}")

# ============================================================
# STEP 6 — Top Coefficients
# ============================================================

print("\n" + "=" * 60)
print("  Step 6/7 — Extracting top 15 coefficients...")
print("=" * 60)

# Get feature names from the fitted preprocessor
num_feature_names = NUM_COLS
cat_feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CAT_COLS)
    .tolist()
)
all_feature_names = num_feature_names + cat_feature_names

# Get the model coefficients
coef = pipeline.named_steps["model"].coef_[0]

coef_df = (
    pd.DataFrame({"feature": all_feature_names, "coefficient": coef})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .head(15)
    .sort_values("coefficient")     # sort for chart readability
)

print(coef_df[["feature", "coefficient"]].to_string(index=False))

colors = ["#F44336" if c > 0 else "#4CAF50" for c in coef_df["coefficient"]]

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(coef_df["feature"], coef_df["coefficient"],
               color=colors, edgecolor="white", height=0.65)
ax.axvline(0, color="#333333", linewidth=1.2)
ax.set_title(
    "Top 15 Features by Logistic Regression Coefficient\n"
    "(Red = increases default risk, Green = decreases default risk)",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Coefficient Value", fontsize=11)
ax.set_ylabel("")
# Add value labels on each bar
for bar, val in zip(bars, coef_df["coefficient"]):
    x_pos = val + (0.01 if val >= 0 else -0.01)
    ha    = "left" if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=9)

path_coef = os.path.join(OUT_FIGURES, "coefficients.png")
fig.savefig(path_coef, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\n  → Saved: {path_coef}")

# ============================================================
# STEP 7 — Save the Model
# ============================================================

print("\n" + "=" * 60)
print("  Step 7/7 — Saving model to disk...")
print("=" * 60)

with open(OUT_MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"  → Model saved: {OUT_MODEL_PATH}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  ALL DONE — FINAL SUMMARY")
print("=" * 60)
print(f"  ROC-AUC        : {roc_auc:.4f}")
print(f"  PR-AUC         : {pr_auc:.4f}")
print(f"  Model saved    : {OUT_MODEL_PATH}")
print(f"  Figures saved  : {OUT_FIGURES}/")
print(f"  Seg metrics    : {OUT_SEG_METRICS}")
print("=" * 60)