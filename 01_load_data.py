# ============================================================
# 01_load_data.py
# LendingClub Risk Project — Data Loading & Sampling Script
# ============================================================
# This script:
#   1. Reads a large CSV file in small pieces (chunks)
#   2. Filters to only loans with a clear outcome
#   3. Creates a 0/1 default label
#   4. Samples 100,000 rows (keeping the default ratio balanced)
#   5. Saves to a Parquet file
#   6. Loads that Parquet into a DuckDB database
#   7. Prints summary statistics
# ============================================================

# --- Step 0: Import the libraries we need ---
# These are pre-installed packages. If any are missing,
# the error message will tell you which one to install.

import pandas as pd          # for reading and manipulating data
import duckdb                # for creating the database
import os                    # for checking file paths

# ============================================================
# CONFIGURATION — change these paths if needed
# ============================================================

INPUT_CSV  = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/accepted_2007_to_2018Q4.csv"
OUTPUT_PQ  = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/loans_100k.parquet"
OUTPUT_DB  = "/Users/suparna_deshmukh/projects/lendingclub-risk/data/lc.duckdb"

SAMPLE_SIZE = 100_000        # total rows we want in the final sample
CHUNK_SIZE  = 100_000        # how many rows to read at a time from the CSV
RANDOM_SEED = 42             # makes the random sample reproducible

# Loan statuses we care about
KEEP_STATUSES = ["Fully Paid", "Charged Off", "Default"]

# ============================================================
# Step 1 & 2 & 3: Read CSV in chunks, filter, and label
# ============================================================

print("=" * 55)
print("Step 1/4 — Reading CSV in chunks and filtering rows...")
print("(This may take a few minutes for a 2-million-row file)")
print("=" * 55)

# We'll collect all the filtered chunks into this list
filtered_chunks = []

# pd.read_csv with chunksize reads the file a little at a time
# so we never have to hold the whole 2M rows in memory at once
chunk_number = 0
for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False):

    chunk_number += 1

    # --- Step 2: Keep only the loan statuses we want ---
    chunk = chunk[chunk["loan_status"].isin(KEEP_STATUSES)]

    # --- Step 3: Create the is_default column ---
    # 1 = Charged Off or Default (bad outcome)
    # 0 = Fully Paid (good outcome)
    chunk["is_default"] = chunk["loan_status"].apply(
        lambda status: 1 if status in ["Charged Off", "Default"] else 0
    )

    filtered_chunks.append(chunk)

    # Print a progress update every 10 chunks
    if chunk_number % 10 == 0:
        print(f"  Processed {chunk_number * CHUNK_SIZE:,} rows so far...")

print(f"\n  Done reading! Processed {chunk_number} chunks total.")

# Combine all the filtered chunks into one big DataFrame
print("\nCombining all filtered chunks into one table...")
df = pd.concat(filtered_chunks, ignore_index=True)
print(f"  Total rows after filtering: {len(df):,}")
print(f"  Defaults in full dataset:   {df['is_default'].sum():,}")

# ============================================================
# Step 4: Stratified sampling — 100,000 rows
# keeping the same default rate as the full filtered dataset
# ============================================================

print("\n" + "=" * 55)
print("Step 2/4 — Stratified sampling (100,000 rows)...")
print("=" * 55)

# groupby("is_default") splits the data into two groups:
#   group 0 → Fully Paid loans
#   group 1 → Charged Off / Default loans
# We then sample proportionally from each group so that the
# default rate in our 100k sample matches the full dataset.

df_sampled = (
    df.groupby("is_default", group_keys=False)
    .apply(
        lambda group: group.sample(
            n=round(SAMPLE_SIZE * len(group) / len(df)),
            random_state=RANDOM_SEED
        )
    )
    .reset_index(drop=True)
)

print(f"  Sampled dataset size: {len(df_sampled):,} rows")

# ============================================================
# Step 5: Save to Parquet
# ============================================================

print("\n" + "=" * 55)
print("Step 3/4 — Saving to Parquet file...")
print("=" * 55)

# Make sure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PQ), exist_ok=True)

# Fix mixed-type columns that confuse Parquet (e.g. the 'id' column)
for col in df_sampled.columns:
    if df_sampled[col].dtype == object:
        df_sampled[col] = df_sampled[col].astype(str)
df_sampled.to_parquet(OUTPUT_PQ, index=False)
print(f"  Parquet saved to: {OUTPUT_PQ}")

# ============================================================
# Step 6: Create DuckDB database and load the Parquet
# ============================================================

print("\n" + "=" * 55)
print("Step 4/4 — Creating DuckDB database...")
print("=" * 55)

# Connect to (or create) the DuckDB file
con = duckdb.connect(OUTPUT_DB)

# Drop the table if it already exists (so re-running is safe)
con.execute("DROP TABLE IF EXISTS loans")

# Create the 'loans' table directly from the Parquet file.
# DuckDB can read Parquet natively — no need to re-load the DataFrame.
con.execute(f"""
    CREATE TABLE loans AS
    SELECT * FROM read_parquet('{OUTPUT_PQ}')
""")

print(f"  DuckDB database saved to: {OUTPUT_DB}")
print(f"  Table 'loans' created successfully.")

con.close()

# ============================================================
# Step 7: Print summary statistics
# ============================================================

print("\n" + "=" * 55)
print("SUMMARY STATISTICS")
print("=" * 55)

# Pull stats directly from the database (already saved successfully)
con2 = duckdb.connect(OUTPUT_DB)
total_rows     = con2.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
total_defaults = con2.execute("SELECT COUNT(*) FROM loans WHERE is_default = '1'").fetchone()[0]
default_rate   = (total_defaults / total_rows) * 100
con2.close()

print(f"  Total rows:     {total_rows:,}")
print(f"  Total defaults: {total_defaults:,}")
print(f"  Default rate:   {default_rate:.2f}%")
print("=" * 55)
print("\nAll done! Your data is ready.")