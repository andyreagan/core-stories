"""Build a timeseries matrix pickle from gutenberg-007 pre-computed word-count CSVs.

The gutenberg-007/{id}.csv.gz files are (10222 x 200) integer matrices:
  - rows: labMT words in labMT1.txt order
  - columns: 200 sliding-window positions across the book

The sentiment timeseries is computed as:
  timeseries[col] = labMT_scores[mask] @ count_matrix[mask, col] / count_matrix[mask, col].sum()

where mask removes stop words (|score - 5| < 1.0).

This covers all 1385 books in the canonical filter, replacing the prior
537-book subset that was limited to the hedonometer display-subset CSVs.

Usage:
    uv run python data/build_matrix.py
"""

import sqlite3
import os
import gzip
import pickle
import numpy as np

# Paths (env-overridable for smoke tests / alternate-corpus runs)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get(
    "CORE_STORIES_DB_PATH",
    os.path.normpath(os.path.join(REPO_ROOT, "..", "core-stories-code", "database", "db.sqlite3")),
)
CSV_DIR = os.environ.get("CORE_STORIES_CSV_DIR", os.path.join(REPO_ROOT, "data", "gutenberg-007"))
OUT_DIR = os.environ.get("CORE_STORIES_OUT_DIR", os.path.join(REPO_ROOT, "data", "gutenberg"))
os.makedirs(OUT_DIR, exist_ok=True)

# Filters (matching the Makefile: 40 false)
FILTERS = {
    "min_dl": 40,
    "length": [20000, 100000],
    "P": True,
    "n_points": 200,
    "salad": False,
}

def get_version_str(filters):
    P_str = "P-" if filters["P"] else ""
    salad_str = "-salad" if filters["salad"] else ""
    version = "{0}{1:.0f}K-{2:.0f}K-{3}dl-{4}pt{5}".format(
        P_str,
        filters["length"][0] / 1000.0,
        filters["length"][1] / 1000.0,
        filters["min_dl"],
        filters["n_points"],
        salad_str,
    )
    return version

version_str = get_version_str(FILTERS)
MATRIX_CACHE = os.path.join(OUT_DIR, f"timeseries-matrix-cache-{version_str}.p")
BOOKS_CACHE = os.path.join(OUT_DIR, f"books-{version_str}.p")

print(f"Version string: {version_str}")
print(f"Will write matrix to: {MATRIX_CACHE}")

# Load the full labMT1 word list (10222 words)
labmt_path = os.path.join(
    REPO_ROOT,
    ".venv/lib/python3.11/site-packages/labMTsimple/data/LabMT/labMT1.txt"
)
words = []
scores = []
with open(labmt_path) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            words.append(parts[0])
            scores.append(float(parts[2]))
scores = np.array(scores)
print(f"Loaded {len(scores)} labMT words")

# Stop-word mask: keep words where |score - 5| >= 1.0
CENTER = 5.0
STOP_VAL = 1.0
mask = np.abs(scores - CENTER) >= STOP_VAL
scores_masked = scores[mask]
print(f"After stopper mask: {mask.sum()} words retained")

# Query books from SQLite (without Django)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(
    """
    SELECT id, gutenberg_id, title, downloads, length, numUniqWords, locc_string, exclude
    FROM library_book
    WHERE exclude=0
      AND length > ?
      AND length <= ?
      AND downloads >= ?
      AND lang_code_id=0
      AND locc_with_P=1
    ORDER BY id
    """,
    (FILTERS["length"][0], FILTERS["length"][1], FILTERS["min_dl"]),
)
rows = cur.fetchall()
conn.close()

print(f"Books matching filter: {len(rows)}")

# Build timeseries matrix from CSV files
book_records = []
matrix_rows = []
missing_count = 0
error_count = 0

for (pk, gutenberg_id, title, downloads, length, numUniqWords, locc_string, exclude) in rows:
    csv_file = os.path.join(CSV_DIR, f"{gutenberg_id}.csv.gz")
    if not os.path.isfile(csv_file):
        missing_count += 1
        continue
    try:
        with gzip.open(csv_file, 'rt') as f:
            # Read (10222 x 200) integer matrix row by row
            csv_rows = []
            for line in f:
                vals = [int(x) for x in line.strip().rstrip(',').split(',') if x]
                csv_rows.append(vals)
        mat = np.array(csv_rows)  # (10222, 200)
        if mat.shape[0] != len(scores):
            error_count += 1
            continue
        # Apply stopper mask and compute valence for each of 200 windows
        mat_masked = mat[mask, :].astype(np.float32)
        col_sums = mat_masked.sum(axis=0)
        numerator = scores_masked @ mat_masked
        ts200 = np.where(col_sums > 0, numerator / col_sums, CENTER)
    except Exception as e:
        error_count += 1
        print(f"  Error on book {gutenberg_id}: {e}")
        continue

    book_records.append({
        "pk": pk,
        "gutenberg_id": gutenberg_id,
        "title": title,
        "downloads": downloads,
        "length": length,
        "numUniqWords": numUniqWords or 0,
        "locc_string": locc_string or "",
        "exclude": bool(exclude),
    })
    matrix_rows.append(ts200)

print(f"  Built timeseries for {len(book_records)} books "
      f"(missing={missing_count}, errors={error_count})")

big_matrix = np.array(matrix_rows)
print(f"  Matrix shape: {big_matrix.shape}")

pickle.dump(big_matrix, open(MATRIX_CACHE, "wb"), pickle.HIGHEST_PROTOCOL)
print(f"  Saved matrix to {MATRIX_CACHE}")

pickle.dump(book_records, open(BOOKS_CACHE, "wb"), pickle.HIGHEST_PROTOCOL)
print(f"  Saved book records to {BOOKS_CACHE}")
