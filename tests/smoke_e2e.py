"""End-to-end smoke test for the core-stories reproducibility chain.

Demonstrates that a fresh user can:
  1. Query the canonical book filter from SQLite.
  2. Download those books from Project Gutenberg.
  3. Run src/build_csvs.py to convert raw .txt → labMT count matrices.
  4. Run src/build_matrix.py to collapse the per-book matrices into the
     final timeseries pickle.
  5. Load the pickle and verify shape + metadata.

Everything happens in a temp directory; no tracked files are touched and the
working full-corpus pickles in data/gutenberg/ are not clobbered. Exits 0 on
success, non-zero on any failure.

Run:
    uv run python tests/smoke_e2e.py
"""

import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

N_BOOKS = 5

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = (REPO_ROOT / ".." / "core-stories-code" / "database" / "db.sqlite3").resolve()


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def query_sample_books(n):
    """Pick the n highest-downloaded books from the canonical filter."""
    if not DB_PATH.is_file():
        fail(
            f"SQLite DB not found at {DB_PATH}. Clone "
            f"https://github.com/andyreagan/core-stories-code next to this repo "
            f"and try again."
        )
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT gutenberg_id, title, downloads
        FROM library_book
        WHERE exclude=0
          AND length > 20000 AND length <= 100000
          AND downloads >= 40
          AND lang_code_id=0
          AND locc_with_P=1
        ORDER BY downloads DESC
        LIMIT ?
        """,
        (n,),
    )
    rows = cur.fetchall()
    conn.close()
    if len(rows) < n:
        fail(f"Only got {len(rows)} books from SQLite; expected {n}")
    return rows


def download_book(gid, dest):
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as e:
        fail(f"Could not download book {gid} from {url}: {e}")
    dest.write_bytes(data)
    return len(data)


def main():
    print(f"[setup] repo root: {REPO_ROOT}")
    print(f"[setup] SQLite:    {DB_PATH}")

    tmp = Path(tempfile.mkdtemp(prefix="cs-smoke-"))
    txt_dir = tmp / "gut-raw"
    csv_dir = tmp / "csvs"
    out_dir = tmp / "matrix-out"
    txt_dir.mkdir()
    csv_dir.mkdir()
    out_dir.mkdir()
    print(f"[setup] tmp dir:   {tmp}")

    try:
        # 1. Sample
        print(f"\n[1/5] sampling {N_BOOKS} books from canonical filter...")
        books = query_sample_books(N_BOOKS)
        for gid, title, dls in books:
            print(f"      {gid:>6}  ({dls} dl)  {title[:60]}")

        # 2. Download
        print(f"\n[2/5] downloading from gutenberg.org...")
        for gid, title, _ in books:
            n = download_book(gid, txt_dir / f"{gid}.txt")
            print(f"      {gid:>6}  {n:>9,} bytes")

        # 3. build_csvs.py
        print(f"\n[3/5] running build_csvs.py...")
        r = subprocess.run(
            [
                "uv", "run", "python",
                str(REPO_ROOT / "src" / "build_csvs.py"),
                "--input-dir", str(txt_dir),
                "--output-dir", str(csv_dir),
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            fail(f"build_csvs.py exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        print("      " + (r.stdout.strip().replace("\n", "\n      ") if r.stdout.strip() else "(no stdout)"))

        # 4. build_matrix.py (with env overrides so we don't clobber data/gutenberg)
        print(f"\n[4/5] running build_matrix.py with temp dir overrides...")
        env = os.environ.copy()
        env["CORE_STORIES_CSV_DIR"] = str(csv_dir)
        env["CORE_STORIES_OUT_DIR"] = str(out_dir)
        r = subprocess.run(
            ["uv", "run", "python", str(REPO_ROOT / "src" / "build_matrix.py")],
            cwd=REPO_ROOT, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            fail(f"build_matrix.py exit {r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        print("      " + r.stdout.strip().replace("\n", "\n      "))

        # 5. Validate
        print(f"\n[5/5] validating output pickle...")
        matrix_pkl = next(out_dir.glob("timeseries-matrix-cache-*.p"))
        books_pkl  = next(out_dir.glob("books-*.p"))
        M = pickle.load(open(matrix_pkl, "rb"))
        rec = pickle.load(open(books_pkl, "rb"))

        if M.shape != (N_BOOKS, 200):
            fail(f"matrix shape {M.shape} != ({N_BOOKS}, 200)")
        if len(rec) != N_BOOKS:
            fail(f"records len {len(rec)} != {N_BOOKS}")
        if not all(4.0 <= float(v) <= 7.0 for v in M.mean(axis=1)):
            fail(f"per-book mean valence out of plausible range: {M.mean(axis=1).tolist()}")

        print(f"      matrix shape:    {M.shape}")
        print(f"      books records:   {len(rec)}")
        print(f"      mean valence:    {M.mean():.4f}  (range {M.mean(0).min():.3f}–{M.mean(0).max():.3f})")
        print(f"      pickles:         {matrix_pkl.name}, {books_pkl.name}")

        print("\nPASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
