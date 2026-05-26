# core-stories

Notebooks and analysis scripts for "[The emotional arcs of stories are dominated by six basic shapes](https://arxiv.org/abs/1606.07772)" — clustering of Project Gutenberg book sentiment timeseries.

> **Which repo is this?** This is the *artifact* repo: the reproducible pipeline that produces the paper's figures. The companion repo `andyreagan/core-stories-private` is the original 2015–2016 research scratch repo (paper LaTeX sources, exploratory notebooks, dev diary). It's private because it includes work on copyrighted texts; nothing in it is needed to run this pipeline. **Start here.**

## Quickstart

```bash
uv sync                                                            # creates .venv with pinned deps
uv run python src/build_csvs.py --input-dir <gutenberg-txt-dir>    # raw .txt(.gz) → per-book labMT matrices
uv run python src/build_matrix.py                                  # per-book matrices → (N x 200) timeseries matrix
make                                                               # runs SOM, hierarchical clustering, PCA/SVD
```

Outputs land in `output/figures/{SOM,clustering,SVD}/<version>/`. The only bootstrap input you need is a directory of Project Gutenberg `.txt` files — see below.

## Reproducing from raw data

The book-metadata table is bundled as `data/library_book.csv.gz` (51,250 rows, ~3 MB gzipped). Everything else needed to run is one input:

### `data/per-book-labmt-counts/` (per-book labMT word-count matrices)

For each Gutenberg book id you want included, the pipeline reads a file at
`data/per-book-labmt-counts/{gutenberg_id}.csv.gz` with this exact format:

- gzipped CSV
- shape **`(10222, 200)`** of integers
- **rows**: word counts in `labMTsimple/data/LabMT/labMT1.txt` order (10222 words; the header line is skipped). Row `i` is the count of word `labMT1[i+1]`.
- **columns**: 200 sliding-window positions across the book, each window 10000 tokens wide. Step size is `floor((n_tokens - 10000) / 199)`; the final window extends to the end of the text.
- one row per line, comma-separated.

#### Producing these CSVs from raw Gutenberg text

1. Download the books from Project Gutenberg ([gutenberg.org](https://www.gutenberg.org/) or any of its [mirrors](https://www.gutenberg.org/MIRRORS.ALL)). The pipeline expects plain-text UTF-8 (with ISO-8859-1 fallback). Name each file `{gutenberg_id}.txt` or `{gutenberg_id}.txt.gz` so the script can recover the book id from the filename.

   The canonical filter requires ≥40 downloads, 20K–100K words, English, locc `P` (literature) — about 1385 books in the snapshot used for the published paper. The exact id list lives in `data/library_book.csv.gz`; `build_matrix.py` applies the filter at load time.

2. Convert raw text → labMT count matrices:

   ```bash
   uv run python src/build_csvs.py --input-dir /path/to/gutenberg/txt --output-dir data/per-book-labmt-counts
   ```

   The script strips Project Gutenberg headers/footers via `*** START/END OF ... PROJECT GUTENBERG EBOOK ***` markers, tokenizes with `[A-Za-z][A-Za-z']*` (see "Tokenizer choice" below for the rationale), slides a 10000-token window 200 times, and lowercase-matches against `labMT1.txt` to produce the integer matrix described above. See the docstring in `src/build_csvs.py` for full algorithm details.

#### Tokenizer choice

The pipeline scores text by counting occurrences of each labMT vocabulary entry. **The labMT word list is the fixed side of the join — the tokenizer's job is to produce tokens shaped like labMT entries, not to mimic some reference tokenizer.** This framing matters because the labMT vocabulary was *intentionally* not aggressively tokenized: an entry like `mother's` is rated as a single affective unit (a possessive context that means something different from `mother` alone). Splitting it destroys the rater's intent.

Composition of the 10222-word labMT1 vocabulary:

| feature | count | examples |
|---|---|---|
| apostrophe-containing | 131 | `mother's`, `don't`, `I'm`, `they're`, `friend's` |
| hyphen-containing | 51 | `father-in-law`, `great-grandfather`, `e-mail`, `t-shirt` |
| digit-containing | 82 | `1980s`, `1st`, `mp3`, `80s` |
| pure alphabetic | ~9960 | typical English words |

Tokenizers tested (per-book Pearson r vs. cached spaCy-1.x CSVs on an 80-book stratified sample; mean valence drift is on the timeseries level):

| tokenizer | median r | r ≥ 0.90 | r ≥ 0.95 | min r | median \|drift\| |
|---|---|---|---|---|---|
| `\b\w+\b` (Python word regex) | 0.843 | 22.5% | 5.0% | −0.12 | 0.046 |
| **`[A-Za-z][A-Za-z']*`** (chosen) | **0.844** | **27.5%** | **8.8%** | **+0.07** | **0.007** |
| `[A-Za-z][A-Za-z'\-]*` (+ hyphens) | 0.838 | 28.7% | 7.5% | +0.10 | 0.005 |
| `[A-Za-z0-9][A-Za-z0-9'\-]*` (+ hyphens + digits) | 0.840 | 27.5% | 7.5% | +0.10 | 0.005 |
| NLTK `word_tokenize` (Penn Treebank) | 0.853 | 27.5% | 1.2% | −0.07 | 0.027 |
| NLTK `TreebankWordTokenizer` | 0.852 | 26.2% | 1.2% | −0.07 | 0.028 |
| spaCy 3.8 `English()` (whole text) | 0.841 | 22.5% | 3.8% | −0.17 | 0.042 |
| spaCy 3.8 `English()` + original paragraph chunking | 0.840 | 23.8% | 3.8% | −0.17 | 0.042 |

The "median r" column is *similar* across all tokenizers because the per-book correlation is dominated by the same population of typical English alphabetic words. The columns that move are:

- **Mean-valence drift**: the apostrophe-preserving regex is ~7× better than `\b\w+\b` or spaCy. This is the direct payoff of treating labMT's intent correctly — counts of frequent possessives like `mother's` now match labMT, instead of being split into `mother` + `s` (where `s` is junk and `mother` has a different valence).
- **Min r**: only the apostrophe-preserving regexes eliminate anticorrelating outlier books.
- **r ≥ 0.95**: the apostrophe regex roughly doubles the fraction of books that closely match the spaCy CSVs at the upper end (8.8% vs 5.0%); ironically, NLTK's Penn-Treebank-style splitting (which we'd expect to be the closest match to spaCy 1.x's habits) is the worst here.

**Why not also keep hyphens?** Tested — they hurt. Only 51 hyphen-words are in labMT, but the common case of `well-known` → joined token `well-known` (not in labMT) destroys *two* labMT matches (`well` and `known`) to capture nothing. Net negative.

**Why not modern spaCy?** Tokenizer rules have evolved meaningfully between spaCy 1.x (2016) and 3.x (2026). The 3.x English tokenizer does not closely match 1.x output, and inherits the same labMT-blind contraction-splitting behavior anyway.

#### A note on reproducibility

Per-book sentiment timeseries from this script will *not* be byte-identical to the values used in the original Reagan et al. 2016 paper, for two reasons:

1. **Project Gutenberg edits its texts.** Books get re-OCR'd, re-formatted, re-edited continuously. The `.txt` file you download today is generally not the same as the one downloaded in 2016. Per-book correlations vs. the 2016 snapshot range from ~0.55 to ~0.95 depending on how much the text has changed.
2. **Tokenizer.** This script uses an apostrophe-preserving regex (see above); the original used spaCy 1.x.

**What this means for the analyses** (tested empirically — regex pipeline vs. cached spaCy-1.x CSVs on the full 1385-book corpus):

- **PCA / SVD modes (the paper's "basic shapes" as eigen-arcs):** robust. The top 6 right singular vectors correlate ≥ 0.96 between the two pipelines. If you only care about identifying the dominant arc shapes, this pipeline reproduces them faithfully.
- **Hierarchical clustering at k=6 (the paper's "six basic shapes" cluster assignment):** *not* fully robust. Adjusted Rand Index between cluster labels is ≈ 0.35. Four of six clusters match centroids at r = 0.90–0.96 — the four non-flat arc shapes survive — but the original pipeline's specific way of splitting low-variance "flat" books into two sub-clusters does not replicate.

So if you want the **eigen-shapes**, this pipeline is fine. If you want to reproduce the exact published **cluster membership** at k=6, you need the original spaCy 1.x runtime (and the 2016 Gutenberg snapshot). The k=6 number itself appears to be partly an artifact of how spaCy 1.x splits the flat-book mass.

## How `src/build_matrix.py` uses it

1. Reads `data/library_book.csv.gz` and applies the filter (default: `P-20K-100K-40dl-200pt` = locc `P` prefix, 20K–100K words, ≥40 downloads, English, 200 windows).
2. For each match, reads the `(10222, 200)` matrix from `data/per-book-labmt-counts/{id}.csv.gz`.
3. Applies the labMT stopper: `mask = |score - 5| >= 1.0` (drops neutral words; 3731 of 10222 retained at default threshold).
4. Per column, computes `valence = scores[mask] @ counts[mask, :] / counts[mask, :].sum(axis=0)`.
5. Stacks into an `(N, 200)` matrix and pickles to `data/gutenberg/timeseries-matrix-cache-<version>.p` plus a parallel `books-<version>.p` metadata list.

With the full corpus this yields **N = 1385** books at the default filter.

## Running the analyses

```bash
make                  # all three
# or individually:
uv run python src/SOM/SOM-002.py 40 false
uv run python src/hierarchical_clustering/hierarchical-clusting-004.py 40 false
uv run python src/PCA_SVD/PCA-SVD-006.py 40 false
```

Args: `<min_downloads> <salad_flag> [exclude_pattern]`. The Makefile's `control` target sweeps exclude patterns `-1` through `-10`.

The first run also produces three pairwise-distance caches under `data/gutenberg/` (≈seconds via `scipy.spatial.distance.cdist`); subsequent runs reuse them.

## Smoke test

```bash
uv run python tests/smoke_e2e.py
```

Downloads 5 books from gutenberg.org, runs them through `build_csvs.py` + `build_matrix.py` in a temp dir (so it doesn't clobber any working full-corpus pickles), validates the output shape and mean-valence range, exits 0 on success. Takes ~30 seconds.

## Layout

- `src/build_csvs.py` — converts raw Gutenberg `.txt` files → per-book `(10222, 200)` labMT count CSVs
- `src/build_matrix.py` — collapses the per-book CSVs into the final `(N, 200)` timeseries matrix pickle
- `src/bookclass_standalone.py` — pickle loader (replaces the old Django-backed `bookclass.py`)
- `src/{SOM,hierarchical_clustering,PCA_SVD}/` — the three analyses
- `src/kitchentable/` — vendored plotting helpers
- `pyproject.toml` / `uv.lock` — pinned environment

## Notes

- The original code targeted Python 2 + Django + spacy. This version is Python ≥3.11, no Django, no spacy — the per-book word counts are already pre-computed in the CSVs, so re-tokenization isn't needed.
- `pdftile.pl` (used by SOM at the end to combine PDFs) is missing; the `subprocess.call` line fails silently and the individual PDFs are otherwise complete.
- Questions, bugs, or suggestions: please [open an issue](https://github.com/andyreagan/core-stories/issues) on this repo.
