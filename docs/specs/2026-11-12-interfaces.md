# Receipt Reader — Implementation Interfaces

Companion to [the design spec](2026-11-12-receipt-pipeline-design.md). The spec says *what*; this
file freezes the file formats and function signatures every module shares, so modules built in
parallel fit together. Change a signature here first, then in code.

## 0. Ground rules

- Python env: `/home/aiface/dangnguyenducanh/rec/.venv` (Python 3.12, torch cu126, transformers,
  paddleocr CPU, sklearn-crfsuite, opencv). Always run it as `.venv/bin/python -s` (`-s` ignores
  `~/.local`, whose broken torch otherwise shadows the venv). Tesseract binary:
  `.tools/tess/bin/tesseract`.
- GPU OCR: `.venv-ocr` holds `paddlepaddle-gpu` (it can't share `.venv`: it would downgrade the
  cuDNN/NCCL that torch needs). `scripts/build_ocr_cache.py --engine paddle` re-execs itself there
  automatically and starts as many GPU workers as free VRAM holds (~2.6 GB each, 3 GB left spare); ~1.4 s/photo on GPU vs ~14 s on CPU.
  `receipt/ocr.py` picks the GPU whenever the running Paddle has CUDA, else the CPU.
- **Serving env = `.venv-ocr`**: paddlepaddle-gpu 3.3.1 + torch 2.7.1+cu126 (same cuDNN 9.5; NCCL pinned
  to 2.26.2 via `.tools/gpu-overrides.txt`, multi-GPU only) + every API/pipeline package. Measured per
  receipt: OCR 0.25–0.33 s + LayoutLMv3 0.03 s on GPU (vs 2.9 s on CPU). Run the API and the pipeline
  tests from `.venv-ocr`; `.venv` (torch 2.14) stays the training env and still works on CPU.
- Run everything from the repo root; paths in data files are relative to it.
- **One coordinate frame:** every box everywhere (gold, OCR, model input) is in the
  **original image's pixel coordinates**. Preprocessing returns a 3×3 matrix `M`
  (original → processed); `ocr.py` maps quads back with `M⁻¹`. The model sees the original image.
  (Builder's call: equivalent to the spec's "transform gold boxes" and keeps one frame.)
- A **quad** is 8 numbers `[x1,y1,x2,y2,x3,y3,x4,y4]`, clockwise from top-left.
- A **word** is the smallest text unit with a box: a WildReceipt annotation segment, a CORD word, a
  SROIE line box, or one whitespace-split token of a PaddleOCR line (box interpolated along the line
  by character offset).
- GPU: shared card that always reads ~100% busy from other users; ignore utilization and size every job
  to the **free VRAM at launch minus ~3 GB**. Jobs may run side by side when free VRAM allows. CPU: 24 cores.
- Never `git commit`; never edit files another lane owns; don't install packages (report missing ones).

## 1. Gold data — `data/gold/{dataset}/{split}.jsonl`

`dataset ∈ {wildreceipt, cord, sroie}`, `split ∈ {train, val, test}` (SROIE: train, test).
Raw downloads live in `data/raw/{dataset}/`. One JSON object per line:

```json
{
  "id": "wildreceipt/test/0001",
  "dataset": "wildreceipt", "split": "test",
  "image": "data/raw/wildreceipt/image_files/Image_16/11/receipt_0001.jpeg",
  "width": 1200, "height": 1600,
  "words": [{"text": "TOTAL", "quad": [..8..], "label": "O", "row": null, "group": null}],
  "roi": null,
  "gold": null,
  "fields": ["merchant", "address", "date", "time", "subtotal", "tax", "tips", "total"],
  "items_labeled": true,
  "dayfirst": false
}
```

- `label`: one of `receipt.schema.LABELS`, or `null` where the dataset has no token labels (SROIE).
- `row` / `group`: CORD `row_id` / `group_id`, else `null`. `roi`: CORD receipt quad or `null`.
- `gold`: a Receipt dict for datasets with document-level truth (CORD from `gt_parse`, SROIE from
  `entities`); `null` for WildReceipt, whose gold is `assemble.build(words, labels)` (spec §10).
- `fields`: header fields the dataset labels — the scorer masks the rest. WildReceipt: all 8;
  SROIE: `merchant, address, date, total`; CORD: `subtotal, tax, total`.
- `dayfirst`: locale hint for date parsing (SROIE `true`, others `false`), used for gold and pred alike.
- WildReceipt val = a deterministic 10% of train by receipt (sorted file names, every 10th).

Label maps: WildReceipt `*_value` classes → `MERCHANT ADDRESS DATE TIME ITEM_NAME ITEM_QTY
ITEM_PRICE SUBTOTAL TAX TIPS TOTAL` (Store_name, Store_addr, Date, Time, Prod_item, Prod_quantity,
Prod_price, Subtotal, Tax, Tips, Total); every `*_key`, `Tel_*`, `Others`, `Ignore` → `O`.
CORD `menu.nm`→ITEM_NAME, `menu.cnt`→ITEM_QTY, `menu.price`→ITEM_PRICE,
`sub_total.subtotal_price`→SUBTOTAL, `sub_total.tax_price`→TAX, `total.total_price`→TOTAL, rest O.

## 2. Shared helpers

`receipt/assemble.py`
```python
def norm_money(s: str | None) -> str | None      # "1,234.5"->"1234.50"; "60.000" (3-digit group)->"60000.00"; junk->None
def norm_date(s: str | None, dayfirst: bool = False) -> str | None   # -> "YYYY-MM-DD"
def norm_time(s: str | None) -> str | None       # -> "HH:MM" (24 h)
def norm_text(s: str | None) -> str | None       # uppercase, ALL whitespace removed, edge punctuation stripped
def build(words: list[dict], labels: list[str], dayfirst: bool = False) -> Receipt
```
`build` groups rows with `rows.group`, merges contiguous same-label words in reading order, joins
all ADDRESS spans, takes the first parseable DATE/TIME, the last parseable SUBTOTAL/TAX/TIPS/TOTAL,
and makes one Item per row holding ITEM_NAME words (qty/price from the same row, else the next
row). Categories stay `"other"` (stage 6 fills them).

`receipt/rows.py`
```python
def group(quads: list[list[float]], tau: float = TAU) -> list[int]   # row id per word, rows top→bottom
def order(quads: list[list[float]], row_ids: list[int]) -> list[int]  # reading order: row, then x
```
Skew is estimated from the words' own top edges (median angle) and undone before grouping, so it
works on gold boxes and OCR boxes alike without the preprocessing matrix.

## 3. OCR — `receipt/preprocess.py`, `receipt/ocr.py`

```python
# preprocess.py
STEPS_DEFAULT = (2, 3, 4, 5)
def quality_gate(img_bgr) -> tuple[bool, str | None]       # (ok, reason); step 1
def run(img_bgr, steps=STEPS_DEFAULT) -> tuple[np.ndarray, np.ndarray]   # (processed image, 3x3 M original->processed)
# ocr.py
def ocr(img_bgr, engine="paddle", steps=STEPS_DEFAULT) -> list[dict]     # [{"text","quad","conf"}], original coords
def cache_path(engine: str, steps, dataset: str, split: str) -> Path
def load_cache(engine, steps, dataset, split) -> dict[str, dict]         # id -> record
```
Cache: `data/ocr_cache/{engine}-s{steps}/{dataset}/{split}.jsonl`, `steps` written as digits
(`2345`, or `none`). Record: `{"id", "words": [{"text","quad","conf"}], "ocr_s": float, "M": [[..3..]x3]}`.
A cache file is written as `.jsonl.tmp` and renamed when complete: **if `{split}.jsonl` exists, it is complete.**

## 4. Extractors — `receipt/extractors/`

```python
# receipt/extractors/__init__.py
def get(name: str, model_dir: str = "models") -> Callable[[dict], list[str]]
```
`name ∈ {rules, crf, roberta, layoutlmv3}` (a model name may carry a phase: `layoutlmv3-B`).
The callable takes a **doc** `{"words": [{"text","quad"}], "width", "height", "image": path | PIL.Image | None}`
and returns one label per word. Model files: `models/crf.pkl`, `models/{roberta|layoutlmv3}-{A|B}/`
(HF `save_pretrained`, label list in `config.id2label`).

## 5. Categorizer — `receipt/categorize.py`

```python
def apply(r: Receipt) -> Receipt        # fills item categories and the receipt category
```
Models: `models/categorizer/{items,merchants}.joblib`. With no model files it falls back to the
keyword dictionary. Training data: `data/categories/{items,merchants}.csv` (`name,category,source,split`).

## 6. Scorer and benchmark

`receipt/score.py`
```python
HEADER = ["merchant", "address", "date", "time", "subtotal", "tax", "tips", "total"]
def gold_receipt(rec: dict) -> Receipt
def score(rows: list[tuple[Receipt, Receipt, dict]]) -> dict   # (pred, gold, gold record)
# -> {"n", "field_p", "field_r", "field_f1", "item_p", "item_r", "item_f1", "exact"}
```
Values are compared after the §2 normalizers (text: `norm_text`). Field counts per labelled field:
TP = both set and equal; FP = pred set and not equal; FN = gold set and not equal. Items match as a
multiset of `(norm_text(name), norm_money(price), qty or "1")`. Exact = every labelled one of
`merchant, date, total` correct. Unlabelled fields and `items_labeled: false` are skipped.

`scripts/benchmark.py --extractor NAME --ocr {gold,paddle,tesseract} [--steps 2,3,4,5] [--limit N]`
runs on WildReceipt, SROIE and CORD **test**, appends one row to `eval/results.csv`
(`extractor,ocr,steps,wr_field_f1,sroie_field_f1,cord_item_f1,wr_item_f1,exact,p50_s`) and writes
details to `eval/runs/{extractor}_{ocr}_s{steps}.json`. `--ocr gold` feeds gold words with labels
removed. `p50_s` = median of OCR time from the cache + extractor time.

## 7. Pipeline (API)

`receipt/pipeline.py`
```python
class Rejected(Exception): ...           # quality gate failed; message is shown to the user
def extract(image_bytes: bytes) -> tuple[Receipt, dict]   # (receipt, raw_ocr) ; raises Rejected
EXTRACTOR: str                            # name of the extractor in use
```
Uses the best available extractor: `layoutlmv3-B` → `layoutlmv3-A` → `crf` → `rules`.
`RECEIPT_PIPELINE=stub` keeps the fixture receipt (API tests use it).
