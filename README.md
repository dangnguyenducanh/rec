# Receipt Reader

Receipt photo → OpenCV preprocessing → PaddleOCR → token-classification extractor → categories → Postgres → React (Vercel).

- Design spec: [docs/specs/2026-11-12-receipt-pipeline-design.md](docs/specs/2026-11-12-receipt-pipeline-design.md)
- Frozen file formats and function signatures: [docs/specs/2026-11-12-interfaces.md](docs/specs/2026-11-12-interfaces.md)

## Demo

[▶ Watch the 45-second walkthrough](docs/demo/receipt-reader-demo.mp4): upload a photo → read it → fix a typo the
checks catch → saved list → spending by category.

| Editor | Spending |
|---|---|
| ![Editor](docs/demo/editor.png) | ![Spending](docs/demo/spending.png) |

Recorded against the real pipeline on the GPU (`receipt/pipeline.py`: quality gate → preprocess → Paddle OCR →
LayoutLMv3 → assemble → categorize → checks), about 0.25–0.3 s per photo, with the earlier LayoutLMv3-B; the API now
serves LayoutLMv3-B4 (below). The receipt in the video is a synthetic photo (`docs/demo/sample-receipt.jpg`), so it
shows the product, not the accuracy: the numbers are under Results.

## Environments

Two virtualenvs, always run with `-s` (ignores `~/.local`). Run everything from the repo root.

| Env | Holds | Used for |
|---|---|---|
| `.venv` | Python 3.12, torch (cu126), transformers, paddleocr (CPU), sklearn-crfsuite, opencv | data prep, training, benchmark, diagnostics |
| `.venv-ocr` | paddlepaddle-gpu 3.3.1 + torch 2.7.1+cu126 (same cuDNN 9.5; NCCL pinned by `.tools/gpu-overrides.txt`) + every API / pipeline / test package | GPU OCR caches, **serving the API**, the test suite |

They are separate because paddlepaddle-gpu would downgrade the cuDNN/NCCL that `.venv`'s torch needs.
`receipt/ocr.py` uses the GPU whenever the running Paddle has CUDA. Tesseract lives in `.tools/tess/bin/tesseract`.
Postgres: `docker compose -p rec up -d` (port 18332).

## Pipeline, step by step

```bash
# 1. Gold data: download + convert WildReceipt / CORD / SROIE -> data/gold/{dataset}/{split}.jsonl
.venv/bin/python -s scripts/prepare_data.py

# 2. OCR caches -> data/ocr_cache/{engine}-s{steps}/{dataset}/{split}.jsonl (re-execs in .venv-ocr for Paddle GPU)
.venv/bin/python -s scripts/build_ocr_cache.py --engine paddle --steps 2,3,4,5 --datasets wildreceipt,sroie,cord --splits test
.venv/bin/python -s scripts/build_ocr_cache.py --engine paddle --datasets wildreceipt --splits train,val   # LayoutLMv3 phase B input
.venv/bin/python -s scripts/build_ocr_cache.py --engine paddle --datasets cord,sroie --splits train,val      # B4's extra training data
bash scripts/ocr_queue.sh     # the whole queue: test sets, train/val, Tesseract, preprocessing ablations

# 3. Train extractors -> models/
.venv/bin/python -s scripts/train_crf.py                                       # models/crf.pkl
.venv/bin/python -s scripts/train_transformer.py --model roberta --phase A     # models/roberta-A/
.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase A  # models/layoutlmv3-A/ (gold words)
.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase B  # models/layoutlmv3-B/ (Paddle words)
.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase B --datasets wildreceipt,cord,sroie \
    --out models/layoutlmv3-B4                                                 # the served model (see Improving phase B)
bash scripts/train_queue.sh   # the same, one GPU job at a time, logs in models/logs/

# 4. Benchmark: one row per (extractor, ocr, steps) -> eval/results.csv, details -> eval/runs/
.venv/bin/python -s scripts/benchmark.py --extractor crf --ocr gold
.venv/bin/python -s scripts/benchmark.py --extractor layoutlmv3-B --ocr paddle --steps 2,3,4,5

# 5. Categorizer: public labelled data (Instacart, CORD, OSM) -> data/categories/, then train -> models/categorizer/
.venv/bin/python -s scripts/build_category_data.py
.venv/bin/python -s scripts/train_categorizer.py                               # scores -> eval/categories.json

# Stage 1-3 diagnostics -> eval/diagnostics/
.venv/bin/python -s scripts/diagnostics.py blur        # also: warp, deskew, rows --ocr paddle, ocr --engine paddle
```

## Serving

```bash
docker compose -p rec up -d                                          # Postgres on :18332
CORS_ORIGINS=https://<your-app>.vercel.app scripts/serve.sh          # API on 127.0.0.1:18300, logs in data/serve.log
cd frontend && npm install && npm run dev                            # UI on :18373 (VITE_API_URL for Vercel)
```

`scripts/serve.sh` runs uvicorn from `.venv-ocr` (default `CORS_ORIGINS=http://localhost:18373`). At startup the API
loads Paddle OCR and the extractor on the GPU and warms them, so the first upload is fast. The extractor is chosen
from `eval/results.csv`: among extractors whose model exists, the best mean of the four F1 columns (header fields
and line items, since items drive the categories), Paddle-OCR rows preferred over gold-OCR rows. Today that is
**LayoutLMv3-B4**; `GET /health` reports it and each saved receipt records it.
`RECEIPT_EXTRACTOR=<name>` forces one; `RECEIPT_PIPELINE=stub` returns the fixed demo receipt.
A blurry or glary photo gets HTTP 422 with a "retake it" message. The raw OCR (words, boxes, matrix `M`, timings)
is stored in `receipts.raw_ocr`.

Measured on the synthetic `docs/demo/sample-receipt.jpg` (warm): with LayoutLMv3-B4, 0.24 s per request on a free GPU
(Paddle OCR 0.22 s, extractor 0.02 s) and up to 0.8 s while other jobs load the card. B4 reads every field and all
five items of that sample right; the address loses its last line ("CA 95814").

Tests: `.venv-ocr/bin/python -s -m pytest -q` (uses a throwaway `rec_test` database; `tests/test_pipeline.py` runs the
real pipeline on the GPU). Re-record the demo (empty DB, both servers up):
`.venv/bin/python -m playwright install chromium && .venv/bin/python scripts/record_demo.py`.

## Results

`eval/results.csv`: same scorer, same frozen test sets for every row. Field F1 = micro F1 over the header fields a
dataset labels; item F1 = name, qty and price all right; exact = merchant + date + total all right; `p50_s` = median
OCR + extractor seconds per receipt. **gold** = the datasets' own words and boxes (upper bound); **paddle** = what
runs live, PaddleOCR on the GPU after preprocessing steps 2–5.

| extractor | ocr | WR field F1 | SROIE field F1 | CORD item F1 | WR item F1 | exact | p50 s |
|---|---|---|---|---|---|---|---|
| rules | gold | 0.765 | **0.630** | 0.543 | 0.562 | 0.507 | 0.001 |
| crf | gold | 0.824 | 0.558 | 0.461 | 0.846 | 0.430 | 0.005 |
| roberta-A | gold | 0.880 | 0.480 | 0.581 | 0.882 | 0.501 | 0.078 |
| layoutlmv3-A | gold | **0.895** | 0.546 | 0.587 | **0.932** | **0.584** | 0.144 |
| layoutlmv3-B | gold | 0.884 | 0.615 | 0.803 | 0.919 | 0.552 | 0.138 |
| layoutlmv3-B3 | gold | 0.874 | 0.594 | **0.937** | 0.900 | 0.539 | 0.182 |
| layoutlmv3-B4 | gold | 0.872 | 0.585 | 0.912 | 0.904 | 0.567 | 0.238 |
| rules | paddle | 0.704 | 0.726 | 0.410 | 0.392 | 0.447 | 0.189 |
| crf | paddle | 0.621 | 0.495 | 0.400 | 0.403 | 0.169 | 0.201 |
| roberta-A | paddle | 0.701 | 0.535 | 0.477 | 0.476 | 0.309 | 0.275 |
| layoutlmv3-A | paddle | 0.720 | 0.539 | 0.501 | 0.434 | 0.362 | 0.473 |
| layoutlmv3-B | paddle | 0.786 | 0.650 | 0.653 | **0.657** | 0.439 | 0.295 |
| layoutlmv3-B3 | paddle | 0.794 | 0.795 | **0.781** | 0.652 | 0.582 | 0.573 |
| **layoutlmv3-B4** | paddle | **0.798** | **0.829** | 0.750 | 0.656 | **0.612** | 0.351 |

Every row was re-run on 2026-11-13 with the current assembly code (`receipt/assemble.py`, `receipt/rows.py`). The
p50 of the transformer rows is inflated: those runs shared the GPU with LayoutLMv3 training (the same LayoutLMv3-B row
took 0.206 s on an idle card); the live API answers in 0.24 s with B4.

What the table says:
- On clean words, layout helps: LayoutLMv3 (text + boxes + image) beats RoBERTa (text) beats CRF beats rules on
  WildReceipt fields and items.
- Real OCR costs a lot: every learned model drops 0.15–0.5 F1 from gold to Paddle words (split, merged and misread
  tokens it never saw in training).
- Training on real OCR output wins that back: LayoutLMv3-B (phase A weights fine-tuned on WildReceipt Paddle words,
  labels projected from gold) gets WR field F1 0.786 vs 0.720 for phase A and item F1 0.657 vs 0.434.
- More training data, with labels that are right, wins more: LayoutLMv3-B4 adds CORD and SROIE train (SROIE through
  weak labels, below). SROIE field F1 0.650 → 0.829 (rules, the best on SROIE until now, get 0.726), CORD item F1
  0.653 → 0.750, exact 0.439 → 0.612, WR fields 0.786 → 0.798; WR items stay level (0.657 → 0.656).
- SROIE is no longer out of domain for B3 and B4 (they train on SROIE train); for every other learned model it
  still is (they train on WildReceipt only).
- B4 was picked over B3 on val (4-F1 mean 0.769 vs 0.756) before either was scored on test. Test agrees on the mean
  (0.758 vs 0.756), but B3 is better on CORD items (0.781 vs 0.750) and B4 on SROIE (0.829 vs 0.795).
- Caveat: the item-assembly rule shared by all extractors was designed partly by looking at CORD **test**
  mismatches, so the CORD item column is somewhat optimistic for every row alike. The 2026-11-13 assembly fixes
  below were chosen on val only, and every test row went up with them.

### Improving phase B: B → B4

1. **More data.** GPU OCR for CORD train/val and SROIE train (`build_ocr_cache.py --datasets cord,sroie --splits
   train,val`). CORD's word labels are projected onto Paddle words like WildReceipt's. SROIE only gives
   merchant / address / date / total per receipt, so its words get **weak labels**: the best fuzzy span for merchant
   and address, the shortest 1–3-word span that parses to the gold date, and the words worth the gold total on a
   TOTAL / AMOUNT row (not SUB-TOTAL, not a payment row). Every 10th SROIE train receipt is held out as SROIE val.
2. **Early stopping on what serving uses**: the mean of four val F1s (WildReceipt field + item, SROIE field, CORD
   item), instead of WildReceipt field F1 alone.
3. **Error analysis with a multi-agent workflow** over B's val errors (every error put down to OCR, training labels,
   model, assembly or gold), a review of the training change, and each proposed fix implemented in its own git
   worktree and kept only if it moved the val mean by ≥ 0.003 without hurting rules or CRF. It found four bugs in
   the first SROIE weak labels (the run trained on them, B2, scored 0.754 on val but found items on 1 of 63 SROIE
   val receipts, against 59 for B): dates took the two words before them; every unannotated SROIE word (items,
   times, tax) was trained as O; SROIE gold-word copies carry text cut at the first comma; SUB-TOTAL and payment
   rows passed as TOTAL rows. In the fix, SROIE words are O in the header block (down to the last merchant/address
   row) and on the fields' own rows, times never, and every other SROIE word is left out of the loss.
4. **Label projection**: a Paddle word whose centre sits in several loose gold boxes takes the one it overlaps best
   (IoU); one whose centre just misses a box (≤ 0.1 word height) still takes its label. Assembling the training
   labels themselves: WildReceipt train items 0.692 → 0.751, CORD train items 0.779 → 0.852.
5. **Assembly**, for every extractor (old B on val 0.668 → 0.712 with no retraining; rules 0.550 → 0.558,
   CRF 0.469 → 0.487): rows never stack two words that overlap in x, and judge overlap against the nearest member;
   total = the last TOTAL run, skipping one equal to a TAX amount and preferring 2-decimal amounts; address = the
   first block only (no bill-to); qty "1.00" → "1" and a count written into the name ("(2x)", "2 @6.70") moves to
   qty; a name-less qty/price row pairs with the item next to it; items with no price are dropped (modifier lines).
6. **Runs** (val 4-F1 mean at the best epoch): B 0.712 · B2 0.754 (buggy SROIE labels, discarded) · B3 0.756 (fixed
   labels, every other SROIE word ignored, which cost SROIE fields) · **B4 0.769** (B3 + SROIE header rows kept as
   O, times never O). Runs with WildReceipt/CORD gold-word copies added (B2g, B3g) trailed at equal epochs and were
   stopped. Each run takes 25–40 min on the GPU.

Known limits of B4: on SROIE receipts it under-tags SUBTOTAL (12 words on SROIE val, against 61 for B), because with
0% GST "Total excl. GST" equals the total and the weak labels call it TOTAL; SROIE does not score subtotals.


- **Categorizer** (`eval/categories.json`, no hand labels): items macro-F1 0.732 clean / 0.569 receipt-style
  (keyword baseline 0.431 / 0.229); merchants 0.621 / 0.535 (keywords 0.508 / 0.232), θ = 0.75 covers 63% at
  0.89 precision; 80% of CORD test receipts come out `dining`. Items on a confidently non-restaurant receipt are
  never `dining`.
- **Blur gate** (`blur.json`): 0% false rejects on sharp SROIE and WildReceipt train, 0.6% at blur σ=1, 76% at σ=2, 100% from σ=3.
- **Warp** (`warp.json`): a paper quad is found on 3 of 37 CORD test photos (IoU 0.92 when found); mean IoU 0.71 vs 0.68 without warping.
- **Deskew** (`deskew.json`): median angle error 0.10°, p90 0.36° on 361 SROIE test scans rotated 1–15°.
- **Row grouping** (`rows.json`): words that overlap in x never share a row, and vertical overlap is judged against the
  nearest member in x; same-row pair F1 0.992 on CORD test gold boxes, 0.988 on Paddle boxes (0.982 / 0.980 with the
  earlier mean-band rule). Val now peaks at τ = 0.40 (0.994 vs 0.990 at 0.45); τ stays 0.45, what the models trained with.
- **OCR** (`ocr.csv`, steps 2–5): Paddle detection H-mean 0.851 / 0.927 / 0.897 and CER 0.139 / 0.094 / 0.105 on
  WildReceipt / SROIE / CORD test, p50 0.12–0.31 s per photo on the GPU; Tesseract 0.599 / 0.718 / 0.414 H-mean.
- **Preprocessing ablation** (Paddle OCR on the three test sets with one step switched off; CER from `ocr.csv`,
  downstream field/item F1 from the CRF rows in `results.csv`):

  | steps | removed | CER WR / SROIE / CORD | CRF WR field F1 | CRF CORD item F1 | p50 s |
  |---|---|---|---|---|---|
  | 2345 (default) | — | 0.139 / 0.094 / 0.105 | 0.621 | 0.400 | 0.186 |
  | 345 | 2 rescale | 0.141 / 0.095 / 0.102 | 0.618 | 0.387 | 0.137 |
  | 245 | 3 warp | 0.139 / 0.094 / 0.103 | 0.620 | 0.401 | 0.134 |
  | 235 | 4 deskew | 0.139 / 0.094 / 0.103 | 0.623 | 0.410 | 0.130 |
  | 234 | 5 shadow removal | 0.140 / 0.094 / 0.128 | 0.624 | 0.409 | 0.135 |
  | none | all | 0.139 / 0.095 / 0.135 | 0.622 | 0.401 | 0.131 |

  (CRF F1 re-run with the current assembly; p50 from the first run, on an idle GPU.)

  PaddleOCR is robust on its own: no step moves the downstream F1 by more than about 0.02. Shadow removal (step 5) is
  the one step with a clear OCR gain, on CORD's phone photos (CER 0.135 → 0.105). Deskew gives a small detection gain
  on WildReceipt (H-mean 0.844 → 0.851 with it). Steps 2–3 are neutral. The live default stays 2–5,
  because LayoutLMv3-B and B4 were trained on that OCR output and the steps cost about 0.05 s per photo.
