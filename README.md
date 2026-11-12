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
LayoutLMv3-B → assemble → categorize → checks), about 0.25–0.3 s per photo. The receipt in the video is a
synthetic photo (`docs/demo/sample-receipt.jpg`), so it shows the product, not the accuracy: the numbers are under
Results.

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
bash scripts/ocr_queue.sh     # the whole queue: test sets, train/val, Tesseract, preprocessing ablations

# 3. Train extractors -> models/
.venv/bin/python -s scripts/train_crf.py                                       # models/crf.pkl
.venv/bin/python -s scripts/train_transformer.py --model roberta --phase A     # models/roberta-A/
.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase A  # models/layoutlmv3-A/ (gold words)
.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase B  # models/layoutlmv3-B/ (Paddle words)
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
**LayoutLMv3-B**; `GET /health` reports it and each saved receipt records it.
`RECEIPT_EXTRACTOR=<name>` forces one; `RECEIPT_PIPELINE=stub` returns the fixed demo receipt.
A blurry or glary photo gets HTTP 422 with a "retake it" message. The raw OCR (words, boxes, matrix `M`, timings)
is stored in `receipts.raw_ocr`.

Measured on the synthetic `docs/demo/sample-receipt.jpg` (shared GPU, warm): Paddle OCR 0.16–0.19 s,
LayoutLMv3-B 0.02–0.04 s, whole request 0.25–0.31 s.

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
| rules | gold | 0.761 | 0.630 | 0.535 | 0.540 | 0.504 | 0.000 |
| crf | gold | 0.823 | 0.544 | 0.437 | 0.838 | 0.423 | 0.001 |
| roberta-A | gold | 0.877 | 0.472 | 0.531 | 0.874 | 0.494 | 0.024 |
| layoutlmv3-A | gold | **0.892** | 0.535 | 0.548 | **0.933** | **0.579** | 0.024 |
| layoutlmv3-B | gold | 0.882 | 0.609 | **0.725** | 0.923 | 0.546 | 0.029 |
| rules | paddle | 0.689 | **0.725** | 0.397 | 0.373 | **0.431** | 0.185 |
| crf | paddle | 0.616 | 0.477 | 0.370 | 0.375 | 0.168 | 0.186 |
| roberta-A | paddle | 0.690 | 0.528 | 0.442 | 0.449 | 0.294 | 0.195 |
| layoutlmv3-A | paddle | 0.708 | 0.505 | 0.482 | 0.418 | 0.337 | 0.211 |
| **layoutlmv3-B** | paddle | **0.776** | 0.619 | **0.607** | **0.634** | 0.414 | 0.206 |

What the table says:
- On clean words, layout helps: LayoutLMv3 (text + boxes + image) beats RoBERTa (text) beats CRF beats rules on
  WildReceipt fields and items.
- Real OCR costs a lot: every learned model drops 0.15–0.5 F1 from gold to Paddle words (split, merged and misread
  tokens it never saw in training).
- Training on real OCR output wins that back: LayoutLMv3-B (phase A weights fine-tuned on Paddle words, labels
  projected from gold) is the best live extractor, with WR field F1 0.776 vs 0.708 for phase A and item F1 0.634 vs 0.418.
- SROIE is out of domain for every learned model (they train on WildReceipt only); there the regex + anchor rules
  stay best (0.725 on Paddle words).
- Caveat: the item-assembly rule shared by all extractors was designed partly by looking at CORD **test**
  mismatches, so the CORD item column is somewhat optimistic for every row alike.

- **Categorizer** (`eval/categories.json`, no hand labels): items macro-F1 0.732 clean / 0.569 receipt-style
  (keyword baseline 0.431 / 0.229); merchants 0.621 / 0.535 (keywords 0.508 / 0.232), θ = 0.75 covers 63% at
  0.89 precision; 80% of CORD test receipts come out `dining`. Items on a confidently non-restaurant receipt are
  never `dining`.
- **Blur gate** (`blur.json`): 0% false rejects on sharp SROIE and WildReceipt train, 0.6% at blur σ=1, 76% at σ=2, 100% from σ=3.
- **Warp** (`warp.json`): a paper quad is found on 3 of 37 CORD test photos (IoU 0.92 when found); mean IoU 0.71 vs 0.68 without warping.
- **Deskew** (`deskew.json`): median angle error 0.10°, p90 0.36° on 361 SROIE test scans rotated 1–15°.
- **Row grouping** (`rows.json`): τ = 0.45; same-row pair F1 0.982 on CORD test gold boxes, 0.980 on Paddle boxes.
- **OCR** (`ocr.csv`, steps 2–5): Paddle detection H-mean 0.851 / 0.927 / 0.897 and CER 0.139 / 0.094 / 0.105 on
  WildReceipt / SROIE / CORD test, p50 0.12–0.31 s per photo on the GPU; Tesseract 0.599 / 0.718 / 0.414 H-mean.
- **Preprocessing ablation** (Paddle OCR on the three test sets with one step switched off; CER from `ocr.csv`,
  downstream field/item F1 from the CRF rows in `results.csv`):

  | steps | removed | CER WR / SROIE / CORD | CRF WR field F1 | CRF CORD item F1 | p50 s |
  |---|---|---|---|---|---|
  | 2345 (default) | — | 0.139 / 0.094 / 0.105 | 0.616 | 0.370 | 0.186 |
  | 345 | 2 rescale | 0.141 / 0.095 / 0.102 | 0.615 | 0.363 | 0.137 |
  | 245 | 3 warp | 0.139 / 0.094 / 0.103 | 0.615 | 0.370 | 0.134 |
  | 235 | 4 deskew | 0.139 / 0.094 / 0.103 | 0.617 | 0.376 | 0.130 |
  | 234 | 5 shadow removal | 0.140 / 0.094 / 0.128 | 0.621 | 0.388 | 0.135 |
  | none | all | 0.139 / 0.095 / 0.135 | 0.621 | 0.382 | 0.131 |

  PaddleOCR is robust on its own: no step moves the downstream F1 by more than about 0.02. Shadow removal (step 5) is
  the one step with a clear OCR gain, on CORD's phone photos (CER 0.135 → 0.105). Deskew gives a small detection gain
  on WildReceipt (H-mean 0.844 → 0.851 with it). Steps 2–3 are neutral. The live default stays 2–5,
  because LayoutLMv3-B was trained on that OCR output and the steps cost about 0.05 s per photo.
