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

> The video was recorded with the stub pipeline. The API now runs the real pipeline
> (`receipt/pipeline.py`: quality gate → preprocess → Paddle OCR → extractor → assemble → categorize → checks);
> `RECEIPT_PIPELINE=stub` still returns the fixed demo receipt.

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
from `eval/results.csv`: among extractors whose model exists, the best mean of WildReceipt and SROIE field F1
(Paddle-OCR rows preferred over gold-OCR rows); `GET /health` reports it and each saved receipt records it.
`RECEIPT_EXTRACTOR=<name>` forces one; `RECEIPT_PIPELINE=stub` returns the fixed demo receipt.
A blurry or glary photo gets HTTP 422 with a "retake it" message. The raw OCR (words, boxes, matrix `M`, timings)
is stored in `receipts.raw_ocr`.

Measured on the synthetic `docs/demo/sample-receipt.jpg` (shared GPU, warm): Paddle OCR 0.13–0.20 s,
extractor 0.001 s (rules) / 0.03 s (roberta-A), whole request 0.3 s.

Tests: `.venv-ocr/bin/python -s -m pytest -q` (uses a throwaway `rec_test` database; `tests/test_pipeline.py` runs the
real pipeline on the GPU). Re-record the demo (empty DB, both servers up):
`.venv/bin/python -m playwright install chromium && .venv/bin/python scripts/record_demo.py`.

## Results

`eval/results.csv` — same scorer, same frozen test sets for every row (field F1 = micro F1 over header fields;
exact = merchant + date + total all right; `p50_s` = median OCR + extractor seconds per receipt).
LayoutLMv3 A/B and Paddle-OCR rows are added when training and the OCR queue finish.

| extractor | ocr | steps | WR field F1 | SROIE field F1 | CORD item F1 | WR item F1 | exact | p50 s |
|---|---|---|---|---|---|---|---|---|
| crf | gold | none | 0.8226 | 0.5435 | 0.4372 | 0.8379 | 0.4234 | 0.001 |
| rules | gold | none | 0.761 | 0.6295 | 0.5349 | 0.5398 | 0.5038 | 0.0003 |
| roberta-A | gold | none | 0.8765 | 0.4718 | 0.5309 | 0.8735 | 0.4941 | 0.0238 |

- **Categorizer** (`eval/categories.json`, no hand labels): items macro-F1 0.732 clean / 0.569 receipt-style
  (keyword baseline 0.431 / 0.229); merchants 0.621 / 0.535 (keywords 0.508 / 0.232), θ = 0.75 covers 63% at
  0.89 precision; 80% of CORD test receipts come out `dining`.
- **Blur gate** (`blur.json`): 0% false rejects on sharp SROIE and WildReceipt train, 0.6% at blur σ=1, 76% at σ=2, 100% from σ=3.
- **Warp** (`warp.json`): a paper quad is found on 3 of 37 CORD test photos (IoU 0.92 when found); mean IoU 0.71 vs 0.68 without warping.
- **Deskew** (`deskew.json`): median angle error 0.10°, p90 0.36° on 361 SROIE test scans rotated 1–15°.
- **Row grouping** (`rows.json`): τ = 0.45; same-row pair F1 0.982 on CORD test gold boxes, 0.980 on Paddle boxes.
- **OCR** (`ocr.csv`, Paddle, steps 2–5): detection H-mean 0.851 / 0.906 / 0.897 and CER 0.139 / 0.132 / 0.105 on
  WildReceipt / SROIE / CORD test, p50 0.12–0.30 s per photo.
