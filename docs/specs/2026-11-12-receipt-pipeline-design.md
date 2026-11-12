# Receipt Reader — Design Spec

Date: 2026-11-12 · Status: draft for review

## 1. Intent

A portfolio project that turns a photo of an **English receipt** into structured, categorized data:
merchant, date, totals, and line items, each item tagged with a spending category, plus one
category for the whole receipt. The point is to show the full classic document-AI stack —
OpenCV preprocessing → OCR → layout-aware token classification → rules → storage → web app —
with **one honest benchmark** that every component is judged by.

### Goals
- End-to-end: upload image → saved, categorized receipt visible in a web UI deployed on **Vercel**.
- A progression of extractors (rules → CRF → text-only transformer → **LayoutLMv3**) scored on the
  same test sets with the same scorer, on both gold OCR and real OCR.
- Item-level **and** receipt-level categories.
- Parallel build: web app and model training happen at the same time (§8).

### Non-goals
- No LLM / VLM anywhere in the pipeline.
- No login / multi-user (single shared receipt list).
- No languages other than English. No handwritten receipts.
- No job queue, no autoscaling: synchronous processing on one GPU box.

### Anti-goals (fails even if it "works")
- Reporting numbers from different scorers or different test sets side by side.
- Leaking test receipts (or duplicate item strings) into training.
- Scoring against ground-truth boxes that were not moved through the preprocessing transform.

## 2. Datasets (verified 2026-11-12)

| Dataset | Split (train/val/test) | Format facts | Role |
|---|---|---|---|
| **WildReceipt** | 1,267 / — / 472 (val = 10% of train, by receipt) | English photos. Boxes are 8 numbers (4 corners). 25 key classes (12 key/value pairs + Others) + Ignore. No license stated. | **Main train + test** for extraction |
| **CORD v2** | 800 / 100 / 100 | Indonesian. 4-corner quads in raw pixels. `row_id`, `group_id`, `roi`. Shop/payment labels removed, shop names blurred. CC BY 4.0. | Items + row grouping + crop eval. Extra training data (experiment only) |
| **SROIE 2019** | 626 / — / 347 (361 in `jsdnrs/ICDAR2019-SROIE`) | English scans (Malaysia). Fields: company, date, address, total — one string per document, uppercase. CC BY 4.0. | **Test-only** for header fields; rules baseline dev |
| ~~FUNSD~~ | — | Forms, not receipts | Dropped |

Official test splits are frozen and never used for tuning.

## 3. The contract (frozen on day 0)

Everything — extractors, scorer, API, frontend — speaks one JSON shape. Defined once in
`receipt/schema.py` (pydantic) and mirrored by hand in `frontend/src/types.ts`.

```json
{
  "merchant": "SAFEWAY", "address": "...", "date": "2023-02-16", "time": "14:02",
  "currency": "USD", "subtotal": "12.40", "tax": "0.99", "tips": null, "total": "13.39",
  "category": "groceries", "needs_review": false, "review_reasons": [],
  "items": [
    {"name": "ORG BNNA 2LB", "qty": "1", "unit_price": null, "price": "1.98", "category": "groceries"}
  ]
}
```

Money is a decimal string with 2 decimals; dates ISO. Unknown = `null`.

**Unified token label set** (all datasets mapped to it):
`MERCHANT ADDRESS DATE TIME ITEM_NAME ITEM_QTY ITEM_PRICE SUBTOTAL TAX TIPS TOTAL O`
(WildReceipt `*_key` boxes → `O`; CORD `menu.nm/cnt/price` → `ITEM_*`, `sub_total.*`/`total.*` → totals).

**Categories** (fixed list, item and receipt level):
`groceries, dining, drinks, household, health, transport_fuel, clothing, electronics, services, other`.

## 4. Pipeline stages

```
image ─► 1 preprocess ─► 2 OCR ─► 3 row grouping ─► 4 extractor (token labels)
      ─► 5 assemble Receipt JSON ─► 6 categorize ─► 7 validate ─► store ─► UI
```

### Stage 1 — Preprocessing (OpenCV)
1. Quality gate: Laplacian variance (blur) + share of near-255 pixels (glare) → reject with "retake photo".
2. Grayscale + rescale so characters are ~30–35 px tall.
3. 4-point perspective warp of the largest 4-corner contour. **Return the homography.**
4. Deskew via min-area boxes / Hough lines. **Return the rotation matrix.**
5. Shadow removal: dilate + median blur → background; `cv2.divide`.
6. Tesseract path only: adaptive Gaussian threshold + 3×3 median. PaddleOCR path stops at step 5.

Every step is a flag (`--steps 1,2,3,4,5`) so the benchmark can ablate it. Ground-truth boxes are
transformed with the returned matrices before scoring.

Diagnostics: blur-gate false-reject rate (<5%) on SROIE scans blurred on purpose at known
strengths (the label is the blur applied, so nothing is tagged by hand); warp IoU vs CORD `roi` (>0.85); deskew error on synthetically rotated SROIE
scans (median <0.5°). *Pass bars are the builder's guesses.*

### Stage 2 — OCR
PaddleOCR det+rec (`enable_mkldnn=False`), Tesseract as baseline. Output: words + 4-corner boxes +
confidence. Results cached to `data/ocr_cache/{engine}/{dataset}/{split}.jsonl` so training and
benchmarking never re-run OCR.
Diagnostics: detection H-mean @ IoU 0.5 and CER on SROIE test + WildReceipt test.

### Stage 3 — Row grouping
Sort by y-center; join a word to a row when vertical overlap > τ·min(height). τ tuned on CORD dev.
Diagnostic: pairwise same-row P/R on CORD test (`row_id`), on gold boxes and Paddle boxes.

### Stage 4 — Extractors (all output token labels from §3)
| Id | Method |
|---|---|
| `rules` | Regex + spatial anchors (rightmost amount on the TOTAL row, date/time patterns, top-rows merchant). |
| `crf` | `sklearn-crfsuite`; features: word shape, is-number, currency sign, relative x/y, row index, ±2 neighbor words, distance to nearest keyword. |
| `roberta` | `roberta-base` token classifier, text only. |
| `layoutlmv3` | `microsoft/layoutlmv3-base`, text + boxes (0–1000) + image. `apply_ocr=False`; first sub-token labeled, rest `-100`; >512 tokens → overflow windows, stride 128, keep higher-confidence prediction. |

LayoutLMv3 training: lr ~3e-5, batch 4, fp16, early stop on val score (§5). Two phases:
**A** on gold OCR words; **B** fine-tune on Paddle OCR words with labels projected from gold (the gold box holding
the word's centre; IoU breaks ties between overlapping boxes; a centre within 0.1 word height of a box still counts;
unmatched → `O`). Augment with small rotation/scale applied to image and boxes together.
**B4** (served): phase B on WildReceipt + CORD + SROIE train Paddle words. SROIE has document-level gold only, so its
words get weak labels (fuzzy merchant/address span, parsed date, total amount on a TOTAL row); unannotated SROIE words
below the header block are left out of the loss (`-100`), never trained as `O`. Early stopping on the mean of the four
val F1s that serving ranks by; SROIE val = every 10th SROIE train receipt.
GPU ~6–8 GB (fits the ≤14–15 GB budget on the shared host).
License note: LayoutLMv3 weights are CC BY-NC-SA (non-commercial; fine for a portfolio). LiLT (MIT)
is the drop-in if that ever matters.

### Stage 5 — Assembly
Token labels + rows → Receipt JSON. Contiguous same-label words merge into one value. Each row with
`ITEM_NAME` takes the `ITEM_QTY`/`ITEM_PRICE` on the same row, or the next row if the name wraps.
Values normalized (money → 2-decimal string, dates → ISO). **Shared by every extractor**, so the
benchmark compares extractors, not assembly code.

### Stage 6 — Categorization (no LLM, **no manual labeling**)
Every training label comes from a public dataset that is already labelled; nobody tags anything.
`scripts/build_category_data.py` downloads and maps them into `data/categories/{items,merchants}.csv`.

| Labels for | Source (already labelled) | Mapping to our categories |
|---|---|---|
| Item names | **Instacart Online Grocery** `products.csv`: ~49.7k product names, each with an aisle and one of 21 departments | Department → category: produce, dairy eggs, bakery, meat seafood, deli, frozen, pantry, canned goods, dry goods pasta, breakfast, snacks, international, bulk → `groceries`; beverages, alcohol → `drinks`; household, pets, babies → `household`; personal care → `health`; missing, other → dropped (leftover bins of mixed goods) |
| Item names (restaurant) | **CORD** `menu.nm` strings (every CORD receipt is a restaurant/café) | → `dining` |
| Item names (US restaurants) | **Souslab** US restaurant menus, 10k-item sample (CC BY-NC 4.0), drink sections left out | → `dining` |
| Merchant names | **OpenStreetMap** POI names via Overpass, several US cities, by tag | `shop=supermarket/convenience/greengrocer` → groceries; `amenity=restaurant/fast_food/cafe` → dining; `amenity=bar/pub`, `shop=alcohol` → drinks; `shop=hardware/doityourself/houseware` → household; `amenity=pharmacy`, `shop=chemist` → health; `amenity=fuel` → transport_fuel; `shop=clothes/shoes` → clothing; `shop=electronics/mobile_phone` → electronics; `shop=hairdresser/laundry/dry_cleaning` → services (ODbL, attributed); lookup-only: `shop=department_store/variety_store/general/wholesale` → `general`, books/stationery/gift/toys → other, and a few more |

- **Receipt-style names:** product names are clean ("Organic Bananas") but receipts abbreviate
  ("ORG BNNA 2LB"), so training adds generated variants: uppercase, vowels dropped, words cut to
  3–5 letters, truncated to 16–20 characters, sizes appended. Rule-based, deterministic.
- **Item level:** keyword dictionary baseline vs TF-IDF char 3–5-grams + logistic regression.
  Instacart only covers groceries, drinks, household, health and other, so an item **inherits the
  merchant's category** when the merchant is confidently dining, fuel, clothing, electronics or
  services (a restaurant's items are dining); the item classifier runs for groceries and general stores.
- **Merchant:** an exact-name lookup over every labelled name first (`categorize.merchant_key`: no spaces,
  store numbers or legal suffixes), then the classifier, confident at θ (val, 0.90 precision) or a stricter
  per-category θ for categories that overwrite items. `general` stores (department/variety/wholesale) are
  lookup-only and never hand their category to items.
- **Receipt level:** a confident merchant's category; else the money-weighted mean of the item model's
  probabilities, with 30% of the vote from an unsure merchant's probabilities. Items follow the receipt:
  a dining (or other INHERIT) receipt's items take its category; any other known receipt rules dining out.
- **Splits:** 70/15/15, grouped so a product or merchant name (normalized) never sits in two splits;
  chain names (e.g. every "Safeway") stay in one split.
- **Scores (no hand-labeled set):** macro-F1 on held-out Instacart products, reported on clean
  names and on their receipt-style variants; macro-F1 on held-out OSM merchant names; and a sanity
  check that CORD test receipts come out `dining`. There is no gold category for WildReceipt or
  SROIE receipts, so real-receipt category accuracy is not claimed.

### Stage 7 — Validation, storage, API
- Checks: Σ item prices ≈ subtotal (±0.01); subtotal + tax (+ tips) ≈ total; date parses.
  Any failure → `needs_review = true` with a reason.
- Postgres: `receipts` (`data jsonb` = the contract JSON, `extractor`, `raw_ocr jsonb`, `image_path`,
  `created_at`) and `corrections` (before/after JSON per edit). Stats query items via `jsonb_array_elements`.
  User edits via `PATCH` are saved and kept as `corrections` — new training labels for stage 4.
- FastAPI, synchronous, no auth:
  `POST /receipts` (multipart image) · `GET /receipts` · `GET /receipts/{id}` ·
  `PATCH /receipts/{id}` · `GET /stats` (spend by category × month) · `GET /health`.
- The pipeline sits behind one function `extract(image_bytes) -> Receipt`. Until models exist it is
  a **stub returning a fixture receipt**, so the web lane never waits on the ML lane.

## 5. Unified benchmark (one scorer for everything)

**One script, one table.** `scripts/benchmark.py --extractor {rules,crf,roberta,layoutlmv3}
--ocr {gold,paddle,tesseract} --steps 1,2,3,4,5` runs stages 1→5 on the frozen test sets and
appends one row to `eval/results.csv`. Every stage change (a preprocessing step, a new OCR engine,
a new extractor) is judged by the same numbers.

**Scorer (`receipt/score.py`)** compares predicted vs gold Receipt JSON, per receipt, after
normalization (uppercase, collapse spaces, money → 2 decimals, dates → ISO):

| Metric | Definition | Computed on |
|---|---|---|
| **Field F1** (headline) | Micro P/R/F1 over `(field, value)` pairs for header fields (`merchant, address, date, time, subtotal, tax, tips, total`). Fields the dataset doesn't label are masked out. Same idea as the official SROIE Task 3 protocol, extended to more fields. | WildReceipt test, SROIE test |
| **Item F1** | An item counts only if name, qty, and price all match a gold item (Hungarian match per receipt). | CORD test (`group_id` gold), WildReceipt test (gold items rebuilt from gold boxes + stage 3) |
| **Receipt exact** | Share of receipts where merchant, date, and total are all correct. | all three test sets |
| **Latency** | p50 / p95 seconds per receipt, stages 1→5. | WildReceipt test |

Token-level `seqeval` F1 is logged per learned model as a diagnostic, but **ranking uses Field F1**.

Results table shape (filled as runs finish):

| extractor | ocr | steps | WR field F1 | SROIE field F1 | CORD item F1 | WR item F1 | exact | p50 s |
|---|---|---|---|---|---|---|---|---|
| rules | gold | — | | | | | | |
| … | paddle | 1–5 | | | | | | |

Reference point: SDMG-R reports 0.89 macro-F1 on WildReceipt (token level, not directly comparable).

The scorer gets one small test file with hand-written gold/pred pairs (exact match, normalization,
masking, item matching) — it is the one piece of code everything else trusts.

## 6. Deployment

- **Frontend:** Vite + React + TypeScript in `frontend/`, deployed to Vercel (project root =
  `frontend`, framework = Vite, env `VITE_API_URL`). Pages: upload → editable result form;
  receipt list; spend-by-category chart.
- **API:** runs on the GPU box (uvicorn) next to Postgres (docker). Exposed over HTTPS with a
  Cloudflare Tunnel because the Vercel page is HTTPS. CORS allows the Vercel domain only.
  A quick tunnel (`trycloudflare.com`) changes URL on restart → redeploy with the new
  `VITE_API_URL`; a named tunnel needs a domain on Cloudflare (open item).
- Ports (builder's call, avoiding 18200/18280 used by another project): API 18300, Postgres 18332, Vite dev 18373.

## 7. Directory layout

```
rec/
├── docs/specs/                     this spec (+ plan later)
├── receipt/                        core Python package — the pipeline
│   ├── schema.py                   Receipt / Item pydantic models, label + category lists (contract)
│   ├── preprocess.py               stage 1 (6 steps, returns image + transforms)
│   ├── ocr.py                      stage 2 (paddle, tesseract, cache read/write)
│   ├── rows.py                     stage 3
│   ├── extractors/
│   │   ├── rules.py                stage 4a
│   │   ├── crf.py                  stage 4b
│   │   └── transformer.py          stage 4c+4d (roberta / layoutlmv3 inference)
│   ├── assemble.py                 stage 5
│   ├── categorize.py               stage 6
│   ├── validate.py                 stage 7 checks
│   ├── score.py                    unified scorer (§5)
│   └── pipeline.py                 extract(image) -> Receipt (stub until models land)
├── scripts/
│   ├── prepare_data.py             download + convert WildReceipt/CORD/SROIE → data/gold/*.jsonl
│   ├── build_ocr_cache.py          run OCR once over every split
│   ├── train_crf.py
│   ├── train_transformer.py        --model roberta|layoutlmv3 --phase A|B
│   ├── build_category_data.py      Instacart + CORD + OSM → data/categories/*.csv (no hand labels)
│   ├── train_categorizer.py
│   └── benchmark.py                one row → eval/results.csv
├── api/
│   ├── main.py                     FastAPI app
│   ├── db.py                       Postgres access
│   └── schema.sql                  tables
├── frontend/                       Vite + React + TS (Vercel)
│   └── src/{types.ts, api.ts, App.tsx}
├── eval/results.csv                benchmark table (committed)
├── tests/                          test_score.py, test_api.py, test_rows.py
├── data/   models/                 gitignored
├── docker-compose.yml              postgres
└── pyproject.toml
```

## 8. Parallel build plan

Only **Lane 0** blocks anything. After it, four lanes run at the same time; GPU jobs run in the
background while code gets written in the other lanes.

```
Day:        0        1        2        3        4        5
Lane 0  ████ contract + data + scorer
Lane W       ██████████████ web: API(stub) + DB + React + Vercel + tunnel ──► done early
Lane V  ██████████ preprocess + Paddle OCR cache + rows ─► ablation rows in benchmark
Lane M       ██ rules+CRF ░░░░ roberta(bg) ░░░░░░ layoutlmv3 A(bg) ░░░ B(bg, needs V cache)
Lane C       ████ build category data (Instacart/CORD/OSM) ── TF-IDF clf
Lane I                                         ████ swap stub → real pipeline, e2e bench, redeploy
            (░ = GPU job running in background while other lanes code)
```

| Lane | Starts when | Work | Done when |
|---|---|---|---|
| **0 Contract & data** | now | `schema.py`, `prepare_data.py` → `data/gold/{dataset}/{split}.jsonl` (words, boxes, unified labels, gold Receipt), `score.py` + its test | Gold files exist for all splits; scorer test passes; `rules` gets a first benchmark row on gold OCR |
| **W Web** | `schema.py` exists (hour 1) | API with stub `extract()`, Postgres, `PATCH` corrections, React UI, Vercel deploy, tunnel | Vercel URL: upload → stub receipt saved → listed → editable |
| **V Vision/OCR** | now (needs only images) | Stage 1 + 2 + 3, `build_ocr_cache.py` for Paddle (+ Tesseract on test) | Paddle cache for all splits; `--ocr paddle` benchmark rows; ablation rows per preprocessing step |
| **M Models** | Lane 0 done | `rules` → `crf` (CPU, minutes) → `roberta` (bg) → `layoutlmv3` phase A (bg) → phase B once Lane V cache exists | One benchmark row per extractor × OCR source |
| **C Categories** | Lane 0 done | `build_category_data.py` from public labelled data, train item + merchant classifiers | Macro-F1 on held-out items (clean + receipt-style) and merchants; CORD → dining check |
| **I Integration** | M phase A + V + C done | `pipeline.py` uses best extractor + categorizer; e2e benchmark; redeploy | Live site returns real extractions |

GPU rule: one transformer training job at a time (≤14–15 GB shared budget); Paddle OCR caching
(~2 GB) can run alongside it.

## 9. Open items
- Cloudflare named tunnel (needs a domain) vs quick tunnel with redeploys.
- ~~Vercel + GitHub~~: resolved — code lives at github.com/dangnguyenducanh/rec (public); import it in
  Vercel with root directory `frontend`.
- Who runs the lanes: separate Claude sessions per lane, or one session + background jobs.

## 10. Calls made by the builder (check these)
- Category list in §3; θ and τ thresholds tuned on val.
- Field F1 as the single ranking metric; seqeval only as a diagnostic.
- WildReceipt gold items rebuilt from gold boxes + row grouping (it has no item groups).
- Ports 18300 / 18332.
- Frontend types hand-mirrored from `schema.py` (no codegen).
- The live API's extractor is the available model with the best mean of the four F1 columns (fields and items) on
  Paddle OCR, not field F1 alone: on fields alone rules edge out LayoutLMv3-B (0.707 vs 0.698) while finding about half
  its items. Field F1 stays the benchmark's headline metric.
- A confidently identified non-restaurant merchant rules out `dining` for its items.
- SROIE train is training data for LayoutLMv3-B4 (weak labels), so SROIE test is no longer out of domain for B4; it
  still is for every other learned extractor.
- Assembly changes are chosen on val only (WildReceipt val, CORD val, the SROIE val tenth), each kept if it raised
  LayoutLMv3-B's val 4-F1 mean by ≥ 0.003 without lowering rules or CRF; every benchmark row is then re-run.
