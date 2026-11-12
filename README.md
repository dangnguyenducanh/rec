# Receipt Reader

Receipt photo → OpenCV preprocessing → PaddleOCR → LayoutLMv3 extraction → categories → Postgres → React (Vercel).

Design spec: [docs/specs/2026-11-12-receipt-pipeline-design.md](docs/specs/2026-11-12-receipt-pipeline-design.md)

## Demo

[▶ Watch the 45-second walkthrough](docs/demo/receipt-reader-demo.mp4): upload a photo → read it → fix a typo the
checks catch → saved list → spending by category.

| Editor | Spending |
|---|---|
| ![Editor](docs/demo/editor.png) | ![Spending](docs/demo/spending.png) |

> The web app is complete; extraction is still a **stub** that returns a fixed receipt (shown as
> "read by: stub"). The OCR + LayoutLMv3 pipeline replaces it in Lane I of the spec.

## Run locally

```bash
docker compose -p rec up -d                                   # Postgres on :18332
uv venv .venv && uv pip install -r pyproject.toml --extra dev
.venv/bin/uvicorn api.main:app --port 18300                   # API on :18300
cd frontend && npm install && npm run dev                     # UI on :18373
```

Tests: `.venv/bin/python -m pytest -q` (uses a throwaway `rec_test` database).
Re-record the demo (empty DB, both servers up): `.venv/bin/python -m playwright install chromium && .venv/bin/python scripts/record_demo.py`.
