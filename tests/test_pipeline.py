"""Real pipeline on docs/demo/sample-receipt.jpg (synthetic). Runs Paddle OCR + the chosen extractor on GPU.
Prints which known fields came out right and the latency per stage (pytest -s); only asserts it runs."""
from pathlib import Path

from receipt import pipeline
from receipt.schema import Receipt

SAMPLE = Path(__file__).resolve().parent.parent / "docs/demo/sample-receipt.jpg"
KNOWN = {"merchant": "GREEN VALLEY MARKET", "date": "2026-11-08", "total": "21.32"}


def test_real_extract(monkeypatch):
    monkeypatch.delenv("RECEIPT_PIPELINE", raising=False)  # test_api sets stub mode process-wide
    data = SAMPLE.read_bytes()
    pipeline.load()  # what the API does at startup: load + warm
    receipt, raw = pipeline.extract(data)  # = the first upload

    assert isinstance(receipt, Receipt)
    assert raw["engine"] == "paddle" and raw["words"] and len(raw["M"]) == 3
    got = {k: str(getattr(receipt, k)) if getattr(receipt, k) is not None else None for k in KNOWN}
    right = {k: got[k] == v for k, v in KNOWN.items()}
    print(f"\nextractor {pipeline.EXTRACTOR}; {len(raw['words'])} words; seconds {raw['seconds']}")
    print(f"fields right {right}; got {got}; items {len(receipt.items)}/5")
    print(receipt.model_dump_json(indent=1))
