"""Run stages 1-5 on frozen test sets; append one row to eval/results.csv (interfaces §6).

.venv/bin/python -s scripts/benchmark.py --extractor rules --ocr gold [--steps 2,3,4,5] [--limit N]
--limit N is a smoke run: N docs per dataset, writes eval/runs/..._limitN.json, no CSV row.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from receipt import ocr  # noqa: E402
from receipt.assemble import build  # noqa: E402
from receipt.score import gold_receipt, score  # noqa: E402

DATASETS = ["wildreceipt", "sroie", "cord"]
COLS = ["extractor", "ocr", "steps", "wr_field_f1", "sroie_field_f1", "cord_item_f1", "wr_item_f1", "exact", "p50_s"]
RESULTS = ROOT / "eval/results.csv"


def load_gold(dataset: str, split: str) -> list[dict]:
    with open(ROOT / f"data/gold/{dataset}/{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def doc_of(rec: dict, words: list[dict]) -> dict:
    return {"words": words, "width": rec["width"], "height": rec["height"], "image": str(ROOT / rec["image"])}


def gold_words(rec: dict) -> list[dict]:
    return [{"text": w["text"], "quad": w["quad"]} for w in rec["words"]]  # labels removed


def run(fn, recs: list[dict], cache: dict | None = None) -> list[dict]:
    """Extract + assemble each record. cache=None feeds gold words. One dict per record."""
    out = []
    for rec in recs:
        c = cache.get(rec["id"], {"words": [], "ocr_s": 0.0}) if cache is not None else None
        words = c["words"] if c else gold_words(rec)
        t = time.perf_counter()
        labels = fn(doc_of(rec, words))
        ext_s = time.perf_counter() - t
        out.append({"rec": rec, "labels": labels, "pred": build(words, labels, rec["dayfirst"]),
                    "gold": gold_receipt(rec), "ocr_s": c["ocr_s"] if c else 0.0, "ext_s": ext_s,
                    "missing": c is not None and rec["id"] not in cache})
    return out


def field_f1(results: list[dict]) -> float:
    return score([(r["pred"], r["gold"], r["rec"]) for r in results])["field_f1"]


def seqeval_f1(results: list[dict]) -> dict:
    """Token-level: every non-O word is its own one-token entity (B-label), so this is word-level micro F1."""
    from seqeval.metrics import classification_report, f1_score
    tag = lambda labs: [["O" if x == "O" else f"B-{x}" for x in labs]]  # noqa: E731
    y_true, y_pred = [], []
    for r in results:
        y_true += tag([w["label"] or "O" for w in r["rec"]["words"]])
        y_pred += tag(r["labels"])
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"f1": round(f1_score(y_true, y_pred), 4),
            "per_label": {k: round(v["f1-score"], 4) for k, v in rep.items() if "avg" not in k}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--ocr", required=True, choices=["gold", "paddle", "tesseract"])
    ap.add_argument("--steps", default="2,3,4,5")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from receipt.extractors import get
    steps = "none" if a.ocr == "gold" else ocr.steps_tag(a.steps)
    fn = get(a.extractor)

    res = {}
    for ds in DATASETS:
        recs = load_gold(ds, "test")[:a.limit]
        cache = None if a.ocr == "gold" else ocr.load_cache(a.ocr, steps, ds, "test")
        res[ds] = run(fn, recs, cache)
        print(f"{ds}: {len(recs)} docs", flush=True)

    sc = {ds: score([(r["pred"], r["gold"], r["rec"]) for r in rs]) for ds, rs in res.items()}
    allrows = [(r["pred"], r["gold"], r["rec"]) for rs in res.values() for r in rs]
    lat = [r["ocr_s"] + r["ext_s"] for r in res["wildreceipt"]]  # spec §5: latency on WildReceipt test
    row = {"extractor": a.extractor, "ocr": a.ocr, "steps": steps,
           "wr_field_f1": sc["wildreceipt"]["field_f1"], "sroie_field_f1": sc["sroie"]["field_f1"],
           "cord_item_f1": sc["cord"]["item_f1"], "wr_item_f1": sc["wildreceipt"]["item_f1"],
           "exact": score(allrows)["exact"], "p50_s": round(statistics.median(lat), 4) if lat else None}
    detail = {"row": row, "scores": sc, "p95_s": round(statistics.quantiles(lat, n=20)[-1], 4) if len(lat) > 1 else None,
              "missing_ocr": {ds: sum(r["missing"] for r in rs) for ds, rs in res.items()},
              "docs": [{"id": r["rec"]["id"], "pred": r["pred"].model_dump(mode="json"), "labels": r["labels"],
                        "ocr_s": round(r["ocr_s"], 4), "ext_s": round(r["ext_s"], 4)} for rs in res.values() for r in rs]}
    if a.extractor != "rules" and a.ocr == "gold":
        detail["seqeval_wr_test"] = seqeval_f1(res["wildreceipt"])

    name = f"{a.extractor}_{a.ocr}_s{steps}" + (f"_limit{a.limit}" if a.limit else "")
    (ROOT / "eval/runs").mkdir(parents=True, exist_ok=True)
    (ROOT / f"eval/runs/{name}.json").write_text(json.dumps(detail, indent=1))
    if not a.limit:  # one row per (extractor, ocr, steps): a rerun replaces its old row
        with open(RESULTS) as f:
            old = [r for r in csv.DictReader(f) if (r["extractor"], r["ocr"], r["steps"]) != (a.extractor, a.ocr, steps)]
        with open(RESULTS, "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows(old + [row])
    print(json.dumps(row))
    if "seqeval_wr_test" in detail:
        print("seqeval WR test:", detail["seqeval_wr_test"]["f1"])


if __name__ == "__main__":
    main()
