"""Categories on real receipts without hand labels -> eval/categories_receipts.json (README, Improving the categorizer).

WildReceipt receipts are labelled restaurant or store by a fixed rule on the merchant name (an unambiguous
name-suggestion-index dining or grocery brand, or a keyword); categorize.apply then runs with the merchant kept and
hidden. The "kept" numbers are optimistic (the rule and the merchant lookup share NSI brand names); the "hidden"
ones see only the items. --extractor reads the items from that extractor on the Paddle OCR cache instead of gold.
  .venv/bin/python -s scripts/eval_categorizer_receipts.py [--extractor layoutlmv3-B4]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_category_data import TAG  # noqa: E402
from receipt import categorize  # noqa: E402
from receipt.categorize import squash  # noqa: E402
from receipt.score import gold_receipt  # noqa: E402

DINING_WORDS = ["restaurant", "cafe", "pizza", "grill", "bistro", "kitchen", "diner", "burger", "sushi", "bbq",
                "noodle", "trattoria", "taqueria", "ristorante", "brasserie", "steakhouse", "tavern"]
GROCERY_WORDS = ["supermarket", "grocery", "market"]


def labeler():
    """merchant name -> 'dining' | 'groceries' | None (ambiguous or unknown)."""
    nsi = json.load(open(ROOT / "data/categories/raw/nsi.min.json"))["nsi"]
    cats = {}
    for (k, v), cat in TAG.items():
        for it in nsi.get(f"brands/{k}/{v}", {}).get("items", []):
            n = squash(it["tags"].get("brand") or it["tags"].get("name") or it["displayName"])
            if len(n) >= 4:
                cats.setdefault(n, set()).add(cat)
    dining = {n for n, c in cats.items() if c == {"dining"}}
    grocery = {n for n, c in cats.items() if c == {"groceries"} and len(n) >= 5}

    def label(merchant):
        s = re.sub(r"^welcometo", "", squash(merchant or ""))
        if not s:
            return None
        d = any(s.startswith(x) for x in dining) or any(w in s for w in DINING_WORDS)
        g = any(s.startswith(x) for x in grocery) or any(w in s for w in GROCERY_WORDS)
        return "dining" if d and not g else "groceries" if g and not d else None
    return label


def receipts(split: str, extractor: str | None):
    """(gold merchant, Receipt to categorize) per WildReceipt receipt."""
    recs = [json.loads(line) for line in open(ROOT / f"data/gold/wildreceipt/{split}.jsonl")]
    if not extractor:
        return [(gold_receipt(r).merchant, gold_receipt(r)) for r in recs]
    from benchmark import run
    from receipt import ocr
    from receipt.extractors import get
    res = run(get(extractor), recs, ocr.load_cache("paddle", "2345", "wildreceipt", split))
    return [(gold_receipt(x["rec"]).merchant, x["pred"]) for x in res]


def score(rows, label) -> dict:
    out, sure = {}, []
    for hide in (False, True):
        rest, store = [], []
        for merchant, g in rows:
            lab = label(merchant)
            if not lab or not g.items:
                continue
            if not hide:
                sure.append(bool(g.merchant) and categorize.merchant_category(g.merchant)[1])
            r = categorize.apply(g.model_copy(update={"merchant": None}) if hide else g)
            if lab == "dining":
                rest.append(r.category == "dining")
            elif "target" not in (merchant or "").lower():  # Target sells groceries and meals: left out
                store.append(r.category == "dining")
        out["merchant_hidden" if hide else "merchant_kept"] = {
            "restaurant_to_dining": round(float(np.mean(rest)), 3), "n_restaurant": len(rest),
            "store_to_dining": round(float(np.mean(store)), 3), "n_store": len(store)}
    out["merchant_confident"] = round(float(np.mean(sure)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor")
    a = ap.parse_args()
    label = labeler()
    res = {f"wildreceipt_{s}_gold": score(receipts(s, None), label) for s in ("train", "val")}
    if a.extractor:
        res[f"wildreceipt_val_{a.extractor}"] = score(receipts("val", a.extractor), label)
    (ROOT / "eval/categories_receipts.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
