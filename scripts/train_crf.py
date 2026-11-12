"""Train the CRF extractor on WildReceipt train (gold words); pick c1/c2 by WildReceipt val field F1.

.venv/bin/python -s scripts/train_crf.py  ->  models/crf.pkl, models/crf_grid.json
"""
import itertools
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import doc_of, field_f1, gold_words, load_gold, run  # noqa: E402
from receipt.extractors import crf  # noqa: E402

GRID = list(itertools.product([0.01, 0.1, 0.5], [0.01, 0.05, 0.2]))


def fit(c):
    c1, c2 = c
    tr = load_gold("wildreceipt", "train")
    model = crf.train([doc_of(r, gold_words(r)) for r in tr], [[w["label"] for w in r["words"]] for r in tr], c1, c2)
    f1 = field_f1(run(crf.predictor(model), load_gold("wildreceipt", "val")))
    print(f"c1={c1} c2={c2} val field F1={f1}", flush=True)
    return f1, c1, c2, model


def main():
    with mp.Pool(min(8, len(GRID))) as pool:
        res = pool.map(fit, GRID)
    f1, c1, c2, model = max(res, key=lambda r: r[0])
    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models/crf.pkl", "wb") as fh:
        pickle.dump(model, fh)
    (ROOT / "models/crf_grid.json").write_text(json.dumps(
        {"grid": [{"c1": r[1], "c2": r[2], "val_field_f1": r[0]} for r in res], "best": {"c1": c1, "c2": c2, "val_field_f1": f1}},
        indent=1))
    print(f"best c1={c1} c2={c2} val field F1={f1} -> models/crf.pkl")


if __name__ == "__main__":
    main()
