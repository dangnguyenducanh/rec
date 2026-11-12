"""Train item + merchant classifiers on data/categories/*.csv (spec §4 stage 6).

Keyword baseline vs TF-IDF char 3-5-grams + LogisticRegression; C picked and merchant theta tuned on
val; scored once on test. Writes models/categorizer/{items,merchants}.joblib and eval/categories.json.
Run: .venv/bin/python -s scripts/train_categorizer.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_category_data import receipt_style  # noqa: E402  (same scripts/ dir)
from receipt import categorize  # noqa: E402
from receipt.schema import Receipt  # noqa: E402

OUT = Path("models/categorizer")
THETA_PRECISION = 0.90  # ponytail: a guessed target; theta = lowest val confidence keeping precision >= this


def macro(y, p):
    return round(float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)), 4)


def fit(train, val):
    """TF-IDF + balanced LogReg; C chosen by val macro-F1."""
    best = None
    for C in [1, 4, 16]:
        m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
                          LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
        m.fit(train.name, train.category)
        f = macro(val.category, m.predict(val.name))
        print(f"  C={C} val macro-F1={f}")
        if best is None or f > best[0]:
            best = (f, C, m)
    return best


def run(kind: str, table: dict, default: str, size: bool) -> dict:
    d = pd.read_csv(f"data/categories/{kind}.csv", keep_default_na=False)
    tr, va, te = (d[d.split == s] for s in ["train", "val", "test"])
    rs = [receipt_style(n, size) for n in te.name]  # receipt-style view of the held-out names
    print(kind, len(tr), len(va), len(te))
    val_f1, C, m = fit(tr, va)
    kw = lambda names: [categorize.keyword(n, table, default)[0] for n in names]  # noqa: E731
    res = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "C": C, "val_macro_f1": val_f1,
           "test_macro_f1": {"keyword_clean": macro(te.category, kw(te.name)), "model_clean": macro(te.category, m.predict(te.name)),
                             "keyword_receipt_style": macro(te.category, kw(rs)), "model_receipt_style": macro(te.category, m.predict(rs))},
           "test_per_class_f1_model_clean": dict(zip(sorted(set(te.category)), np.round(f1_score(
               te.category, m.predict(te.name), labels=sorted(set(te.category)), average=None, zero_division=0), 4).tolist()))}
    theta = 0.0
    if kind == "merchants":
        theta, res["theta"] = tune_theta(m, va), {}
        for name, part in [("val", va), ("test", te)]:
            p = m.predict_proba(part.name)
            ok = p.max(1) >= theta
            right = m.classes_[p.argmax(1)] == part.category.to_numpy()
            res["theta"][name] = {"coverage": round(float(ok.mean()), 4), "precision": round(float(right[ok].mean()), 4)}
        res["theta"]["value"] = theta
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "theta": theta}, OUT / f"{kind}.joblib")
    return res


def tune_theta(m, va) -> float:
    """Lowest confidence threshold whose accepted val merchants are >= THETA_PRECISION correct."""
    p = m.predict_proba(va.name)
    conf = p.max(1)
    order = np.argsort(-conf)
    right = (m.classes_[p.argmax(1)] == va.category.to_numpy())[order]
    prec = np.cumsum(right) / np.arange(1, len(right) + 1)
    ok = np.where(prec >= THETA_PRECISION)[0]
    return round(float(conf[order][ok[-1]]), 4) if len(ok) else 1.0


def cord_dining_share() -> float:
    """Sanity check: CORD test receipts (all restaurants, merchant blurred) should come out dining."""
    categorize._load.cache_clear()
    rs = [categorize.apply(Receipt.model_validate(json.loads(line)["gold"])) for line in open("data/gold/cord/test.jsonl")]
    return round(sum(r.category == "dining" for r in rs) / len(rs), 4)


if __name__ == "__main__":
    res = {"items": run("items", categorize._ITEM_RE, "groceries", True),
           "merchants": run("merchants", categorize._MERCHANT_RE, "other", False)}
    res["cord_test_dining_share"] = cord_dining_share()
    Path("eval/categories.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
