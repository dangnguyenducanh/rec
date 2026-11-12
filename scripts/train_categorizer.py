"""Train item + merchant classifiers on data/categories/*.csv (spec §4 stage 6).

Keyword baseline vs TF-IDF char 3-5-grams + LogisticRegression; C picked and merchant theta tuned on
val; scored once on test. Merchants also get an exact-name lookup (merchant_key -> majority category over every
NSI brand entry and OSM POI, lookup-only ones included); reported numbers use a lookup of train names only.
Writes models/categorizer/{items,merchants}.joblib and eval/categories.json.
Run: .venv/bin/python -s scripts/train_categorizer.py
"""
import collections
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, make_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_category_data import majority, norm, nsi_rows, osm_rows, receipt_style, split_of  # noqa: E402
from receipt import categorize  # noqa: E402
from receipt.schema import Receipt  # noqa: E402

OUT = Path("models/categorizer")
THETA_PRECISION = 0.90  # ponytail: a guessed target; theta = lowest val confidence keeping precision >= this


def macro(y, p):
    return round(float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)), 4)


def fit(train, val, class_weight):
    """TF-IDF (word-edge char n-grams + the same with spaces squashed out) + LogReg; C by val macro-F1."""
    tfidf = lambda **kw: TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True, min_df=2, **kw)  # noqa: E731
    best = None
    for C in [1, 4, 16]:
        m = make_pipeline(make_union(tfidf(analyzer="char_wb"), tfidf(analyzer="char", preprocessor=categorize.squash)),
                          LogisticRegression(C=C, class_weight=class_weight, max_iter=3000))
        m.fit(train.name, train.category)
        f = macro(val.category, m.predict(val.name))
        print(f"  C={C} val macro-F1={f}")
        if best is None or f > best[0]:
            best = (f, C, m)
    return best


def lookup(rows) -> dict:
    """merchant_key -> majority category of (name, category) rows, one per NSI brand entry or OSM POI: a chain's
    264 Shell stations outvote the name variants of its 95 Shell convenience stores (keys of 3+ characters)."""
    t = {}
    for n, c in rows:
        k = categorize.merchant_key(n)
        if len(k) >= 3:
            t.setdefault(k, collections.Counter())[c] += 1
    return {k: majority(c) for k, c in t.items()}


def run(kind: str, table: dict, default: str, size: bool) -> dict:
    d = pd.read_csv(f"data/categories/{kind}.csv", keep_default_na=False)
    d = d[~d.source.str.endswith("-lk")]  # lookup-only rows (incl. every "general"/"other" merchant) never train
    tr, va, te = (d[d.split == s] for s in ["train", "val", "test"])
    rs = [receipt_style(n, size) for n in te.name]  # receipt-style view of the held-out names
    print(kind, len(tr), len(va), len(te))
    # Items unweighted: balanced, the small dining class (CORD + US menus) swallows store items on real receipts.
    # Merchants balanced, or dining (over half the names) swallows the few health/electronics/fuel names.
    val_f1, C, m = fit(tr, va, None if kind == "items" else "balanced")
    kw = lambda names: [categorize.keyword(n, table, default)[0] for n in names]  # noqa: E731
    res = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "C": C, "val_macro_f1": val_f1,
           "test_macro_f1": {"keyword_clean": macro(te.category, kw(te.name)), "model_clean": macro(te.category, m.predict(te.name)),
                             "keyword_receipt_style": macro(te.category, kw(rs)), "model_receipt_style": macro(te.category, m.predict(rs))},
           "test_per_class_f1_model_clean": dict(zip(sorted(set(te.category)), np.round(f1_score(
               te.category, m.predict(te.name), labels=sorted(set(te.category)), average=None, zero_division=0), 4).tolist()))}
    out = {"model": m, "theta": 0.0}
    if kind == "merchants":
        out["theta"], res["theta"], res["with_train_lookup"] = tune_theta(m, va), {}, {}
        out["theta_class"] = tune_theta_class(m, va, out["theta"])
        raw = [(n, c, split_of(key or norm(n))) for n, c, _, key in [*nsi_rows(), *osm_rows()]]  # cached raw files
        lk = lookup((n, c) for n, c, sp in raw if sp == "train")
        for name, part in [("val", va), ("test", te)]:
            p = m.predict_proba(part.name)
            pred, gold = m.classes_[p.argmax(1)], part.category.to_numpy()
            ok = p.max(1) >= np.array([out["theta_class"].get(c, out["theta"]) for c in pred])
            res["theta"][name] = {"coverage": round(float(ok.mean()), 4), "precision": round(float((pred == gold)[ok].mean()), 4)}
            hit = [lk.get(categorize.merchant_key(n)) for n in part.name]
            pred = np.array([h or c for h, c in zip(hit, pred)])
            ok |= np.array([h is not None for h in hit])
            res["with_train_lookup"][name] = {"hits": sum(h is not None for h in hit), "macro_f1": macro(gold, pred),
                                              "coverage": round(float(ok.mean()), 4), "precision": round(float((pred == gold)[ok].mean()), 4)}
        res["theta"]["value"], res["theta"]["class"] = out["theta"], out["theta_class"]
        out["lookup"] = lookup((n, c) for n, c, _ in raw)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, OUT / f"{kind}.joblib")
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


def tune_theta_class(m, va, theta, min_n=10) -> dict:
    """Per INHERIT category (a confident one overwrites every item's category): the lowest confidence at which val
    names predicted as it are >= THETA_PRECISION correct over >= min_n names, never below theta; 1.01 (never
    confident) when no cut is."""
    p = m.predict_proba(va.name)
    conf, pred, gold = p.max(1), m.classes_[p.argmax(1)], va.category.to_numpy()
    out = {}
    for c in sorted(categorize.INHERIT):
        s = pred == c
        order = np.argsort(-conf[s])
        n = np.arange(1, s.sum() + 1)
        ok = np.where((np.cumsum(gold[s][order] == c) / n >= THETA_PRECISION) & (n >= min_n))[0]
        out[c] = round(max(theta, float(conf[s][order][ok[-1]])), 4) if len(ok) else 1.01
    return out


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
