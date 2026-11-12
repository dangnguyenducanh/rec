"""Stage 4b: sklearn-crfsuite token tagger. One sequence per receipt, words in reading order.
Features (spec §4): word shape, is-number, currency sign, relative x/y, row index, ±2 neighbour words,
row distance to the nearest keyword of each kind."""
import pickle
import re

from receipt import rows
from receipt.assemble import norm_date
from receipt.extractors.rules import SUBTOTAL, TAX, TIME, TIPS, TOTAL, _is_amount

KEYWORDS = {"total": TOTAL, "subtotal": SUBTOTAL, "tax": TAX, "tips": TIPS,
            "date": re.compile(r"DATE"), "time": re.compile(r"TIME"), "qty": re.compile(r"QTY|QUANTITY"),
            "cash": re.compile(r"CASH|CHANGE|TENDER|CARD|VISA"), "tel": re.compile(r"TEL|PHONE|FAX")}


def _shape(t: str) -> str:
    s = re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"\d", "d", t)))
    return re.sub(r"(.)\1+", r"\1\1", s)[:12]  # runs of 3+ collapse to 2


def _word(t: str, p: str) -> dict:
    return {f"{p}w": t.lower()[:20], f"{p}shape": _shape(t)}


def features(doc: dict) -> tuple[list[int], list[dict]]:
    """(reading-order word indices, one feature dict per word in that order)."""
    words = doc["words"]
    if not words:
        return [], []
    quads = [w["quad"] for w in words]
    rid = rows.group(quads)
    order = rows.order(quads, rid)
    W, H = doc.get("width") or 1, doc.get("height") or 1
    nrows = max(rid) + 1
    row_text = {}
    for i in order:
        row_text[rid[i]] = row_text.get(rid[i], "") + " " + words[i]["text"].upper()
    kw_rows = {k: [r for r, t in row_text.items() if rx.search(t)] for k, rx in KEYWORDS.items()}
    texts = [words[i]["text"] for i in order]
    feats = []
    for n, i in enumerate(order):
        t, q, r = words[i]["text"], words[i]["quad"], rid[i]
        u = t.upper()
        x, y = sum(q[0::2]) / 4 / W, sum(q[1::2]) / 4 / H
        f = {"bias": 1.0, "x": x, "y": y, f"xb={int(x * 5)}": 1.0, f"yb={int(y * 10)}": 1.0,
             f"row={min(r, 8)}": 1.0, f"rowrev={min(nrows - 1 - r, 8)}": 1.0,
             "first_in_row": n == 0 or rid[order[n - 1]] != r,
             "last_in_row": n == len(order) - 1 or rid[order[n + 1]] != r,
             "is_number": bool(re.fullmatch(r"[\d.,\-]+", t)), "has_digit": any(c.isdigit() for c in t),
             "is_amount": _is_amount(u), "is_date": norm_date(t) is not None, "is_time": bool(TIME.search(u)),
             "currency": bool(re.search(r"[$€£￡¥]|^RM|^RP|USD", u)), "colon": ":" in t, "pct": "%" in t,
             "hash": "#" in t, "len": min(len(t), 20) / 20, "upper": t.isupper(),
             "pre3": t[:3].lower(), "suf3": t[-3:].lower(), **_word(t, "")}
        for k, rs in kw_rows.items():
            d = min((r - kr for kr in rs if kr <= r), default=None)
            f[f"kw_{k}"] = "none" if d is None else str(min(d, 3))
        for o in (-2, -1, 1, 2):
            if 0 <= n + o < len(order):
                f.update(_word(texts[n + o], f"{o}:"))
                f[f"{o}:same_row"] = rid[order[n + o]] == r
            else:
                f[f"{o}:edge"] = 1.0
        feats.append(f)
    return order, feats


def train(docs: list[dict], labels: list[list[str]], c1: float, c2: float, max_iter: int = 150):
    import sklearn_crfsuite
    X, y = [], []
    for doc, labs in zip(docs, labels):
        order, f = features(doc)
        X.append(f)
        y.append([labs[i] for i in order])
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iter,
                               all_possible_transitions=True)
    crf.fit(X, y)
    return crf


def predictor(crf):
    def extract(doc: dict) -> list[str]:
        order, f = features(doc)
        out = ["O"] * len(doc["words"])
        if f:
            for i, lab in zip(order, crf.predict_single(f)):
                out[i] = lab
        return out
    return extract


def load(path):
    with open(path, "rb") as fh:
        return predictor(pickle.load(fh))
