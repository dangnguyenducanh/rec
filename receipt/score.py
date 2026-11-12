"""Unified scorer (spec §5, interfaces §6): field F1, item F1, receipt exact."""
from collections import Counter

from receipt.assemble import build, norm_money, norm_text, norm_time
from receipt.schema import Receipt

HEADER = ["merchant", "address", "date", "time", "subtotal", "tax", "tips", "total"]
EXACT = ["merchant", "date", "total"]


def gold_receipt(rec: dict) -> Receipt:
    if rec["gold"] is not None:
        return Receipt.model_validate(rec["gold"])
    words = rec["words"]
    return build(words, [w["label"] or "O" for w in words], rec["dayfirst"])


def _norm(field: str, v):
    if v is None:
        return None
    if field in ("merchant", "address"):
        return norm_text(v)
    if field == "date":
        return v.isoformat()
    if field == "time":
        return norm_time(v)
    return norm_money(v)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def _item_keys(r: Receipt) -> Counter:
    return Counter((norm_text(i.name), norm_money(i.price), i.qty or "1") for i in r.items)


def score(rows: list[tuple[Receipt, Receipt, dict]]) -> dict:
    """rows = (pred, gold, gold record). Unlabelled fields and items_labeled=false are skipped."""
    ftp = ffp = ffn = itp = ifp = ifn = exact_hits = exact_n = 0
    for pred, gold, rec in rows:
        fields = [f for f in HEADER if f in rec["fields"]]
        ok = {}
        for f in fields:
            p, g = _norm(f, getattr(pred, f)), _norm(f, getattr(gold, f))
            ok[f] = p == g
            ftp += p is not None and p == g
            ffp += p is not None and p != g
            ffn += g is not None and p != g
        keys = [f for f in EXACT if f in ok]
        if keys:
            exact_n += 1
            exact_hits += all(ok[f] for f in keys)
        if rec["items_labeled"]:
            P, G = _item_keys(pred), _item_keys(gold)
            m = sum((P & G).values())
            itp, ifp, ifn = itp + m, ifp + sum(P.values()) - m, ifn + sum(G.values()) - m
    fp_, fr_, ff_ = _prf(ftp, ffp, ffn)
    ip_, ir_, if_ = _prf(itp, ifp, ifn)
    return {"n": len(rows), "field_p": fp_, "field_r": fr_, "field_f1": ff_,
            "item_p": ip_, "item_r": ir_, "item_f1": if_,
            "exact": round(exact_hits / exact_n, 4) if exact_n else 0.0,
            "field_tp": ftp, "field_fp": ffp, "field_fn": ffn,
            "item_tp": itp, "item_fp": ifp, "item_fn": ifn, "exact_hits": exact_hits, "exact_n": exact_n}
