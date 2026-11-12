"""Hand-written gold/pred pairs for the unified scorer."""
from receipt.schema import Item, Receipt
from receipt.score import HEADER, gold_receipt, score

ALL = {"fields": HEADER, "items_labeled": True}


def R(**kw):
    return Receipt(**kw)


GOLD = R(merchant="Safe Way", address="12 Main St", date="2023-02-16", time="17:42", subtotal="12.40",
         tax="0.99", total="13.39", items=[Item(name="ORG BNNA", price="1.98"), Item(name="MILK", qty="2", price="3.00")])


def test_exact_match():
    s = score([(GOLD, GOLD, ALL)])
    assert (s["n"], s["field_f1"], s["item_f1"], s["exact"]) == (1, 1.0, 1.0, 1.0)
    assert (s["field_tp"], s["field_fp"], s["field_fn"]) == (7, 0, 0)  # tips is None on both sides


def test_normalization_equalities():
    pred = R(merchant="SAFEWAY", address=" 12 main st. ", date="2023-02-16", time="17:42", subtotal="12.40",
             tax="0.99", total="13.39", items=[Item(name="org  bnna", qty="1", price="1.98"),
                                               Item(name="Milk", qty="2", price="3.00")])
    s = score([(pred, GOLD, ALL)])
    assert (s["field_f1"], s["item_f1"], s["exact"]) == (1.0, 1.0, 1.0)


def test_fp_fn_counting_and_masking():
    pred = R(merchant="SAFEWAY", date="2023-02-17", tips="1.00", total="13.39")
    s = score([(pred, GOLD, ALL)])
    # TP merchant,total; date wrong -> FP+FN; tips spurious -> FP; address,time,subtotal,tax missed -> FN
    assert (s["field_tp"], s["field_fp"], s["field_fn"]) == (2, 2, 5)
    assert (s["field_p"], s["field_r"], s["field_f1"]) == (0.5, 0.2857, 0.3636)
    assert s["exact"] == 0.0
    sroie = {"fields": ["merchant", "address", "date", "total"], "items_labeled": False}
    s = score([(pred, GOLD, sroie)])  # tips/time/subtotal/tax masked; items skipped
    assert (s["field_tp"], s["field_fp"], s["field_fn"]) == (2, 1, 2)
    assert (s["item_tp"], s["item_fp"], s["item_fn"]) == (0, 0, 0)
    cord = {"fields": ["subtotal", "tax", "total"], "items_labeled": True}
    s = score([(R(total="13.39"), GOLD, cord)])  # exact uses only the labelled total
    assert s["exact"] == 1.0 and (s["field_tp"], s["field_fn"]) == (1, 2)


def test_item_multiset():
    gold = R(items=[Item(name="COKE", price="2.00")] * 2 + [Item(name="FRIES", price="3.00")])
    pred = R(items=[Item(name="COKE", price="2.00")] * 3 + [Item(name="FRIES", qty="2", price="3.00")])
    s = score([(pred, gold, ALL)])
    assert (s["item_tp"], s["item_fp"], s["item_fn"]) == (2, 2, 1)


def test_exact_over_receipts():
    bad = R(merchant="SAFEWAY", date="2023-02-16", total="13.00")
    s = score([(GOLD, GOLD, ALL), (bad, GOLD, ALL)])
    assert (s["exact"], s["exact_hits"], s["n"]) == (0.5, 1, 2)


def test_gold_receipt():
    rec = {"gold": GOLD.model_dump(mode="json"), "dayfirst": False}
    assert gold_receipt(rec) == GOLD
    q = lambda x: [x, 0, x + 50, 0, x + 50, 30, x, 30]
    rec = {"gold": None, "dayfirst": True, "words": [
        {"text": "TOTAL", "quad": q(0), "label": "O"}, {"text": "9,50", "quad": q(100), "label": "TOTAL"},
        {"text": "05/04/23", "quad": q(200), "label": "DATE"}, {"text": "x", "quad": q(300), "label": None}]}
    g = gold_receipt(rec)
    assert (g.total, str(g.date)) == ("9.50", "2023-04-05")
