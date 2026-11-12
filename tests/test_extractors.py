"""Rules on a synthetic receipt; window merge keeps the confident window; phase-B label projection."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from receipt.assemble import build  # noqa: E402
from receipt.extractors import rules, transformer  # noqa: E402


def W(text, x, y, w=None):
    w = w or 12 * len(text)
    return {"text": text, "quad": [x, y, x + w, y, x + w, y + 20, x, y + 20]}


DOC = {"width": 600, "height": 900, "image": None, "words": [
    W("GREEN", 200, 10), W("VALLEY", 270, 10), W("MARKET", 350, 10),
    W("1420", 200, 40), W("J", 255, 40), W("STREET", 275, 40),
    W("SACRAMENTO,", 200, 70), W("CA", 340, 70), W("95814", 375, 70),
    W("TEL", 200, 100), W("916-555-0100", 250, 100),
    W("11/08/2026", 50, 140), W("5:42", 300, 140), W("PM", 360, 140),
    W("BANANAS", 50, 200), W("1.98", 450, 200),
    W("MILK", 50, 230), W("3.49", 450, 230),
    W("SUBTOTAL", 50, 280), W("5.47", 450, 280),
    W("TAX", 50, 310), W("0.40", 450, 310),
    W("TOTAL", 50, 340), W("5.87", 450, 340),
    W("CASH", 50, 370), W("10.00", 450, 370),
]}


def test_rules_synthetic_receipt():
    r = build(DOC["words"], rules.extract(DOC))
    assert (r.merchant, r.address) == ("GREEN VALLEY MARKET", "1420 J STREET SACRAMENTO, CA 95814")
    assert (str(r.date), r.time) == ("2026-11-08", "17:42")
    assert (r.subtotal, r.tax, r.total) == ("5.47", "0.40", "5.87")
    assert [(i.name, i.price) for i in r.items] == [("BANANAS", "1.98"), ("MILK", "3.49")]


class Enc(dict):  # stand-in for a tokenizer BatchEncoding: 2 windows overlapping on word 1
    def __init__(self):
        super().__init__(input_ids=torch.zeros(2, 4))

    def word_ids(self, k):
        return [None, 0, 1, None] if k == 0 else [None, 1, 1, 2]


def test_merge_keeps_higher_confidence_window():
    probs = torch.tensor([[[1, 0], [0.9, 0.1], [0.6, 0.4], [1, 0]],
                          [[1, 0], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7]]])
    assert transformer.merge(Enc(), probs, 3, {0: "O", 1: "TOTAL"}) == ["O", "TOTAL", "TOTAL"]


def test_project_labels_by_word_centre():
    from train_transformer import project
    gold = [dict(W("TOTAL", 50, 340, 100), label="O"), dict(W("5.87", 400, 340, 100), label="TOTAL")]
    ocr = [W("TOT", 60, 342, 40), W("5.87", 410, 341, 60), W("THANKS", 200, 500)]
    assert project(ocr, gold) == ["O", "TOTAL", "O"]
