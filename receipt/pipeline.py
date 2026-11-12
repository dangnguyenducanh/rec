"""extract(image_bytes) -> (Receipt, raw_ocr): spec §4 stages 1-7, interfaces §7.
RECEIPT_PIPELINE=stub returns a fixed receipt (API tests); RECEIPT_EXTRACTOR=<name> forces an extractor."""
import csv
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from receipt import assemble, categorize, ocr, preprocess, validate
from receipt.extractors import get
from receipt.schema import Item, Receipt

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["rules", "crf", "roberta-A", "layoutlmv3-A", "layoutlmv3-B"]
STEPS = "".join(map(str, preprocess.STEPS_DEFAULT))  # results.csv writes steps as digits
log = logging.getLogger(__name__)

EXTRACTOR: str | None = None  # name of the extractor in use, set by load()
_fn = None
# ponytail: one global lock, so concurrent uploads queue up; Paddle's predictor is not thread-safe.
_lock = threading.Lock()


class Rejected(Exception):
    """Quality gate failed; the message is shown to the user."""


def _stub() -> bool:
    return os.environ.get("RECEIPT_PIPELINE") == "stub"


def _available(name: str) -> bool:
    return name == "rules" or (ROOT / "models" / ("crf.pkl" if name == "crf" else f"{name}/config.json")).exists()


F1_COLUMNS = ("wr_field_f1", "sroie_field_f1", "cord_item_f1", "wr_item_f1")


def choose() -> str:
    """Available extractor with the best mean of the four F1 columns in eval/results.csv: header fields
    and line items both, since items drive the spending categories (on Paddle OCR, rules edge out
    LayoutLMv3-B on fields alone, 0.707 vs 0.698, while finding about half its items). An extractor is
    scored on its Paddle row at the default steps (what runs live) when it has one, else its gold row."""
    # ponytail: a gold-only score beside a paddle score favours the gold-only extractor; fair once every
    # candidate has a paddle-s2345 row.
    score = {}  # name -> (has paddle row, mean F1, ocr)
    with open(ROOT / "eval/results.csv") as f:
        for r in csv.DictReader(f):
            name, src = r["extractor"], r["ocr"]
            if name not in NAMES or not _available(name):
                continue
            if not (src == "gold" or (src == "paddle" and r["steps"] == STEPS)):
                continue
            s = (src == "paddle", sum(float(r[c]) for c in F1_COLUMNS) / len(F1_COLUMNS), src)
            if name not in score or s[0] >= score[name][0]:
                score[name] = s
    best = max(score, key=lambda n: score[n][1], default="rules")
    log.info("extractor %s; mean F1 (fields + items): %s", best,
             ", ".join(f"{n} {m:.3f} ({src})" for n, (_, m, src) in sorted(score.items(), key=lambda x: -x[1][1])))
    return best


def load():
    """Choose + load the extractor and warm Paddle OCR, once."""
    global EXTRACTOR, _fn
    if _stub():
        EXTRACTOR = "stub"
        return
    with _lock:
        if _fn is None:
            name = os.environ.get("RECEIPT_EXTRACTOR") or choose()
            _fn = get(name)
            # one throwaway pass: the first GPU call pays ~0.5 s of kernel setup
            img = np.full((120, 480, 3), 255, np.uint8)
            cv2.putText(img, "TOTAL 12.34", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            _fn({"words": ocr.ocr(img), "width": 480, "height": 120, "image": None})
            EXTRACTOR = name


def extract(image_bytes: bytes) -> tuple[Receipt, dict]:
    if _stub():
        load()
        return validate.check(fixture()), {"engine": "stub", "steps": [], "words": [], "M": None}
    load()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # applies EXIF rotation
    if img is None:
        raise Rejected("That file isn't an image we can read.")
    ok, reason = preprocess.quality_gate(img)
    if not ok:
        raise Rejected(reason)
    t0 = time.perf_counter()
    with _lock:
        words, M = ocr.run(img, "paddle")
        t1 = time.perf_counter()
        h, w = img.shape[:2]
        doc = {"words": words, "width": w, "height": h,
               "image": Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))}  # model sees the original
        labels = _fn(doc)
    t2 = time.perf_counter()
    r = validate.check(categorize.apply(assemble.build(words, labels)))
    s = {"ocr": t1 - t0, "extractor": t2 - t1, "total": time.perf_counter() - t0}
    log.info("%d words; %s", len(words), " ".join(f"{k} {v:.3f}s" for k, v in s.items()))
    raw = {"engine": "paddle", "steps": list(preprocess.STEPS_DEFAULT), "words": words, "M": M.tolist(),
           "seconds": {k: round(v, 3) for k, v in s.items()}}
    return r, raw


def fixture() -> Receipt:
    return Receipt(
        merchant="GREEN VALLEY MARKET", address="1420 J STREET, SACRAMENTO, CA 95814",
        date="2026-11-08", time="17:42", subtotal="20.54", tax="0.78", total="21.32",
        category="groceries",
        items=[
            Item(name="ORG BANANAS 2LB", qty="1", price="1.98", category="groceries"),
            Item(name="WHOLE MILK 1GAL", qty="1", price="3.49", category="groceries"),
            Item(name="SOURDOUGH BREAD", qty="1", price="4.29", category="groceries"),
            Item(name="PAPER TOWELS 6PK", qty="1", price="7.99", category="household"),
            Item(name="DISH SOAP", qty="1", price="2.79", category="household"),
        ],
    )
