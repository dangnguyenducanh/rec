"""Train roberta | layoutlmv3, phase A (gold words) | B (paddle-s2345 words, labels projected from gold).

.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase A [--max-steps N] [--out DIR]
.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase B --datasets wildreceipt,cord,sroie \
    [--gold-copies] --out models/layoutlmv3-B2
Training data: the train splits of --datasets. SROIE has document-level gold only, so its words get weak labels
(fuzzy span match of merchant/address, parsed date, total amount); every 10th SROIE train receipt is held out as val.
Early stopping on the mean of four val F1s, the serving criterion: WildReceipt field + item, SROIE field, CORD item.
Best checkpoint -> models/{model}-{phase}/. Phase B starts from models/{model}-A/.
"""
import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark import doc_of, gold_words, load_gold, run  # noqa: E402
from receipt import ocr, rows  # noqa: E402
from receipt.assemble import norm_date, norm_money, norm_text  # noqa: E402
from receipt.extractors import transformer as T  # noqa: E402
from receipt.schema import LABELS  # noqa: E402
from receipt.score import score  # noqa: E402

cv2.setNumThreads(1)  # dataloader workers: keep CPU use to ~4 cores

BASE = {"roberta": "FacebookAI/roberta-base", "layoutlmv3": "microsoft/layoutlmv3-base"}
LR = {"roberta": 5e-5, "layoutlmv3": 3e-5}
MIN_FREE_GB = 9
MAX_SIDE = 1000  # images are shrunk before augmentation; the processor resizes to 224 anyway


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


PROJECT_TOL = 0.1  # x word height; 0.05-0.15 all help, 0.25+ starts mislabelling WildReceipt header fields


def project(ocr_words: list[dict], gold: list[dict]) -> list[str]:
    """Label of the gold quad containing the OCR word's centre. Loose gold boxes overlap: when several contain
    it, the one the word overlaps best (IoU) wins. When none does, the nearest quad within PROJECT_TOL word
    heights (narrow words such as a qty '2' in an offset box); else O."""
    polys = [np.asarray(g["quad"], np.float32).reshape(4, 2) for g in gold]
    out = []
    for w in ocr_words:
        q = w["quad"]
        c = (float(sum(q[0::2]) / 4), float(sum(q[1::2]) / 4))
        d = [cv2.pointPolygonTest(p, c, True) for p in polys]  # signed distance, >= 0 inside
        hits = [k for k, v in enumerate(d) if v >= 0]
        if len(hits) > 1:
            wp = np.asarray(q, np.float32).reshape(4, 2)
            k = max(hits, key=lambda k: _iou(wp, polys[k]))
        elif hits:
            k = hits[0]
        else:
            k = max(range(len(d)), key=d.__getitem__, default=None)
            if k is not None and d[k] < -PROJECT_TOL * math.dist(q[:2], q[6:]):
                k = None
        out.append(gold[k]["label"] or "O" if k is not None else "O")
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = cv2.convexHull(a), cv2.convexHull(b)
    inter = cv2.intersectConvexConvex(a, b)[0]
    return inter / (cv2.contourArea(a) + cv2.contourArea(b) - inter + 1e-9)


SROIE_VAL_EVERY = 10  # SROIE ships train/test only: every 10th train receipt (by id) is the val split
TOTAL_ROW = re.compile(r"(?<!SUB)(?<!SUB[ .-])TOTAL|AMOUNT|NETT|JUMLAH|DUE")
PAY_ROW = re.compile(r"CASH|CHANGE|PAID|TENDER")  # a payment row that happens to equal the total


def _best_span(words, order, target, starts, max_len) -> tuple[float, range]:
    """Reading-order word span whose joined text best matches target (difflib ratio, spaces ignored)."""
    best = (0.0, range(0))
    for s in starts:
        txt = ""
        for e in range(s, min(s + max_len, len(order))):
            txt += norm_text(words[order[e]]["text"]) or ""
            if len(txt) > 1.3 * len(target):
                break
            if len(txt) >= 0.7 * len(target):
                r = SequenceMatcher(None, txt, target, autojunk=False).ratio()
                if r > best[0]:
                    best = (r, range(s, e + 1))
    return best


def weak_labels(words: list[dict], gold: dict, dayfirst: bool) -> list[str]:
    """Word labels from document-level gold (SROIE): merchant / address = best fuzzy span (ratio >= 0.8),
    date = the shortest 1-3 word span that parses to the gold date, total = words worth the gold total, those
    on a TOTAL / AMOUNT row when there are any. SROIE annotates nothing else, so its item, time and tax words
    are not O: they get None (ignored in training), except on the matched fields' own rows ("TOTAL:", "DATE")."""
    labs = [None] * len(words)
    if not words:
        return labs
    quads = [w["quad"] for w in words]
    rid = rows.group(quads)
    order = rows.order(quads, rid)
    for field, lab, n_start, max_len in (("merchant", "MERCHANT", 60, 25), ("address", "ADDRESS", 100, 60)):
        target = norm_text(gold.get(field))
        if not target:
            continue
        r, span = _best_span(words, order, target, range(min(n_start, len(order))), max_len)
        if r >= 0.8:
            for k in span:
                if labs[order[k]] is None:
                    labs[order[k]] = lab
    if gold.get("date"):
        done = False
        for n in (1, 2, 3):  # shortest span first: norm_date finds a date inside longer text too
            for k in range(len(order)):
                span = order[k:k + n]
                if len(span) == n and norm_date(" ".join(words[i]["text"] for i in span), dayfirst) == gold["date"]:
                    for i in span:
                        labs[i] = "DATE"
                    done = True
                    break
            if done:
                break
    total = norm_money(gold.get("total"))
    if total:
        hits = [i for i in order if labs[i] is None and re.search(r"\d", words[i]["text"])
                and norm_money(words[i]["text"]) == total]
        row_text = defaultdict(str)
        for i in order:
            row_text[rid[i]] += " " + words[i]["text"].upper()
        keyed = [i for i in hits if TOTAL_ROW.search(row_text[rid[i]]) and not PAY_ROW.search(row_text[rid[i]])]
        for i in keyed or hits:
            labs[i] = "TOTAL"
    field_rows = {rid[i] for i, lab in enumerate(labs) if lab}
    return [lab or ("O" if rid[i] in field_rows else None) for i, lab in enumerate(labs)]


def load_split(phase: str, split: str, dataset: str = "wildreceipt") -> tuple[list[dict], dict | None]:
    """(gold records, paddle cache or None). Phase A uses gold words."""
    src = "train" if dataset == "sroie" else split
    recs = load_gold(dataset, src)
    if dataset == "sroie":
        recs = [r for i, r in enumerate(sorted(recs, key=lambda r: r["id"]))
                if (i % SROIE_VAL_EVERY == 0) == (split == "val")]
    return recs, (ocr.load_cache("paddle", "2345", dataset, src) if phase == "B" else None)


def labels_for(rec: dict, words: list[dict], gold: bool) -> list[str | None]:
    """One label per word; None = unannotated (SROIE), ignored by the loss."""
    if rec["dataset"] == "sroie":
        return weak_labels(words, rec["gold"], rec["dayfirst"])
    return [w["label"] or "O" for w in rec["words"]] if gold else project(words, rec["words"])


def examples(recs, cache, gold_copies: bool = False) -> list[tuple[dict, list[str]]]:
    """Phase A: gold words. Phase B: Paddle words, plus the gold words too when gold_copies."""
    out = []
    for r in recs:
        variants = []
        if cache is None or (gold_copies and r["dataset"] != "sroie"):  # SROIE gold text is cut at commas
            variants.append((gold_words(r), True))
        if cache is not None and r["id"] in cache:
            variants.append((cache[r["id"]]["words"], False))
        for words, gold in variants:
            if words:
                out.append((doc_of(r, words), labels_for(r, words, gold)))
    return out


def val_score(fn, vals: dict) -> tuple[float, dict]:
    """Mean of WildReceipt field + item F1, SROIE field F1 and CORD item F1 on the val splits."""
    s = {}
    for ds, (recs, cache) in vals.items():
        sc = score([(r["pred"], r["gold"], r["rec"]) for r in run(fn, recs, cache)])
        if ds != "cord":
            s[f"{ds}_field_f1"] = sc["field_f1"]
        if ds != "sroie":
            s[f"{ds}_item_f1"] = sc["item_f1"]
    return round(statistics.mean(s.values()), 4), s


def augment(img: np.ndarray, quads: list[list[float]]):
    """Small rotation (±3°) + scale (0.9-1.1) about the centre, applied to image and quads together."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-3, 3), random.uniform(0.9, 1.1))
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    pts = np.asarray(quads, np.float64).reshape(-1, 4, 2)
    pts = pts @ M[:, :2].T + M[:, 2]
    return img, pts.reshape(-1, 8).tolist()


class Collate:
    def __init__(self, proc, kind, aug):
        self.proc, self.kind, self.aug = proc, kind, aug

    def __call__(self, batch):
        encs, labs = [], []
        for doc, labels in batch:
            order = T.reading_order(doc["words"])
            texts = [doc["words"][i]["text"] for i in order]
            quads = [doc["words"][i]["quad"] for i in order]
            W, H, image = doc["width"], doc["height"], None
            if self.kind == "layoutlmv3":
                img = cv2.imread(doc["image"])
                s = min(1.0, MAX_SIDE / max(img.shape[:2]))
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
                sx, sy = img.shape[1] / W, img.shape[0] / H
                quads = [[v * (sx if j % 2 == 0 else sy) for j, v in enumerate(q)] for q in quads]
                W, H = img.shape[1], img.shape[0]
                if self.aug and random.random() < 0.5:
                    img, quads = augment(img, quads)
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            enc = T.encode(self.proc, self.kind, texts, T.boxes_1000(quads, W, H), image)
            ids = [LABELS.index(labels[i]) if labels[i] else -100 for i in order]
            lab = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
            for k, t, wid in T.first_subtokens(enc):
                lab[k, t] = ids[wid]
            encs.append(enc)
            labs.append(lab)
        keys = ["input_ids", "attention_mask"] + (["bbox", "pixel_values"] if self.kind == "layoutlmv3" else [])
        out = {k: torch.cat([e[k] for e in encs]) for k in keys}
        out["labels"] = torch.cat(labs)
        return out


def wait_for_gpu():
    while (free := torch.cuda.mem_get_info()[0] / 2**30) < MIN_FREE_GB:
        log(f"GPU free {free:.1f} GB < {MIN_FREE_GB} GB, waiting 60 s")
        time.sleep(60)
    log(f"GPU free {free:.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["roberta", "layoutlmv3"])
    ap.add_argument("--phase", required=True, choices=["A", "B"])
    ap.add_argument("--max-steps", type=int, help="smoke run: stop after N optimizer steps")
    ap.add_argument("--out", help="default models/{model}-{phase}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--datasets", default="wildreceipt", help="train splits to train on, e.g. wildreceipt,cord,sroie")
    ap.add_argument("--gold-copies", action="store_true", help="phase B: also train on the gold words")
    a = ap.parse_args()
    from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

    out = Path(a.out) if a.out else ROOT / f"models/{a.model}-{a.phase}"
    init = BASE[a.model] if a.phase == "A" else str(ROOT / f"models/{a.model}-A")
    train = [ex for ds in a.datasets.split(",") for ex in examples(*load_split(a.phase, "train", ds), a.gold_copies)]
    vals = {ds: load_split(a.phase, "val", ds) for ds in ("wildreceipt", "sroie", "cord")}
    log(f"{a.model}-{a.phase}: {len(train)} train docs ({a.datasets}{', + gold words' if a.gold_copies else ''}), "
        f"val {[len(v[0]) for v in vals.values()]}, init {init} -> {out}")

    wait_for_gpu()
    torch.manual_seed(0)
    random.seed(0)
    proc = T.load_processor(init, a.model)
    model = AutoModelForTokenClassification.from_pretrained(
        init, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    ).cuda()
    dl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=4,
                                     collate_fn=Collate(proc, a.model, aug=a.model == "layoutlmv3"))
    steps_per_epoch = math.ceil(len(dl) / a.accum)
    total = a.max_steps or a.epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=LR[a.model], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler()

    best, bad, step, hist, t0 = -1.0, 0, 0, [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        losses = []
        for b, batch in enumerate(dl, 1):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / a.accum
            scaler.scale(loss).backward()
            losses.append(loss.item() * a.accum)
            if b % a.accum == 0 or b == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 50 == 0 or step == a.max_steps:
                    log(f"epoch {epoch} step {step}/{total} loss {np.mean(losses[-50:]):.4f} "
                        f"{(time.time() - t0) / step:.3f} s/step")
                if step == a.max_steps:
                    break
        model.eval()
        te = time.time()
        f1, parts = val_score(T.predictor(model, proc, a.model), vals)
        hist.append({"epoch": epoch, "step": step, "loss": round(float(np.mean(losses)), 4), "val_mean_f1": f1,
                     **parts, "eval_s": round(time.time() - te, 1), "elapsed_s": round(time.time() - t0, 1)})
        log(json.dumps(hist[-1]))
        if f1 > best:
            best, bad = f1, 0
            model.save_pretrained(out)
            proc.save_pretrained(out)
            log(f"saved best (val mean F1 {f1}) -> {out}")
        else:
            bad += 1
        (out / "train_log.json").write_text(json.dumps({"best_val_mean_f1": best, "datasets": a.datasets,
                                                        "gold_copies": a.gold_copies, "history": hist,
                                                        "s_per_step": round((time.time() - t0) / max(step, 1), 3)}, indent=1))
        if step == a.max_steps or bad >= a.patience:
            break
    log(f"done: best val mean F1 {best}, {step} steps, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
