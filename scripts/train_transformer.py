"""Train roberta | layoutlmv3, phase A (gold words) | B (paddle-s2345 words, labels projected from gold).

.venv/bin/python -s scripts/train_transformer.py --model layoutlmv3 --phase A [--max-steps N] [--out DIR]
Early stopping on WildReceipt val field F1 (assemble + score); best checkpoint -> models/{model}-{phase}/.
Phase B starts from models/{model}-A/.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark import doc_of, field_f1, load_gold, run  # noqa: E402
from receipt import ocr  # noqa: E402
from receipt.extractors import transformer as T  # noqa: E402
from receipt.schema import LABELS  # noqa: E402

cv2.setNumThreads(1)  # dataloader workers: keep CPU use to ~4 cores

BASE = {"roberta": "FacebookAI/roberta-base", "layoutlmv3": "microsoft/layoutlmv3-base"}
LR = {"roberta": 5e-5, "layoutlmv3": 3e-5}
MIN_FREE_GB = 9
MAX_SIDE = 1000  # images are shrunk before augmentation; the processor resizes to 224 anyway


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def project(ocr_words: list[dict], gold: list[dict]) -> list[str]:
    """Label of the gold quad containing the OCR word's centre, else O."""
    polys = [np.asarray(g["quad"], np.float32).reshape(4, 2) for g in gold]
    out = []
    for w in ocr_words:
        q = w["quad"]
        c = (float(sum(q[0::2]) / 4), float(sum(q[1::2]) / 4))
        out.append(next((g["label"] or "O" for g, p in zip(gold, polys) if cv2.pointPolygonTest(p, c, False) >= 0), "O"))
    return out


def load_split(phase: str, split: str) -> tuple[list[dict], dict | None]:
    """(gold records, paddle cache or None). Phase A uses gold words."""
    recs = load_gold("wildreceipt", split)
    return recs, (ocr.load_cache("paddle", "2345", "wildreceipt", split) if phase == "B" else None)


def examples(recs, cache) -> list[tuple[dict, list[str]]]:
    out = []
    for r in recs:
        if cache is None:
            words = [{"text": w["text"], "quad": w["quad"]} for w in r["words"]]
            labs = [w["label"] or "O" for w in r["words"]]
        elif r["id"] in cache:
            words = cache[r["id"]]["words"]
            labs = project(words, r["words"])
        else:
            continue
        if words:
            out.append((doc_of(r, words), labs))
    return out


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
            ids = [LABELS.index(labels[i]) for i in order]
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
    a = ap.parse_args()
    from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

    out = Path(a.out) if a.out else ROOT / f"models/{a.model}-{a.phase}"
    init = BASE[a.model] if a.phase == "A" else str(ROOT / f"models/{a.model}-A")
    train = examples(*load_split(a.phase, "train"))
    val_recs, val_cache = load_split(a.phase, "val")
    log(f"{a.model}-{a.phase}: {len(train)} train docs, {len(val_recs)} val docs, init {init} -> {out}")

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
        f1 = field_f1(run(T.predictor(model, proc, a.model), val_recs, val_cache))
        hist.append({"epoch": epoch, "step": step, "loss": round(float(np.mean(losses)), 4), "val_field_f1": f1,
                     "eval_s": round(time.time() - te, 1), "elapsed_s": round(time.time() - t0, 1)})
        log(json.dumps(hist[-1]))
        if f1 > best:
            best, bad = f1, 0
            model.save_pretrained(out)
            proc.save_pretrained(out)
            log(f"saved best (val field F1 {f1}) -> {out}")
        else:
            bad += 1
        (out / "train_log.json").write_text(json.dumps({"best_val_field_f1": best, "history": hist,
                                                        "s_per_step": round((time.time() - t0) / max(step, 1), 3)}, indent=1))
        if step == a.max_steps or bad >= a.patience:
            break
    log(f"done: best val field F1 {best}, {step} steps, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
