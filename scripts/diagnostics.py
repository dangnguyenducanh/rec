"""Stage 1-3 diagnostics (spec §4) -> eval/diagnostics/.

.venv/bin/python -s scripts/diagnostics.py blur      # blur gate on SROIE train blurred on purpose + WildReceipt train
.venv/bin/python -s scripts/diagnostics.py warp      # step-3 quad IoU vs CORD test roi
.venv/bin/python -s scripts/diagnostics.py deskew    # step-4 angle error on rotated SROIE test scans
.venv/bin/python -s scripts/diagnostics.py rows [--ocr paddle]   # tau on CORD val, pairwise P/R on CORD test
.venv/bin/python -s scripts/diagnostics.py ocr --engine paddle --steps 2,3,4,5 [--datasets ..] [--limit N]
    detection H-mean @ IoU 0.5 + CER per test set from a finished cache -> eval/diagnostics/ocr.csv
    (--limit N reads the {split}.limitN.jsonl smoke cache and only prints)
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from receipt import ocr, preprocess, rows  # noqa: E402

OUT = ROOT / "eval/diagnostics"
WORKERS = 8
SIGMAS = (0, 1, 2, 3, 5, 8)
ANGLES = (1, 2, 3, 5, 8, 12, 15)
GAP = 1.0  # detection: same-row words closer than GAP x word height merge into one phrase


def gold(ds, split):
    with open(ROOT / "data/gold" / ds / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def pool():
    return mp.Pool(WORKERS, cv2.setNumThreads, (1,))


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1) + "\n")
    print(json.dumps({k: v for k, v in obj.items() if k != "per_image"}, indent=1))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}


# ---------------------------------------------------------------- blur
def _blur_one(path):
    """Blur at the gate's own <=1024 px scale, so sigma means the same on a 500 px and a 7000 px scan
    (blurred at full size, sigma 5 on a 7000 px scan is ~0.7 px to the gate and stays sharp)."""
    img = cv2.imread(path)
    s = min(1.0, 1024 / max(img.shape[:2]))
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    return preprocess.scores(img)[1], [preprocess.scores(cv2.GaussianBlur(img, (0, 0), sg) if sg else img)[0]
                                       for sg in SIGMAS]


def _scores_one(path):
    return preprocess.scores(cv2.imread(path))


def blur(_):
    with pool() as p:
        s = p.map(_blur_one, [str(ROOT / r["image"]) for r in gold("sroie", "train")])
        wr = np.array(p.map(_scores_one, [str(ROOT / r["image"]) for r in gold("wildreceipt", "train")]))
    sroie, sroie_glare = np.array([b for _, b in s]), np.array([g for g, _ in s])
    lo = float(sroie[:, SIGMAS.index(5):].max())  # BLUR_MIN above this rejects every sigma>=5 scan
    hi = float(min(np.quantile(sroie[:, 0], 0.04), np.quantile(wr[:, 0], 0.04)))  # at most 4% false rejects
    t = round(math.sqrt(lo * hi), 1) if lo < hi else round(hi, 1)  # geometric middle of the gap
    dump("blur.json", {
        "blur_min": t, "feasible": lo < hi, "max_score_sigma_ge5": round(lo, 2), "q04_sharp": round(hi, 2),
        "n_sroie_train": len(sroie), "n_wildreceipt_train": len(wr),
        "reject_rate_sroie_by_sigma": {str(sg): round(float((sroie[:, i] < t).mean()), 4) for i, sg in enumerate(SIGMAS)},
        "reject_rate_wildreceipt_train": round(float((wr[:, 0] < t).mean()), 4),
        "median_score_sroie_by_sigma": {str(sg): round(float(np.median(sroie[:, i])), 1) for i, sg in enumerate(SIGMAS)},
        "median_score_wildreceipt_train": round(float(np.median(wr[:, 0])), 1),
        "glare_max": preprocess.GLARE_MAX,
        "glare_reject_rate_sroie_sigma0": round(float((sroie_glare > preprocess.GLARE_MAX).mean()), 4),
        "glare_reject_rate_wildreceipt_train": round(float((wr[:, 1] > preprocess.GLARE_MAX).mean()), 4),
    })


# ---------------------------------------------------------------- warp
def poly_iou(a, b):
    a, b = (cv2.convexHull(np.float32(q).reshape(-1, 2)) for q in (a, b))
    inter = cv2.intersectConvexConvex(a, b)[0]
    return float(inter / (cv2.contourArea(a) + cv2.contourArea(b) - inter))


def _warp_one(rec):
    g = cv2.imread(str(ROOT / rec["image"]), cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    full = [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1]
    q = preprocess._find_quad(g)
    r = np.float32(rec["roi"]).reshape(4, 2)
    pad = 0.01 * max(h, w)  # same border margin as _find_quad: a corner outside it cannot be found
    inside = bool(((r > pad) & (r < [w - 1 - pad, h - 1 - pad])).all())
    return {"id": rec["id"], "found": q is not None, "roi_inside_frame": inside, "iou": round(poly_iou(full if q is None else q, rec["roi"]), 4),
            "iou_full_image": round(poly_iou(full, rec["roi"]), 4)}


def warp(_):
    recs = [r for r in gold("cord", "test") if r["roi"]]
    with pool() as p:
        res = p.map(_warp_one, recs)
    found = [r["iou"] for r in res if r["found"]]
    every = [r["iou"] for r in res]
    dump("warp.json", {
        "n_with_roi": len(res), "quad_found": len(found),
        "roi_inside_frame": sum(r["roi_inside_frame"] for r in res),
        "quad_found_when_roi_inside": sum(r["found"] and r["roi_inside_frame"] for r in res),
        "iou_mean_found": round(float(np.mean(found)), 4) if found else None,
        "iou_median_found": round(float(np.median(found)), 4) if found else None,
        "share_found_iou_ge_0.85": round(float(np.mean(np.array(found) >= 0.85)), 4) if found else None,
        "iou_mean_all": round(float(np.mean(every)), 4),  # no quad -> the pipeline keeps the full image
        "iou_mean_no_warp": round(float(np.mean([r["iou_full_image"] for r in res])), 4),
        "per_image": res,
    })


# ---------------------------------------------------------------- deskew
def _applied_angle(img):
    """Degrees step 4 rotates by on the Paddle path (steps 2,4), read from M = R @ S."""
    M = preprocess.run(img, (2, 4), max_upscale=1.0)[1]
    return math.degrees(math.atan2(M[0, 1], M[0, 0]))


def _deskew_one(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    base = _applied_angle(img)  # the scan's own skew; truth after rotating by a is base - a
    out = []
    for a in (*ANGLES, *(-x for x in ANGLES)):
        R = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        c, s = abs(R[0, 0]), abs(R[0, 1])
        nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
        R[0, 2] += nw / 2 - w / 2
        R[1, 2] += nh / 2 - h / 2
        rot = cv2.warpAffine(img, R, (nw, nh), borderValue=(255, 255, 255))
        out.append((a, _applied_angle(rot), base))
    return out


def deskew(_):
    with pool() as p:
        res = [t for r in p.map(_deskew_one, [str(ROOT / r["image"]) for r in gold("sroie", "test")]) for t in r]
    a, est, base = np.array(res).T
    err, raw = np.abs(est - (base - a)), np.abs(est + a)
    dump("deskew.json", {
        "n_images": len(res) // (2 * len(ANGLES)), "n_trials": len(res),
        "median_abs_err_deg": round(float(np.median(err)), 3),  # vs applied angle + the scan's own skew
        "p90_abs_err_deg": round(float(np.quantile(err, 0.9)), 3),
        "median_abs_err_vs_applied_only_deg": round(float(np.median(raw)), 3),
        "share_not_rotated": round(float(np.mean(est == 0)), 4),
        "median_abs_err_by_angle": {str(x): round(float(np.median(err[np.abs(a) == x])), 3) for x in ANGLES},
    })


# ---------------------------------------------------------------- rows
def pair_counts(pred, truth):
    """(tp, fp, fn) over word pairs i<j for 'same row'."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    iu = np.triu_indices(len(pred), 1)
    P, T = (pred[:, None] == pred[None, :])[iu], (truth[:, None] == truth[None, :])[iu]
    return int((P & T).sum()), int((P & ~T).sum()), int((~P & T).sum())


def _rows_score(recs, tau):
    tp = fp = fn = 0
    for quads, truth in recs:
        c = pair_counts(rows.group(quads, tau), truth)
        tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
    return prf(tp, fp, fn)


def _gold_rows(split):
    return [([w["quad"] for w in r["words"]], [w["row"] for w in r["words"]]) for r in gold("cord", split)]


def rows_diag(a):
    val = _gold_rows("val")
    grid = {round(t, 2): _rows_score(val, round(t, 2)) for t in np.arange(0.05, 0.951, 0.05)}
    best = max(grid, key=lambda t: grid[t]["f1"])
    res = {"tau_best_val": best, "tau_in_use": rows.TAU, "val_f1_by_tau": {str(t): v["f1"] for t, v in grid.items()},
           "cord_test_gold": _rows_score(_gold_rows("test"), rows.TAU)}
    if a.ocr:
        cache = ocr.load_cache(a.ocr, a.steps, "cord", "test")
        recs, kept, total = [], 0, 0
        for r in gold("cord", "test"):
            words = cache[r["id"]]["words"]
            gq = [np.float32(w["quad"]).reshape(-1, 1, 2) for w in r["words"]]
            truth = []
            for w in words:  # row truth = row of the gold box this word's centre falls in
                c = (float(np.mean(w["quad"][0::2])), float(np.mean(w["quad"][1::2])))
                hit = next((k for k, q in enumerate(gq) if cv2.pointPolygonTest(q, c, False) >= 0), None)
                truth.append(None if hit is None else r["words"][hit]["row"])
            ids = rows.group([w["quad"] for w in words]) if words else []  # grouped on all words, like the pipeline
            keep = [i for i, t in enumerate(truth) if t is not None]
            kept, total = kept + len(keep), total + len(words)
            recs.append(([ids[i] for i in keep], [truth[i] for i in keep]))
        tp = fp = fn = 0
        for pred, truth in recs:
            c = pair_counts(pred, truth)
            tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
        res[f"cord_test_{a.ocr}_s{ocr.steps_tag(a.steps)}"] = {**prf(tp, fp, fn), "words_in_a_gold_box": kept,
                                                               "words": total}
    dump("rows.json", res)


# ---------------------------------------------------------------- ocr
def phrases(boxes, row_ids):
    """Merge same-row neighbours closer than GAP x height: gold units differ by dataset (CORD words,
    SROIE lines, WildReceipt segments without spaces) and OCR emits tokens, so both sides are compared
    as phrases. ponytail: fixed gap rule; a column gap narrower than GAP x height merges two phrases."""
    out = []
    for i in sorted(range(len(boxes)), key=lambda i: (row_ids[i], boxes[i][0])):
        x0, y0, x1, y1 = boxes[i]
        if out and out[-1][4] == row_ids[i] and x0 - out[-1][2] <= GAP * (y1 - y0):
            o = out[-1]
            out[-1] = [min(o[0], x0), min(o[1], y0), max(o[2], x1), max(o[3], y1), o[4]]
        else:
            out.append([x0, y0, x1, y1, row_ids[i]])
    return np.array(out, float).reshape(-1, 5)[:, :4]


def match(g, p, thr=0.5):
    """Greedy one-to-one matches with IoU >= thr between two (n, 4) box arrays."""
    if not len(g) or not len(p):
        return 0
    ix = np.clip(np.minimum(g[:, None, 2], p[None, :, 2]) - np.maximum(g[:, None, 0], p[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(g[:, None, 3], p[None, :, 3]) - np.maximum(g[:, None, 1], p[None, :, 1]), 0, None)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    inter = ix * iy
    iou = inter / (area(g)[:, None] + area(p)[None, :] - inter + 1e-9)
    gi, pi = np.nonzero(iou >= thr)
    used_g, used_p = set(), set()
    for k in np.argsort(-iou[gi, pi], kind="stable"):
        if gi[k] not in used_g and pi[k] not in used_p:
            used_g.add(gi[k])
            used_p.add(pi[k])
    return len(used_g)


def detect(gold_quads, pred_quads):
    """(matched, n_gold, n_pred) phrases, both sides axis-aligned in one shared deskewed frame."""
    boxes = rows._deskew(gold_quads + pred_quads)
    g = phrases(boxes[: len(gold_quads)], rows.group(gold_quads))
    p = phrases(boxes[len(gold_quads):], rows.group(pred_quads)) if pred_quads else np.zeros((0, 4))
    return match(g, p), len(g), len(p)


def text(words):
    """Words in reading order, uppercased, all whitespace removed (WildReceipt gold has no spaces)."""
    q = [w["quad"] for w in words]
    return "".join(re.sub(r"\s+", "", words[i]["text"]).upper() for i in rows.order(q, rows.group(q)))


def ocr_diag(a):
    tag = ocr.steps_tag(a.steps)
    for ds in a.datasets.split(","):
        path = ocr.cache_path(a.engine, a.steps, ds, "test")
        if a.limit:
            path = path.with_name(f"test.limit{a.limit}.jsonl")
        if not path.exists():
            print(f"skip {ds}: no {path}")
            continue
        with open(path) as f:
            cache = {r["id"]: r for r in map(json.loads, f)}
        tp = ng = npred = edits = chars = 0
        for r in gold(ds, "test"):
            if r["id"] not in cache:
                continue
            words = cache[r["id"]]["words"]
            m, g, p = detect([w["quad"] for w in r["words"]], [w["quad"] for w in words])
            tp, ng, npred = tp + m, ng + g, npred + p
            gt = text(r["words"])
            edits, chars = edits + Levenshtein.distance(text(words), gt), chars + len(gt)
        d = prf(tp, npred - tp, ng - tp)
        row = {"engine": a.engine, "steps": tag, "dataset": ds, "n": len(cache), "det_p": d["p"], "det_r": d["r"],
               "det_hmean": d["f1"], "cer": round(edits / chars, 4),
               "p50_ocr_s": round(float(np.median([c["ocr_s"] for c in cache.values()])), 3),
               "errors": sum("error" in c for c in cache.values())}
        print(row)
        if a.limit:
            continue
        f = OUT / "ocr.csv"  # one row per (engine, steps, dataset); a rerun replaces its row
        old = [x for x in csv.DictReader(f.open())] if f.exists() else []
        old = [x for x in old if (x["engine"], x["steps"], x["dataset"]) != (a.engine, tag, ds)]
        OUT.mkdir(parents=True, exist_ok=True)
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(row))
            w.writeheader()
            w.writerows([*old, row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["blur", "warp", "deskew", "rows", "ocr"])
    ap.add_argument("--ocr", help="rows: also score this engine's CORD test cache")
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--steps", default="2,3,4,5")
    ap.add_argument("--datasets", default="wildreceipt,sroie,cord")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    {"blur": blur, "warp": warp, "deskew": deskew, "rows": rows_diag, "ocr": ocr_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
