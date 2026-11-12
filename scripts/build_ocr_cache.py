"""Run OCR once over every split -> data/ocr_cache/{engine}-s{steps}/{dataset}/{split}.jsonl.

.venv/bin/python -s scripts/build_ocr_cache.py --engine paddle --steps 2,3,4,5 \
    --datasets wildreceipt,sroie,cord --splits test --workers 8 [--limit N]
A finished split is skipped. --limit N writes {split}.limitN.jsonl so it never passes for complete.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from receipt import ocr  # noqa: E402

_cfg = {}
WORKER_MIB = 4000  # one Paddle GPU worker's peak on a large photo, with its cache emptied after each image
MARGIN_MIB = 3000  # left free for the card's other users, whose usage swings
TRAIN_RESERVE_MIB = 12000  # kept free while a transformer is training or waiting to start
MAX_GPU_WORKERS = 4  # ~4 photos/s already; more workers only take VRAM from training


def gpu_workers(requested: int) -> int:
    """As many GPU workers as free VRAM holds (utilization is ignored: the card is shared), leaving room
    for a transformer training job so OCR never starves it."""
    import subprocess
    free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])
    training = subprocess.run(["pgrep", "-f", "train_transformer.py"], capture_output=True).returncode == 0
    room = free - MARGIN_MIB - (TRAIN_RESERVE_MIB if training else 0)
    return max(1, min(requested, MAX_GPU_WORKERS, room // WORKER_MIB))


def _init(engine, steps, threads, lock):
    if engine == "tesseract":  # never for Paddle: a 1-thread OMP limit under cpu_threads>1 deadlocks it
        os.environ["OMP_THREAD_LIMIT"] = "1"
    cv2.setNumThreads(1)
    ocr.PADDLE_THREADS = threads
    _cfg.update(engine=engine, steps=steps)
    if engine == "paddle":  # load + warm up so the first record's ocr_s is not model start-up
        with lock:  # workers loading models at the same moment can hang; one at a time costs ~10 s each
            ocr.run(np.full((64, 256, 3), 255, np.uint8), engine, ())


def _one(job):
    rid, image = job
    try:
        img = cv2.imread(str(ROOT / image))
        if img is None:
            raise ValueError(f"cannot read {image}")
        t = time.perf_counter()
        words, M = ocr.run(img, _cfg["engine"], _cfg["steps"])
        dt = round(time.perf_counter() - t, 3)
        if _cfg["engine"] == "paddle":  # Paddle keeps every block it allocated; hand them back per image
            import paddle
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        return {"id": rid, "words": words, "ocr_s": dt, "M": M.tolist()}
    except Exception as e:  # one bad image must not kill an hour-long run
        return {"id": rid, "words": [], "ocr_s": 0.0, "M": np.eye(3).tolist(), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["paddle", "tesseract"], default="paddle")
    ap.add_argument("--steps", default="2,3,4,5", help="e.g. 2,3,4,5 or none")
    ap.add_argument("--datasets", default="wildreceipt,sroie,cord")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=3, help="Paddle CPU threads per worker")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repair", action="store_true", help="only re-run failed records of existing caches")
    a = ap.parse_args()
    steps = tuple(int(s) for s in a.steps.split(",") if s.isdigit())

    if a.engine == "paddle":
        import paddle
        gpu_python = ROOT / ".venv-ocr/bin/python"
        if not paddle.device.is_compiled_with_cuda() and gpu_python.exists():
            # ponytail: re-exec under the GPU env (10x faster than CPU) so every caller gets it for free
            os.execv(gpu_python, [str(gpu_python), "-s", *sys.argv])
        if paddle.device.is_compiled_with_cuda():
            a.workers = gpu_workers(a.workers)
            print(f"GPU OCR with {a.workers} workers (sized to free VRAM)", flush=True)
        ocr._get_paddle()  # downloads the model once, before workers race for it

    if a.repair:
        return repair(a.engine, steps, a.datasets, a.splits, a.workers, a.threads)
    ctx = mp.get_context("spawn")  # Paddle predictors are not fork-safe
    with ctx.Pool(a.workers, _init, (a.engine, steps, a.threads, ctx.Lock())) as pool:
        for ds in a.datasets.split(","):
            for split in a.splits.split(","):
                gold = ROOT / "data/gold" / ds / f"{split}.jsonl"
                if not gold.exists():
                    print(f"skip {ds}/{split}: no {gold}", flush=True)
                    continue
                out = ocr.cache_path(a.engine, steps, ds, split)
                if a.limit:
                    out = out.with_name(f"{split}.limit{a.limit}.jsonl")
                if out.exists():
                    print(f"skip {ds}/{split}: {out} exists", flush=True)
                    continue
                jobs = [(r["id"], r["image"]) for r in map(json.loads, gold.open())][: a.limit]
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".jsonl.tmp")
                t0, secs, errs = time.time(), [], 0
                with tmp.open("w") as f:
                    for i, rec in enumerate(pool.imap(_one, jobs, chunksize=1), 1):
                        f.write(json.dumps(rec) + "\n")
                        secs.append(rec["ocr_s"])
                        errs += "error" in rec
                        if i % 25 == 0 or i == len(jobs):
                            el = time.time() - t0
                            print(f"{ds}/{split} {i}/{len(jobs)} wall {el:.0f}s eta {el / i * (len(jobs) - i):.0f}s "
                                  f"median ocr_s {np.median(secs):.2f} errors {errs}", flush=True)
                tmp.rename(out)
                print(f"wrote {out}", flush=True)
                if errs:
                    _retry_errors(pool, out, jobs)
        _shutdown(pool)


def _shutdown(pool) -> None:
    """End the workers the gentle way. Leaving the `with` block calls Pool.terminate(), whose SIGTERM lands
    in Paddle's own fatal-signal handler, which can hang a worker forever (a finished run that never exits)."""
    pool.close()
    pool.join()


def _retry_errors(pool, out: Path, jobs, rounds: int = 3) -> int:
    """Re-run the records that failed (a full shared GPU makes Paddle run out of memory) and rewrite the
    cache in place; returns how many still fail."""
    image_of = dict(jobs)
    recs = [json.loads(line) for line in out.open()]
    for attempt in range(1, rounds + 1):
        bad = [i for i, r in enumerate(recs) if "error" in r and r["id"] in image_of]
        if not bad:
            break
        if attempt > 1:
            time.sleep(30)  # let the other users' memory settle before trying again
        for i, rec in zip(bad, pool.imap(_one, [(recs[i]["id"], image_of[recs[i]["id"]]) for i in bad], chunksize=1)):
            recs[i] = rec
        left = sum("error" in r for r in recs)
        print(f"retry {attempt}: {len(bad)} failed records re-run, {left} still failing", flush=True)
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    tmp.rename(out)
    return sum("error" in r for r in recs)


def repair(engine: str, steps, datasets: str, splits: str, workers: int, threads: int) -> None:
    """--repair: retry only the failed records of caches that already exist."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, _init, (engine, steps, threads, ctx.Lock())) as pool:
        for ds in datasets.split(","):
            for split in splits.split(","):
                out = ocr.cache_path(engine, steps, ds, split)
                gold = ROOT / "data/gold" / ds / f"{split}.jsonl"
                if out.exists() and gold.exists():
                    jobs = [(r["id"], r["image"]) for r in map(json.loads, gold.open())]
                    print(f"{ds}/{split}: {_retry_errors(pool, out, jobs)} failed records left", flush=True)
        _shutdown(pool)


if __name__ == "__main__":
    main()
