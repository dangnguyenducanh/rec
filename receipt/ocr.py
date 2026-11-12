"""Stage 2: PaddleOCR / Tesseract wrappers + OCR cache read/write. All boxes in original-image coords."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from receipt import preprocess

ROOT = Path(__file__).resolve().parent.parent
TESSERACT = ROOT / ".tools/tess/bin/tesseract"
PADDLE_THREADS = 8   # per process; build_ocr_cache lowers it for its workers
_paddle = None


def _get_paddle():
    global _paddle
    if _paddle is None:
        import paddle
        from paddleocr import PaddleOCR
        # GPU when this env's Paddle has CUDA (.venv-ocr, used for the caches); CPU in .venv, whose
        # torch pins newer cuDNN/NCCL than paddlepaddle-gpu allows.
        device = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
        # PP-OCRv6 medium (lang="en"): on the sample receipt it reads every word incl. spaces at
        # ~5 s/image; PP-OCRv5 server drops spaces ("GREENVALLEYMARKET") and is ~1.5x slower.
        _paddle = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                            use_textline_orientation=False, enable_mkldnn=False, device=device,
                            cpu_threads=PADDLE_THREADS)
    return _paddle


def _paddle_words(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    r = _get_paddle().predict(img)[0]
    out = []
    for text, score, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"]):
        tl, tr, br, bl = np.asarray(poly, dtype=np.float64)[:4]
        n = len(text)
        for m in re.finditer(r"\S+", text):  # split the line; boxes interpolated by char offset
            f0, f1 = m.start() / n, m.end() / n
            q = [tl + f0 * (tr - tl), tl + f1 * (tr - tl), bl + f1 * (br - bl), bl + f0 * (br - bl)]
            out.append({"text": m.group(), "quad": np.ravel(q), "conf": float(score)})
    return out


def _tesseract_words(img):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT)
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    out = []
    for text, conf, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
        if text.strip() and float(conf) >= 0:
            out.append({"text": text.strip(), "quad": np.array([x, y, x + w, y, x + w, y + h, x, y + h], float),
                        "conf": float(conf) / 100})
    return out


def run(img_bgr, engine="paddle", steps=preprocess.STEPS_DEFAULT) -> tuple[list[dict], np.ndarray]:
    """(words in original coords, M original->processed)."""
    # Step 2 only shrinks for Paddle: its recognizer resizes line crops to 48 px anyway, and on 60 test
    # receipts upscaling to 32 px text cost recall (CORD 0.961 -> 0.947) at 2x the time. Tesseract
    # needs the upscale (WildReceipt recall 0.58 without, 0.66 with).
    img, M = preprocess.run(img_bgr, steps, max_upscale=1.0 if engine == "paddle" else preprocess.MAX_UPSCALE)
    words = _paddle_words(img) if engine == "paddle" else _tesseract_words(img)
    Minv = np.linalg.inv(M)
    for w in words:
        pts = cv2.perspectiveTransform(w["quad"].reshape(-1, 1, 2), Minv).ravel()
        w["quad"] = [round(float(v), 1) for v in pts]
    return words, M


def ocr(img_bgr, engine="paddle", steps=preprocess.STEPS_DEFAULT) -> list[dict]:
    return run(img_bgr, engine, steps)[0]


def steps_tag(steps) -> str:
    """(2,3,4,5) / "2,3,4,5" / "2345" -> "2345"; () / "none" -> "none"."""
    s = steps if isinstance(steps, str) else ",".join(map(str, steps or ()))
    return "".join(sorted(set(re.findall(r"\d", s)))) or "none"


def cache_path(engine: str, steps, dataset: str, split: str) -> Path:
    return ROOT / "data/ocr_cache" / f"{engine}-s{steps_tag(steps)}" / dataset / f"{split}.jsonl"


def load_cache(engine, steps, dataset, split) -> dict[str, dict]:
    with open(cache_path(engine, steps, dataset, split)) as f:
        return {r["id"]: r for r in map(json.loads, f)}
