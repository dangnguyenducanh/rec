"""Stage 4c/4d: RoBERTa / LayoutLMv3 token-classification inference (spec §4).
Words go in reading order; >512 tokens -> overflow windows (stride 128); a word's label comes from its
first sub-token, and a word seen in two windows keeps the higher-confidence prediction."""
import os
from pathlib import Path

import torch
from PIL import Image

from receipt import rows

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MAX_LEN, STRIDE = 512, 128


def reading_order(words: list[dict]) -> list[int]:
    quads = [w["quad"] for w in words]
    return rows.order(quads, rows.group(quads)) if words else []


def boxes_1000(quads: list[list[float]], width: float, height: float) -> list[list[int]]:
    """Quads -> axis-aligned [x0, y0, x1, y1] scaled to 0-1000 (LayoutLMv3 input)."""
    c = lambda v: max(0, min(1000, int(round(v))))  # noqa: E731
    return [[c(min(q[0::2]) / width * 1000), c(min(q[1::2]) / height * 1000),
             c(max(q[0::2]) / width * 1000), c(max(q[1::2]) / height * 1000)] for q in quads]


def as_image(img) -> Image.Image:
    if img is None:
        return Image.new("RGB", (224, 224), "white")  # ponytail: no image -> blank page, layout+text only
    return (img if isinstance(img, Image.Image) else Image.open(img)).convert("RGB")


def encode(proc, kind: str, texts: list[str], boxes: list[list[int]], image=None):
    """One doc -> BatchEncoding of N overflow windows (pixel_values stacked to N for LayoutLMv3)."""
    kw = dict(truncation=True, max_length=MAX_LEN, stride=STRIDE, return_overflowing_tokens=True,
              padding="max_length", return_tensors="pt")
    if kind == "layoutlmv3":
        enc = proc(images=image, text=texts, boxes=boxes, **kw)
        pv = enc["pixel_values"]
        enc["pixel_values"] = torch.stack([torch.as_tensor(p) for p in pv]) if isinstance(pv, list) else pv
    else:
        enc = proc(texts, is_split_into_words=True, **kw)
    return enc


def model_inputs(enc, device) -> dict:
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "bbox", "pixel_values")}


def first_subtokens(enc):
    """(window, token position, word index) for the first sub-token of each word in each window."""
    for k in range(len(enc["input_ids"])):
        prev = None
        for t, wid in enumerate(enc.word_ids(k)):
            if wid is not None and wid != prev:
                yield k, t, wid
            prev = wid


def merge(enc, probs: torch.Tensor, n_words: int, id2label: dict) -> list[str]:
    """probs [windows, tokens, labels] -> one label per word; the higher-confidence window wins."""
    conf, pred = probs.max(-1)
    best = [(-1.0, "O")] * n_words
    for k, t, wid in first_subtokens(enc):
        c = float(conf[k, t])
        if c > best[wid][0]:
            best[wid] = (c, id2label[int(pred[k, t])])
    return [lab for _, lab in best]


def predictor(model, proc, kind: str):
    device = next(model.parameters()).device
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    @torch.no_grad()
    def extract(doc: dict) -> list[str]:
        words = doc["words"]
        if not words:
            return []
        order = reading_order(words)
        texts = [words[i]["text"] for i in order]
        boxes = boxes_1000([words[i]["quad"] for i in order], doc["width"], doc["height"])
        enc = encode(proc, kind, texts, boxes, as_image(doc.get("image")) if kind == "layoutlmv3" else None)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**model_inputs(enc, device)).logits
        labs = merge(enc, logits.float().softmax(-1).cpu(), len(texts), id2label)
        out = ["O"] * len(words)
        for i, lab in zip(order, labs):
            out[i] = lab
        return out
    return extract


def load_processor(path, kind: str):
    from transformers import AutoProcessor, AutoTokenizer
    if kind == "layoutlmv3":
        return AutoProcessor.from_pretrained(path, apply_ocr=False)
    return AutoTokenizer.from_pretrained(path, add_prefix_space=True)


def load(path, device: str | None = None):
    from transformers import AutoConfig, AutoModelForTokenClassification
    path = Path(path)
    kind = AutoConfig.from_pretrained(path).model_type  # "roberta" | "layoutlmv3"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
    return predictor(model, load_processor(path, kind), kind)
