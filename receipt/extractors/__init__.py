"""Stage 4 extractors: rules, crf, transformer (roberta / layoutlmv3). Interfaces §4."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def get(name: str, model_dir: str = "models"):
    """Callable doc -> one label per word. 'roberta' / 'layoutlmv3' without a phase pick B, else A."""
    d = Path(model_dir) if Path(model_dir).is_absolute() else ROOT / model_dir
    if name == "rules":
        from receipt.extractors.rules import extract
        return extract
    if name == "crf":
        from receipt.extractors import crf
        return crf.load(d / "crf.pkl")
    model, _, phase = name.partition("-")
    if model in ("roberta", "layoutlmv3"):
        from receipt.extractors import transformer
        phase = phase or ("B" if (d / f"{model}-B").exists() else "A")
        return transformer.load(d / f"{model}-{phase}")
    raise ValueError(f"unknown extractor {name!r}")
