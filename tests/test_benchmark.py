"""Benchmark plumbing: an oracle extractor (gold labels) scores perfect field F1 and seqeval on WildReceipt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import benchmark  # noqa: E402


def test_oracle_scores_perfect():
    recs = benchmark.load_gold("wildreceipt", "val")[:20]
    gold = iter([[w["label"] for w in r["words"]] for r in recs])  # run() calls the extractor in record order
    res = benchmark.run(lambda doc: next(gold), recs)
    assert benchmark.field_f1(res) == 1.0
    assert benchmark.seqeval_f1(res)["f1"] == 1.0
