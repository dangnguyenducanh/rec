"""Detection matching and pairwise row counts in scripts/diagnostics.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import diagnostics as d  # noqa: E402


def _q(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def test_detect_merges_tokens_and_counts_misses():
    gold = [_q(0, 0, 200, 20), _q(400, 0, 480, 20), _q(0, 40, 100, 60)]   # "SAN ANTONIO,TX", "9.99", "TOTAL"
    pred = [_q(0, 0, 60, 20), _q(70, 0, 200, 20), _q(400, 0, 480, 20),    # first phrase as two tokens
            _q(0, 100, 50, 120)]                                           # one false box; TOTAL missed
    assert d.detect(gold, pred) == (2, 3, 3)
    assert d.detect(gold, []) == (0, 3, 0)


def test_pair_counts():
    # truth rows: {0,1}, {2}; pred puts all three together -> 1 TP pair, 2 FP pairs, 0 FN
    assert d.pair_counts([5, 5, 5], [0, 0, 1]) == (1, 2, 0)
    assert d.pair_counts([0, 1, 2], [0, 0, 1]) == (0, 0, 1)
