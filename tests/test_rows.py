"""Row grouping on a synthetic receipt rotated 4 degrees."""
import math

from receipt import rows


def _receipt(deg: float):
    """10 rows x 4 words (30 px tall, 10 px gaps, 600 px wide), rotated about (300, 200)."""
    a = math.radians(deg)
    quads, truth = [], []
    for r in range(10):
        for k, (x0, x1) in enumerate([(0, 80), (100, 260), (300, 380), (500, 600)][: 2 + r % 3]):
            y0, y1 = 40 * r, 40 * r + 30
            pts = []
            for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                dx, dy = x - 300, y - 200
                pts += [300 + dx * math.cos(a) - dy * math.sin(a), 200 + dx * math.sin(a) + dy * math.cos(a)]
            quads.append(pts)
            truth.append((r, k))
    return quads, truth


def test_group_and_order_undo_skew():
    for deg in (4, -4):
        quads, truth = _receipt(deg)
        assert rows.group(quads) == [r for r, _ in truth]
        # the words were generated in reading order
        assert rows.order(quads, rows.group(quads)) == list(range(len(quads)))


def test_skew_matters():
    """Without deskew, 4 degrees over 600 px (42 px drop) is more than a row; prove the test bites."""
    quads, truth = _receipt(4)
    flat = [[q[0], q[1], q[2], q[1], q[2], q[5], q[0], q[5]] for q in quads]  # axis-aligned, still skewed
    assert rows.group(flat) != [r for r, _ in truth]


def test_empty():
    assert rows.group([]) == [] and rows.order([], []) == []
