"""Stage 3: group words into rows by vertical overlap (interfaces §2)."""
import math
import statistics

TAU = 0.45  # best pairwise same-row F1 on CORD val gold boxes (0.986); eval/diagnostics/rows.json
XOV = 0.3  # max x-overlap (fraction of the narrower word) with a row member; 0.2-0.5 score the same


def _deskew(quads: list[list[float]]) -> list[tuple[float, float, float, float]]:
    """Undo the median top-edge angle; return (x0, y0, x1, y1) per word in the rotated frame."""
    # ponytail: median angle breaks on upside-down receipts (angles wrap at ±180°)
    a = statistics.median(math.atan2(q[3] - q[1], q[2] - q[0]) for q in quads) if quads else 0.0
    c, s = math.cos(a), math.sin(a)
    boxes = []
    for q in quads:
        xs = [q[i] * c + q[i + 1] * s for i in (0, 2, 4, 6)]
        ys = [-q[i] * s + q[i + 1] * c for i in (0, 2, 4, 6)]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def group(quads: list[list[float]], tau: float = TAU) -> list[int]:
    """Row id per word, rows numbered top to bottom."""
    boxes = _deskew(quads)
    rows = []  # [sum of tops, sum of bottoms, member indices]
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i][1] + boxes[i][3]):
        x0, y0, x1, y1 = boxes[i]
        best, best_ov = None, 0.0
        for r in rows:
            # two words overlapping in x are on different lines (Paddle boxes can be taller than the pitch)
            if any(min(x1, boxes[j][2]) - max(x0, boxes[j][0]) > XOV * min(x1 - x0, boxes[j][2] - boxes[j][0])
                   for j in r[2]):
                continue
            # judge against the member nearest in x, so a curved or skewed line does not drift
            j = min(r[2], key=lambda j: max(boxes[j][0] - x1, x0 - boxes[j][2], 0))
            top, bot = boxes[j][1], boxes[j][3]
            ov = min(y1, bot) - max(y0, top)
            if ov > tau * min(y1 - y0, bot - top) and ov > best_ov:
                best, best_ov = r, ov
        if best is None:
            rows.append([y0, y1, [i]])
        else:
            best[0] += y0
            best[1] += y1
            best[2].append(i)
    rows.sort(key=lambda r: (r[0] + r[1]) / len(r[2]))
    ids = [0] * len(quads)
    for rid, r in enumerate(rows):
        for i in r[2]:
            ids[i] = rid
    return ids


def order(quads: list[list[float]], row_ids: list[int]) -> list[int]:
    """Word indices in reading order: row, then left x (in the deskewed frame)."""
    boxes = _deskew(quads)
    return sorted(range(len(quads)), key=lambda i: (row_ids[i], boxes[i][0]))
