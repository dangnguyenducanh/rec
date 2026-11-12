"""Stage 1 (spec §4): OpenCV preprocessing. run() returns (image, 3x3 M original->processed)."""
import cv2
import numpy as np

STEPS_DEFAULT = (2, 3, 4, 5)
BLUR_MIN = 18.1   # Laplacian variance (long side <= 1024 px); tuned in eval/diagnostics/blur.json
# ponytail: global share of pixels >= 250; clean SROIE scans reach 0.96, so this stays loose.
# Upgrade: count saturated blobs inside the receipt quad only.
GLARE_MAX = 0.99
TEXT_PX = 32      # step 2 target character height
MAX_SIDE = 4000   # step 2 never grows the image past this (Paddle's det limit)
MAX_UPSCALE = 4.0  # step 2 never enlarges more than this
MAX_SKEW = 20.0   # step 4 leaves larger estimated angles alone
MIN_QUAD = 0.3    # step 3 needs a quad covering at least this share of the image


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def scores(img_bgr) -> tuple[float, float]:
    """(blur score = Laplacian variance, glare = share of near-255 pixels), on a <=1024 px copy."""
    g = _gray(img_bgr)
    s = min(1.0, 1024 / max(g.shape))
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float((g >= 250).mean())


def quality_gate(img_bgr) -> tuple[bool, str | None]:
    blur, glare = scores(img_bgr)
    if blur < BLUR_MIN:
        return False, "The photo is too blurry to read. Please retake it with the receipt in focus."
    if glare > GLARE_MAX:
        return False, "The photo has too much glare. Please retake it without direct light on the receipt."
    return True, None


def _ink(g):
    """Dark-on-light text mask."""
    block = max(31, (min(g.shape) // 40) | 1)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15)


def text_height(g) -> float | None:
    """Pixel-count-weighted median height of character-sized connected components (None if too
    few). The weighting ignores specks: on 60 test receipts it is 0.7-0.9x the gold box height,
    where a plain median drops to 0.35x on CORD photos."""
    _, _, st, _ = cv2.connectedComponentsWithStats(_ink(g), connectivity=8)
    w, h, a = st[1:, 2], st[1:, 3], st[1:, 4]
    k = (h >= 5) & (h <= g.shape[0] / 8) & (w <= 2 * h)
    if k.sum() < 10:
        return None
    o = np.argsort(h[k])
    cum = np.cumsum(a[k][o])
    return float(h[k][o][np.searchsorted(cum, cum[-1] / 2)])


def _find_quad(g):
    """Corners (tl, tr, br, bl) of the largest convincing 4-corner bright contour, or None."""
    s = min(1.0, 800 / max(g.shape))
    small = cv2.GaussianBlur(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (5, 5), 0)
    # ponytail: Otsu paper-vs-background mask only; white receipt on a white table is skipped.
    _, m = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = small.shape
    pad = 0.01 * max(H, W)
    for c in sorted(cs, key=cv2.contourArea, reverse=True)[:3]:
        a = cv2.contourArea(c)
        if a < MIN_QUAD * H * W:
            return None
        q = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float64)
        # a corner on the image border is where the paper leaves the frame, not a paper corner:
        # warping to it cut whole receipts away (CORD test 0015, WildReceipt test 0004)
        inside = ((q > pad) & (q < [W - 1 - pad, H - 1 - pad])).all()
        if len(q) == 4 and inside and cv2.isContourConvex(q.astype(np.int32)):
            q /= s
            c0 = q.mean(0)
            q = q[np.argsort(np.arctan2(q[:, 1] - c0[1], q[:, 0] - c0[0]))]  # clockwise (y down)
            return np.roll(q, -int(np.argmin(q.sum(1))), axis=0)          # start at top-left
    return None


def _skew(g) -> float | None:
    """Text-line angle in degrees (positive = lines run down to the right), weighted median of
    min-area rects of horizontally dilated text; None if too few lines."""
    th = text_height(g) or 20.0
    k = max(3, int(1.5 * th))
    m = cv2.dilate(_ink(g), cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # LIST: lines inside a paper-edge band
    angs, lens = [], []
    for c in cs:
        p = cv2.boxPoints(cv2.minAreaRect(c))
        e1, e2 = p[1] - p[0], p[2] - p[1]
        long_, short = (e1, e2) if np.hypot(*e1) >= np.hypot(*e2) else (e2, e1)
        L, S = np.hypot(*long_), np.hypot(*short)
        if L < 3 * S or L < 3 * th or S > 2.5 * th:
            continue
        a = np.degrees(np.arctan2(long_[1], long_[0]))
        a = (a + 90) % 180 - 90
        angs.append(a)
        lens.append(L)
    if len(angs) < 3:
        return None
    o = np.argsort(angs)
    cum = np.cumsum(np.array(lens)[o])
    return float(np.array(angs)[o][np.searchsorted(cum, cum[-1] / 2)])


def run(img_bgr, steps=STEPS_DEFAULT, max_upscale=MAX_UPSCALE) -> tuple[np.ndarray, np.ndarray]:
    """Apply the given steps (2..6). Returns (processed image, 3x3 M original->processed).
    The image is grayscale once step 2, 5 or 6 ran; callers convert if their engine needs BGR."""
    img, M = img_bgr, np.eye(3)
    if 2 in steps:
        img = _gray(img)
        h, w = img.shape
        th = text_height(img)
        s = min(TEXT_PX / th if th else 1.0, max_upscale, MAX_SIDE / max(h, w))
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            sx, sy = nw / w, nh / h  # cv2.resize maps pixel centres: x' = sx*(x+0.5)-0.5
            M = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]]) @ M
    if 3 in steps:
        q = _find_quad(_gray(img))
        if q is not None:
            tl, tr, br, bl = q
            w = int(round(max(np.hypot(*(tr - tl)), np.hypot(*(br - bl)))))
            h = int(round(max(np.hypot(*(bl - tl)), np.hypot(*(br - tr)))))
            W = cv2.getPerspectiveTransform(q.astype(np.float32),
                                            np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
            img = cv2.warpPerspective(img, W, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            M = W @ M
    if 4 in steps:
        a = _skew(_gray(img))
        if a is not None and 0.1 <= abs(a) <= MAX_SKEW:
            h, w = img.shape[:2]
            R = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)  # positive = counter-clockwise
            c, s = abs(R[0, 0]), abs(R[0, 1])
            nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
            R[0, 2] += nw / 2 - w / 2
            R[1, 2] += nh / 2 - h / 2
            img = cv2.warpAffine(img, R, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            M = np.vstack([R, [0, 0, 1]]) @ M
    if 5 in steps:  # shadow removal; kernels assume ~32 px text (step 2)
        g = _gray(img)
        bg = cv2.medianBlur(cv2.dilate(g, np.ones((7, 7), np.uint8)), 21)
        img = cv2.divide(g, bg, scale=255)
    if 6 in steps:  # Tesseract path only
        img = cv2.medianBlur(cv2.adaptiveThreshold(_gray(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 31, 15), 3)
    return img, M
