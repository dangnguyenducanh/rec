import cv2
import numpy as np
import pytest

from receipt import preprocess

SAMPLE = "docs/demo/sample-receipt.jpg"


def _synthetic(angle=8.0):
    """Dark table, white receipt with text lines and a black dot, rotated by `angle` degrees.
    Returns (image, paper corners, dot centre), both in the rotated image's coordinates."""
    img = np.full((1000, 800), 60, np.uint8)
    x0, y0, x1, y1 = 200, 150, 600, 850
    cv2.rectangle(img, (x0, y0), (x1, y1), 235, -1)
    for i, y in enumerate(range(220, 800, 40)):
        cv2.putText(img, f"ITEM {i:02d} NAME   {i * 1.37:6.2f}", (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 20, 2)
    dot = (x1 - 40, y1 - 30)
    cv2.circle(img, dot, 9, 0, -1)
    R = cv2.getRotationMatrix2D((400, 500), angle, 1.0)
    img = cv2.warpAffine(img, R, (800, 1000), borderValue=60)
    move = lambda pts: cv2.transform(np.float64([pts]), R)[0]  # noqa: E731
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), move([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), move([dot])[0]


@pytest.mark.parametrize("steps", [(2,), (2, 4), (2, 3, 4)])
def test_matrix_maps_original_points(steps):
    img, corners, dot = _synthetic()
    out, M = preprocess.run(img, steps)
    p = M @ [*dot, 1]
    x, y = p[:2] / p[2]
    assert out[int(round(y)), int(round(x))] < 100            # the dot lands where M says
    back = np.linalg.inv(M) @ p
    assert np.allclose(back[:2] / back[2], dot, atol=1e-6)     # and comes back through inv(M)
    if 4 in steps:
        assert abs(preprocess._skew(out) or 0.0) < 1.0         # deskewed
    if 3 in steps:  # paper corners go to the processed image corners
        h, w = out.shape[:2]
        got = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
        assert np.abs(got - [(0, 0), (w, 0), (w, h), (0, h)]).max() < 0.03 * max(h, w)


def test_quality_gate_rejects_blur_accepts_sample():
    img = cv2.imread(SAMPLE)
    assert preprocess.quality_gate(img) == (True, None)
    ok, reason = preprocess.quality_gate(cv2.GaussianBlur(img, (0, 0), 6))
    assert not ok and "blurry" in reason
