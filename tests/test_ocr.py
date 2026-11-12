import cv2
import numpy as np

from receipt import ocr


def test_cache_path():
    assert ocr.cache_path("paddle", (2, 3, 4, 5), "sroie", "test").as_posix().endswith(
        "data/ocr_cache/paddle-s2345/sroie/test.jsonl")
    assert ocr.steps_tag(()) == ocr.steps_tag("none") == "none"
    assert ocr.steps_tag("2,3,4,5") == "2345"


def test_paddle_sample_receipt():
    img = cv2.imread("docs/demo/sample-receipt.jpg")
    words = ocr.ocr(img, "paddle")
    texts = [w["text"] for w in words]
    assert "TOTAL" in texts and "21.32" in texts
    h, w = img.shape[:2]
    for q in (x["quad"] for x in words if x["text"] == "21.32"):
        cx, cy = np.mean(q[0::2]), np.mean(q[1::2])  # original coords: right lower half
        assert cx > w / 2 and cy > h / 2
