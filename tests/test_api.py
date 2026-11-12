"""Upload -> DB row -> PATCH -> GET round trip, plus review checks. Needs docker Postgres on 18332."""
import io
import os

import psycopg
import pytest
from PIL import Image

ADMIN_DSN = "postgresql://rec:rec@localhost:18332/rec"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with psycopg.connect(ADMIN_DSN, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS rec_test")
        c.execute("CREATE DATABASE rec_test")
    os.environ["DATABASE_URL"] = "postgresql://rec:rec@localhost:18332/rec_test"
    os.environ["UPLOAD_DIR"] = str(tmp_path_factory.mktemp("uploads"))
    os.environ["RECEIPT_PIPELINE"] = "stub"
    from fastapi.testclient import TestClient

    from api.main import app
    with TestClient(app) as c:
        yield c


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), "white").save(buf, "PNG")
    return buf.getvalue()


def test_round_trip(client):
    r = client.post("/receipts", files={"file": ("r.png", png(), "image/png")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["receipt"]["needs_review"] is False  # fixture adds up

    edited = body["receipt"] | {"total": "99.00"}
    r = client.patch(f"/receipts/{body['id']}", json=edited)
    assert r.status_code == 200, r.text
    assert r.json()["receipt"]["needs_review"] is True
    assert "total says 99.00" in r.json()["receipt"]["review_reasons"][0]

    assert client.get(f"/receipts/{body['id']}").json()["receipt"]["total"] == "99.00"
    assert client.get(f"/receipts/{body['id']}/image").status_code == 200
    cats = {c["category"]: c["amount"] for c in client.get("/stats").json()["by_category"]}
    assert cats == {"household": "10.78", "groceries": "9.76"}


def test_rejects_non_image(client):
    r = client.post("/receipts", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_missing_receipt_is_404(client):
    assert client.get("/receipts/999999").status_code == 404


def test_rejected_photo_is_422(client, monkeypatch):
    from receipt import pipeline

    def blurry(_data):
        raise pipeline.Rejected("The photo is too blurry to read.")
    monkeypatch.setattr(pipeline, "extract", blurry)
    r = client.post("/receipts", files={"file": ("r.png", png(), "image/png")})
    assert r.status_code == 422
    assert r.json()["detail"] == "The photo is too blurry to read."
