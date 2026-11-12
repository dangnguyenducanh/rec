"""FastAPI app: /receipts (POST/GET/PATCH), /stats, /health (spec §4 stage 7)."""
import datetime as dt
import io
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from api import db
from receipt import pipeline
from receipt.schema import Receipt
from receipt.validate import check

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", Path(__file__).resolve().parents[1] / "data" / "uploads"))
MAX_BYTES = 10 * 1024 * 1024
FORMATS = {"jpeg", "png", "webp"}


class Stored(BaseModel):
    id: int
    created_at: dt.datetime
    extractor: str
    receipt: Receipt


class CategoryTotal(BaseModel):
    category: str
    amount: str


class MonthTotal(BaseModel):
    month: str
    amount: str
    receipts: int


class Stats(BaseModel):
    by_category: list[CategoryTotal]
    by_month: list[MonthTotal]


def _out(row: dict) -> Stored:
    return Stored(id=row["id"], created_at=row["created_at"], extractor=row["extractor"],
                  receipt=Receipt.model_validate(row["data"]))


def _row(rid: int) -> dict:
    row = db.get(rid)
    if row is None:
        raise HTTPException(404, "Receipt not found")
    return row


@asynccontextmanager
async def lifespan(_app):
    db.init()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Receipt Reader", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:18373").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "extractor": pipeline.EXTRACTOR}


@app.post("/receipts", response_model=Stored, status_code=201)
def create_receipt(file: UploadFile):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").lower()
        img.verify()
    except Exception:
        raise HTTPException(400, "That file isn't an image we can read")
    if fmt not in FORMATS:
        raise HTTPException(400, "Use a JPG, PNG or WebP photo")
    name = f"{uuid.uuid4().hex}.{fmt}"
    (UPLOAD_DIR / name).write_bytes(data)
    receipt = check(pipeline.extract(data))
    return _out(db.insert(pipeline.EXTRACTOR, name, receipt.model_dump(mode="json")))


@app.get("/receipts", response_model=list[Stored])
def list_receipts():
    return [_out(r) for r in db.all_receipts()]


@app.get("/receipts/{rid}", response_model=Stored)
def get_receipt(rid: int):
    return _out(_row(rid))


@app.get("/receipts/{rid}/image")
def get_image(rid: int):
    return FileResponse(UPLOAD_DIR / _row(rid)["image_path"])


@app.patch("/receipts/{rid}", response_model=Stored)
def update_receipt(rid: int, receipt: Receipt):
    row = db.update(rid, check(receipt).model_dump(mode="json"))
    if row is None:
        raise HTTPException(404, "Receipt not found")
    return _out(row)


@app.get("/stats", response_model=Stats)
def stats():
    return db.stats()
