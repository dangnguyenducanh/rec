"""Postgres access. One connection per call: single-user app, no pool needed."""
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

COLS = "id, created_at, extractor, image_path, data"


def connect():
    dsn = os.environ.get("DATABASE_URL", "postgresql://rec:rec@localhost:18332/rec")
    return psycopg.connect(dsn, row_factory=dict_row)


def init():
    with connect() as c:
        c.execute((Path(__file__).parent / "schema.sql").read_text())


def insert(extractor: str, image_path: str, data: dict, raw_ocr: dict) -> dict:
    with connect() as c:
        return c.execute(
            f"INSERT INTO receipts (extractor, image_path, data, raw_ocr) VALUES (%s, %s, %s, %s) RETURNING {COLS}",
            (extractor, image_path, Jsonb(data), Jsonb(raw_ocr)),
        ).fetchone()


def all_receipts() -> list[dict]:
    with connect() as c:
        return c.execute(f"SELECT {COLS} FROM receipts ORDER BY id DESC").fetchall()


def get(rid: int) -> dict | None:
    with connect() as c:
        return c.execute(f"SELECT {COLS} FROM receipts WHERE id = %s", (rid,)).fetchone()


def update(rid: int, data: dict) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT data FROM receipts WHERE id = %s FOR UPDATE", (rid,)).fetchone()
        if row is None:
            return None
        c.execute("INSERT INTO corrections (receipt_id, before, after) VALUES (%s, %s, %s)",
                  (rid, Jsonb(row["data"]), Jsonb(data)))
        return c.execute(f"UPDATE receipts SET data = %s WHERE id = %s RETURNING {COLS}",
                         (Jsonb(data), rid)).fetchone()


def stats() -> dict:
    with connect() as c:
        by_category = c.execute("""
            SELECT COALESCE(i->>'category', 'other') AS category,
                   SUM((i->>'price')::numeric)::numeric(12,2)::text AS amount
            FROM receipts, jsonb_array_elements(data->'items') i
            WHERE i->>'price' IS NOT NULL
            GROUP BY 1 ORDER BY SUM((i->>'price')::numeric) DESC""").fetchall()
        by_month = c.execute("""
            SELECT to_char((data->>'date')::date, 'YYYY-MM') AS month,
                   SUM((data->>'total')::numeric)::numeric(12,2)::text AS amount,
                   COUNT(*) AS receipts
            FROM receipts WHERE data->>'date' IS NOT NULL AND data->>'total' IS NOT NULL
            GROUP BY 1 ORDER BY 1""").fetchall()
    return {"by_category": by_category, "by_month": by_month}
