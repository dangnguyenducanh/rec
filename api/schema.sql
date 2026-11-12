-- Spec §4 stage 7. Receipt JSON (contract, spec §3) lives in `data`; edits are kept as corrections
-- because they are future training labels for the extractors.
CREATE TABLE IF NOT EXISTS receipts (
  id          serial PRIMARY KEY,
  created_at  timestamptz NOT NULL DEFAULT now(),
  extractor   text NOT NULL,
  image_path  text,
  data        jsonb NOT NULL,
  raw_ocr     jsonb
);

CREATE TABLE IF NOT EXISTS corrections (
  id          serial PRIMARY KEY,
  receipt_id  int NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
  created_at  timestamptz NOT NULL DEFAULT now(),
  before      jsonb NOT NULL,
  after       jsonb NOT NULL
);
