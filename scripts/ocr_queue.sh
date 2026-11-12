#!/usr/bin/env bash
# Lane V OCR queue. Run detached from the repo root:
#   setsid nohup bash scripts/ocr_queue.sh > data/ocr_cache/queue.log 2>&1 < /dev/null &
# build_ocr_cache.py skips every cache file that already exists, so a rerun resumes where it stopped.
# Ends by touching data/ocr_cache/QUEUE_DONE.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -s"
rm -f data/ocr_cache/QUEUE_DONE

cache() {  # engine steps datasets splits
  echo "== $(date '+%F %T') OCR $1 s$2 $3 $4"
  $PY scripts/build_ocr_cache.py --engine "$1" --steps "$2" --datasets "$3" --splits "$4" --workers 8
}

test_sets() {  # engine steps: each test set, then its detection/CER row in eval/diagnostics/ocr.csv
  for ds in wildreceipt sroie cord; do
    cache "$1" "$2" "$ds" test
    $PY scripts/diagnostics.py ocr --engine "$1" --steps "$2" --datasets "$ds"
  done
}

bench() {  # steps: crf benchmark on this Paddle ablation, once lane M's models/crf.pkl exists (wait <= 2 h total)
  : "${CRF_DEADLINE:=$(($(date +%s) + 7200))}"
  while [ ! -f models/crf.pkl ] && [ "$(date +%s)" -lt "$CRF_DEADLINE" ]; do sleep 60; done
  if [ -f scripts/benchmark.py ] && [ -f models/crf.pkl ]; then
    echo "== $(date '+%F %T') benchmark crf paddle s$1"
    $PY scripts/benchmark.py --extractor crf --ocr paddle --steps "$1" || echo "benchmark failed: s$1"
  else
    echo "skip benchmark s$1: no models/crf.pkl or scripts/benchmark.py"
  fi
}

# a) Paddle, default preprocessing, the three test sets; then row grouping on Paddle words
test_sets paddle 2,3,4,5
$PY scripts/diagnostics.py rows --ocr paddle
# b) Paddle on WildReceipt train/val (LayoutLMv3 phase B input), CORD train/val and SROIE train (phase B val for
#    every run, and B4's extra training data; SROIE has no val split: build_ocr_cache.py skips it)
cache paddle 2,3,4,5 wildreceipt train,val
cache paddle 2,3,4,5 cord,sroie train,val
# c) Tesseract baseline (step 6 = binarize) on the test sets
test_sets tesseract 2,3,4,5,6
# d) preprocessing ablation: no preprocessing, then each of steps 2-5 left out
for s in none 3,4,5 2,4,5 2,3,5 2,3,4; do
  test_sets paddle "$s"
  bench "$s"
done

echo "== $(date '+%F %T') queue done"
touch data/ocr_cache/QUEUE_DONE
