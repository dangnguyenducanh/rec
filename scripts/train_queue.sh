#!/usr/bin/env bash
# Lane M queue: one GPU job at a time, in order. Per-job logs in models/logs/. Launch detached:
#   setsid nohup scripts/train_queue.sh > models/logs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -s"
L=models/logs
C=data/ocr_cache/paddle-s2345
mkdir -p $L
rm -f models/QUEUE_DONE

job() {  # job NAME CMD... ; a failed job is logged and the queue moves on
  local name=$1; shift
  echo "$(date '+%F %T') start $name: $*"
  "$@" > "$L/$name.log" 2>&1
  echo "$(date '+%F %T') end $name rc=$?"
}

job train_roberta-A $PY scripts/train_transformer.py --model roberta --phase A
job bench_roberta-A_gold $PY scripts/benchmark.py --extractor roberta-A --ocr gold
job train_layoutlmv3-A $PY scripts/train_transformer.py --model layoutlmv3 --phase A
job bench_layoutlmv3-A_gold $PY scripts/benchmark.py --extractor layoutlmv3-A --ocr gold

for f in wildreceipt/train wildreceipt/val wildreceipt/test sroie/train sroie/test cord/train cord/val cord/test; do
  [ -f "$C/$f.jsonl" ] || echo "$(date '+%F %T') waiting for $C/$f.jsonl (poll 60 s)"
  until [ -f "$C/$f.jsonl" ]; do sleep 60; done
done
echo "$(date '+%F %T') paddle caches ready"

job train_layoutlmv3-B $PY scripts/train_transformer.py --model layoutlmv3 --phase B
job bench_layoutlmv3-B_gold $PY scripts/benchmark.py --extractor layoutlmv3-B --ocr gold
job bench_layoutlmv3-B_paddle $PY scripts/benchmark.py --extractor layoutlmv3-B --ocr paddle --steps 2,3,4,5
job train_layoutlmv3-B4 $PY scripts/train_transformer.py --model layoutlmv3 --phase B --datasets wildreceipt,cord,sroie \
    --out models/layoutlmv3-B4
job bench_layoutlmv3-B4_gold $PY scripts/benchmark.py --extractor layoutlmv3-B4 --ocr gold
job bench_layoutlmv3-B4_paddle $PY scripts/benchmark.py --extractor layoutlmv3-B4 --ocr paddle --steps 2,3,4,5
for x in rules crf roberta-A layoutlmv3-A; do
  job "bench_${x}_paddle" $PY scripts/benchmark.py --extractor "$x" --ocr paddle --steps 2,3,4,5
done
touch models/QUEUE_DONE
echo "$(date '+%F %T') queue done"
