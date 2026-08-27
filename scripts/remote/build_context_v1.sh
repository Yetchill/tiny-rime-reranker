#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 QUERY_GENERATION_GIT_COMMIT" >&2
  exit 2
fi

code_commit=$1
project_dir=/root/autodl-tmp/tiny-rime-reranker
source_dir="$project_dir/source"
v1_dir="$project_dir/datasets/TinyRime-Context-v1"
report_dir="$project_dir/reports/TinyRime-Context-v1"
corpus="$project_dir/corpus/wikipedia-zh-cn-20260501.json"

available_kb=$(df -Pk "$project_dir" | awk 'NR==2 {print $4}')
minimum_kb=$((15 * 1024 * 1024))
if (( available_kb < minimum_kb )); then
  echo "refusing benchmark build: data-disk free space is below 15GiB" >&2
  exit 3
fi

mkdir -p "$v1_dir" "$report_dir"

g++ -DGLOG_USE_GLOG_EXPORT -std=c++17 -O2 \
  -I"$project_dir/upstream/librime/src" \
  -I"$project_dir/upstream/librime/build-static/src" \
  -I"$project_dir/upstream/librime/deps/glog/src" \
  -I"$project_dir/upstream/librime/deps/glog/build" \
  -I"$project_dir/upstream/librime/include" \
  "$source_dir/plugin/src/rime_runner.cc" \
  "$project_dir/upstream/librime/build-static/lib/librime.a" \
  "$project_dir/upstream/librime/lib/libglog.a" \
  "$project_dir/upstream/librime/lib/libleveldb.a" \
  "$project_dir/upstream/librime/lib/libmarisa.a" \
  "$project_dir/upstream/librime/lib/libopencc.a" \
  "$project_dir/upstream/librime/lib/libyaml-cpp.a" \
  -lboost_regex -ldl -lpthread \
  -o "$project_dir/bin/tinyrime_rime_runner"

PYTHONPATH="$source_dir" python "$source_dir/data_pipeline/wiki/sample_documents.py" \
  "$corpus" "$v1_dir/documents.jsonl.zst" \
  --source-format jsonl \
  --snapshot 20260501 \
  --source-sha256 c8c719a84d402371ffa6b99b57bc9bc524bf66e07d72dfc724e51d0224eaee62 \
  --sample-documents 30000 \
  --seed 20260827 \
  --stats "$report_dir/reservoir-statistics.json"

PYTHONPATH="$source_dir" python "$source_dir/data_pipeline/build_queries.py" \
  "$v1_dir/documents.jsonl.zst" "$v1_dir/queries" \
  --train-queries 250000 \
  --val-queries 25000 \
  --test-queries 25000 \
  --max-queries-per-document 16 \
  --stats "$report_dir/query-statistics.json"

rm -rf "$project_dir/work/context-v1-rime-workers"
PYTHONPATH="$source_dir" python "$source_dir/data_pipeline/score_queries_v1.py" \
  "$v1_dir/queries" "$v1_dir/scored-rime-top32" \
  --workers 8 \
  --batch-size 512 \
  --top-k 32 \
  --recall-k 1,3,5,8,12,16,24,32 \
  --display-k 8 \
  --miss-sample-per-split 5000 \
  --seed 20260827 \
  --profile "$report_dir/rime-top32-profile.json" \
  --runner "$project_dir/bin/tinyrime_rime_runner" \
    --shared-data "$project_dir/upstream/rime-ice" \
    --user-data "$project_dir/work/context-v1-rime-workers/{worker}" \
    --prebuilt-data "$project_dir/work/rime-prebuilt" \
    --schema tinyrime_ice \
    --top-k 32 --skip-maintenance --skip-deploy

PYTHONPATH="$source_dir" python "$source_dir/data_pipeline/derive_ranking_dataset.py" \
  "$v1_dir/scored-rime-top32" "$v1_dir/ranking-k8" \
  --pool-k 8 \
  --train-examples 100000 \
  --seed 20260827 \
  --stats "$report_dir/ranking-k8-statistics.json"

PYTHONPATH="$source_dir" python "$source_dir/data_pipeline/validate_dataset.py" \
  "$v1_dir/ranking-k8/train.jsonl.zst" \
  "$v1_dir/ranking-k8/val.jsonl.zst" \
  "$v1_dir/ranking-k8/test.jsonl.zst" \
  --require-canonical-contested \
  --output "$report_dir/ranking-k8-validation.json"

PYTHONPATH="$source_dir" python "$source_dir/training/build_vocabulary.py" \
  "$v1_dir/ranking-k8/train.jsonl.zst" \
  --embedding-capacity 32768 \
  --hash-capacities 4096,8192,16384,32768 \
  --vocabulary "$v1_dir/exact-vocabulary-32768.json" \
  --audit "$report_dir/vocabulary-audit.json"

PYTHONPATH="$source_dir" python -m benchmark.offline.freeze_benchmark \
  --dataset-manifest "$source_dir/reports/dataset-manifest.json" \
  --reservoir-stats "$report_dir/reservoir-statistics.json" \
  --query-stats "$report_dir/query-statistics.json" \
  --query-dir "$v1_dir/queries" \
  --scored-query-dir "$v1_dir/scored-rime-top32" \
  --ranking-stats "$report_dir/ranking-k8-statistics.json" \
  --ranking-dir "$v1_dir/ranking-k8" \
  --query-generation-commit "$code_commit" \
  --output "$report_dir/benchmark_manifest.json"

df -h /root/autodl-tmp
du -sh "$project_dir" "$v1_dir"
