#!/usr/bin/env bash
set -euo pipefail

project_dir=/root/autodl-tmp/tiny-rime-reranker
source_dir="$project_dir/source"
v1_dir="$project_dir/datasets/TinyRime-Context-v1"
report_dir="$project_dir/reports/TinyRime-Context-v1"

test -f "$report_dir/benchmark_manifest.json"
test -f "$v1_dir/ranking-k8/test.jsonl.zst"

g++ -DGLOG_USE_GLOG_EXPORT -std=c++17 -O2 \
  -I"$project_dir/upstream/librime/src" \
  -I"$project_dir/upstream/librime/build-octagram/src" \
  -I"$project_dir/upstream/librime/deps/glog/src" \
  -I"$project_dir/upstream/librime/deps/glog/build" \
  -I"$project_dir/upstream/librime/include" \
  "$source_dir/plugin/src/rime_runner.cc" \
  "$project_dir/upstream/librime/build-octagram/lib/librime.a" \
  "$project_dir/upstream/librime/lib/libglog.a" \
  "$project_dir/upstream/librime/lib/libleveldb.a" \
  "$project_dir/upstream/librime/lib/libmarisa.a" \
  "$project_dir/upstream/librime/lib/libopencc.a" \
  "$project_dir/upstream/librime/lib/libyaml-cpp.a" \
  -lboost_regex -ldl -lpthread \
  -o "$project_dir/bin/tinyrime_rime_runner_octagram"

rm -rf "$project_dir/work/context-v1-wanxiang-workers"
PYTHONPATH="$source_dir" python "$source_dir/data_pipeline/score_queries_v1.py" \
  "$v1_dir/queries" "$v1_dir/scored-wanxiang-top32" \
  --workers 8 \
  --batch-size 512 \
  --top-k 32 \
  --recall-k 1,3,5,8,12,16,24,32 \
  --display-k 8 \
  --miss-sample-per-split 5000 \
  --seed 20260827 \
  --profile "$report_dir/wanxiang-top32-profile.json" \
  --runner "$project_dir/bin/tinyrime_rime_runner_octagram" \
    --shared-data "$project_dir/upstream/rime-ice" \
    --user-data "$project_dir/work/context-v1-wanxiang-workers/{worker}" \
    --prebuilt-data "$project_dir/work/rime-prebuilt-octagram" \
    --schema tinyrime_ice_octagram \
    --top-k 32 --skip-maintenance --skip-deploy

PYTHONPATH="$source_dir" python -m benchmark.offline.candidate_union \
  --rime-scored-dir "$v1_dir/scored-rime-top32" \
  --wanxiang-scored-dir "$v1_dir/scored-wanxiang-top32" \
  --budgets 16,24,32 \
  --output "$report_dir/candidate-union.json"

rm -rf "$project_dir/work/phase2b-export-user"
PYTHONPATH="$source_dir" python -m benchmark.offline.export_prediction_artifact \
  --dataset-dir "$v1_dir/ranking-k8" \
  --output "$report_dir/phase2b-predictions.jsonl.zst" \
  --checkpoint linear="$project_dir/experiments/real-first-sweep/linear/best.safetensors" \
  --checkpoint mlp="$project_dir/experiments/real-first-sweep/mlp/best.safetensors" \
  --checkpoint tiny_2m="$project_dir/experiments/real-first-sweep/tiny-2m/best.safetensors" \
  --checkpoint tiny_4m="$project_dir/experiments/real-first-sweep/tiny-4m/best.safetensors" \
  --checkpoint tiny_8m="$project_dir/experiments/real-first-sweep/tiny-8m/best.safetensors" \
  --wanxiang-runner "$project_dir/bin/tinyrime_rime_runner_octagram" \
    --shared-data "$project_dir/upstream/rime-ice" \
    --user-data "$project_dir/work/phase2b-export-user" \
    --prebuilt-data "$project_dir/work/rime-prebuilt-octagram" \
    --schema tinyrime_ice_octagram \
    --top-k 8 --skip-maintenance --skip-deploy

PYTHONPATH="$source_dir" python -m benchmark.offline.error_overlap \
  --predictions "$report_dir/phase2b-predictions.jsonl.zst" \
  --output "$report_dir/error-overlap" \
  --seed 20260827

python "$source_dir/scripts/remote/create_sync_manifest.py" \
  "$project_dir" \
  "reports/TinyRime-Context-v1/benchmark_manifest.json" \
  "reports/TinyRime-Context-v1/reservoir-statistics.json" \
  "reports/TinyRime-Context-v1/query-statistics.json" \
  "reports/TinyRime-Context-v1/rime-top32-profile.json" \
  "reports/TinyRime-Context-v1/wanxiang-top32-profile.json" \
  "reports/TinyRime-Context-v1/candidate-union.json" \
  "reports/TinyRime-Context-v1/ranking-k8-statistics.json" \
  "reports/TinyRime-Context-v1/ranking-k8-validation.json" \
  "reports/TinyRime-Context-v1/vocabulary-audit.json" \
  "reports/TinyRime-Context-v1/error-overlap" \
  --output "$project_dir/work/sync-manifest-context-v1.json"

df -h /root/autodl-tmp
du -sh "$project_dir" "$v1_dir" "$report_dir"
