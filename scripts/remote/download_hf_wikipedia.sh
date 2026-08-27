#!/usr/bin/env bash
set -euo pipefail

project_dir=/root/autodl-tmp/tiny-rime-reranker
corpus_dir="$project_dir/corpus"
hf_runtime_dir="$project_dir/tmp/hf-home"
dataset_repo=fjcanyue/wikipedia-zh-cn
dataset_file=wikipedia-zh-cn-20260501.json

mkdir -p "$corpus_dir" "$hf_runtime_dir"
available_kb=$(df -Pk "$project_dir" | awk 'NR==2 {print $4}')
minimum_after_kb=$((15 * 1024 * 1024))
expected_download_kb=$((3 * 1024 * 1024))
if (( available_kb - expected_download_kb < minimum_after_kb )); then
  echo "refusing download: projected data-disk free space would fall below 15GiB" >&2
  exit 2
fi

env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HOME="$hf_runtime_dir" \
  HF_HUB_DOWNLOAD_TIMEOUT=120 \
  hf download "$dataset_repo" "$dataset_file" \
    --repo-type dataset \
    --local-dir "$corpus_dir"

test -s "$corpus_dir/$dataset_file"
sha256sum "$corpus_dir/$dataset_file"
