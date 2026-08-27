# Local artifact policy

This directory is intentionally deny-by-default. The Mac may retain only a reviewed best deployment candidate, its tokenizer/vocabulary/metadata, checksums, and final Core ML output under `deployment/`.

Remote corpora, complete 100k/1M datasets, large candidate caches, optimizer states, training temporaries, build caches, package caches, and complete raw training logs must never be copied here.
