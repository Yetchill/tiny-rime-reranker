# Local artifact policy

This directory is intentionally deny-by-default. The Mac may retain only reviewed deployment candidates, their tokenizer/vocabulary/metadata, checksums, and final Core ML output under `deployment/`.

Remote corpora, complete 100k/1M datasets, large candidate caches, optimizer states, training temporaries, build caches, package caches, and complete raw training logs must never be copied here.

Before every AutoDL-to-Mac transfer, produce a manifest and total byte count. A transfer above 500 MB triggers review, not automatic cancellation: verify that no prohibited data was included, then proceed only when the size comes from necessary model or deployment artifacts and record the reason. The whole local working directory targets roughly 3 GB including ignored builds/caches; inspect its composition above 5 GB. Git-tracked content remains deliberately small.
