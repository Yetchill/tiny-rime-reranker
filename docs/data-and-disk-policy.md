# Data and disk policy

The Mac is the source of truth for Git-tracked code and configuration, not a training-data store. It may retain dataset statistics, at most 1,000 reproducibility fixtures from public or de-identified sources, evaluation reports and metrics, a selected deployment candidate with required vocabulary/metadata/checksums, and final Core ML artifacts.

Never transfer a Wikipedia dump, complete cleaned corpus, complete 100k/1M dataset, large Rime candidate cache, optimizer state, training temporary, remote build cache, package cache, or complete raw training-log dump from AutoDL to the Mac.

Every AutoDL-to-Mac transfer must first create a manifest containing relative path, byte count, and SHA-256 plus a total byte count. Above 500 MB, review the list for accidental prohibited content. The transfer may proceed when the size is genuinely required by model or deployment artifacts; record the reason in the manifest note. The 500 MB value is not a project-size cap and does not stop local builds or benchmarks.

The complete local working tree, including ignored builds/caches, should normally stay near 3 GB. Inspect its composition above 5 GB. Large builds, datasets, caches, and checkpoints are never Git-tracked.
