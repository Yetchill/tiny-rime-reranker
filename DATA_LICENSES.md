# Data licenses and release status

| Data | License/status | Permitted use in this project |
|---|---|---|
| Chinese Wikipedia article text | Wikimedia text is generally CC BY-SA 4.0 and GFDL; individual contributions and attribution requirements still apply | Remote research pipeline only; preserve source document IDs and dump metadata. No dump or corpus is copied to Mac. Weight release requires a separate documented review. |
| `fjcanyue/wikipedia-zh-cn`, `wikipedia-zh-cn-20260501.json` | Dataset card states the source is Chinese Wikipedia and identifies GFDL 1.3 and CC BY-SA 4.0; individual-page exceptions may still apply. HF revision `38a697eb24e84c569ce05cb5f23336bdeb6a94c3`; file SHA-256 `c8c719a84d402371ffa6b99b57bc9bc524bf66e07d72dfc724e51d0224eaee62`. | User-selected remote research source. Never copy the 2.39GB JSONL or complete derived dataset to Mac. Preserve `id` as the document split key. Weight release still requires provenance review. |
| rime-ice dictionary | GPL-3.0-only repository license | External candidate-generation baseline. The committed 100-item fixture is small reproducibility evidence; preserve attribution. Do not vendor the full dictionary. |
| RIME-LMDG grammar/model data | CC-BY-4.0 repository license | External strong baseline only with attribution. AutoDL file `wanxiang-lts-zh-hans.gram`: 420,250,668 bytes, SHA-256 `01ffe37f22607bf8a5cd5d82a3349f6df97744369464aee4577585112d85469d`. It is not vendored or copied to Mac. |
| OpenIME People’s Daily/TouchPal processed data | No repository license file found at locked commit | Do not download, train on, redistribute, or claim releasable weights. |
| User typing history or personal dictionary | Private user data | Prohibited from collection, fixtures, training, logs, and transfer. |

Until training-input provenance and downstream weight obligations have been reviewed, experimental weights must not be described as publicly redistributable.
