# Data licenses and release status

| Data | License/status | Permitted use in this project |
|---|---|---|
| Chinese Wikipedia article text | Wikimedia text is generally CC BY-SA 4.0 and GFDL; individual contributions and attribution requirements still apply | Remote research pipeline only; preserve source document IDs and dump metadata. No dump or corpus is copied to Mac. Weight release requires a separate documented review. |
| rime-ice dictionary | GPL-3.0-only repository license | External candidate-generation baseline. The committed 100-item fixture is small reproducibility evidence; preserve attribution. Do not vendor the full dictionary. |
| RIME-LMDG grammar/model data | CC-BY-4.0 repository license; release asset must be separately checksummed | External strong baseline only with attribution. No `.gram` file is currently vendored. |
| OpenIME People’s Daily/TouchPal processed data | No repository license file found at locked commit | Do not download, train on, redistribute, or claim releasable weights. |
| User typing history or personal dictionary | Private user data | Prohibited from collection, fixtures, training, logs, and transfer. |

Until training-input provenance and downstream weight obligations have been reviewed, experimental weights must not be described as publicly redistributable.
