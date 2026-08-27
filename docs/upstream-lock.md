# Upstream lock

Resolved from each repository's `HEAD` on 2026-08-27. The full SHA, rather than a moving branch name, is the reproducibility identity.

| Repository | Commit | License observed in repository | Use in TinyRime |
|---|---|---|---|
| rime/librime | `13faefe2819d01fce208752c2539744094bb4787` | BSD-3-Clause | C API runner and candidate/filter contracts |
| rime/squirrel | `0cd71a6130a5866b0ae6ba0494929ebdc8211194` | GPL-3.0 | macOS lifecycle and frontend boundary analysis only |
| iDvel/rime-ice | `75e6572bebc05b49021e842949ce947882e3e4b2` | GPL-3.0-only | baseline dictionary/config; deterministic fixture output |
| amzxyz/RIME-LMDG | `bf665619e2807ab5b7f27557ad21602dd01c0c4e` | CC-BY-4.0 | Wanxiang/octagram comparison design; no model vendored |
| wyjrichhh/librime-ai-predict | `31cb20d6b786f460207451a5ec9150906bda417f` | BSD-3-Clause | filter/lifecycle/context/fallback analysis only |
| shenmin/cassotis-ime | `786fd5daedf985483bb2d8c1314c061dae1a8be1` | GPL-3.0 | conservative residual-ranking design analysis only |
| cooelf/OpenIME | `20c60197b580ce8d55d856067080860c360bce76` | No repository license file found | Paper/dataset landscape only; no code/data used |

The rime-ice baseline is deliberately reduced to the core `script_translator` in `fixtures/rime/tinyrime_ice.schema.yaml`: Lua, emoji, private custom phrases, and user learning are excluded for deterministic headless fixtures. This is not falsely labeled as the full desktop rime-ice experience.

The checked librime revision requests Boost 1.77 or newer, while Ubuntu 22.04 supplied 1.74. Gate 1 therefore built librime statically with external dynamic-plugin loading disabled. Core modules, dictionary compilation, translation, and the C API runner were built and exercised; this compatibility choice must be removed before building the production plugin stack.
