# Upstream source analysis

## librime

`Candidate` owns type, input range, and quality and exposes text, comment, and preedit. `ShadowCandidate` preserves the wrapped candidate's range, quality, preedit, and optionally comment; this is the right mechanism when metadata decoration is unavoidable. TinyRime's intended filter should instead buffer `an<Candidate>` objects and return those genuine objects in a stable reordered translation, avoiding reconstruction.

`Translation` is a lazy candidate generator (`Peek`, `Next`, `exhausted`). `Filter::Apply` receives the upstream translation and the prior candidate list, so a filter-style bounded Top-8 prefetch is the natural insertion point. Candidate quality/type are available in the C++ API but not in public `RimeCandidate`; consequently the headless C API runner emits `null` for them and documents this limitation rather than inventing values.

The current `Context` owns `CommitHistory`, whose records preserve a type and committed text with at most 20 records. The plugin should traverse recent non-raw/non-thru commits and cap the resulting UTF-8 context to 32 characters. Global plugin registration belongs to module initialization; model lifetime should be schema/filter scoped or shared behind an immutable backend, never initialized for each candidate.

## Squirrel

The app delegate calls global Rime setup/start/finalize; each `SquirrelInputController` owns a Rime session. Candidate UI state is read synchronously via `get_context`, copied into Swift strings, then freed with `free_context`. A librime filter therefore keeps the first prototype below Squirrel's Swift/C boundary and requires no panel mutation. Squirrel currently displays the final menu it reads after processing a key, so TinyRime must finish synchronously before that read or abstain for the composition.

## librime-ai-predict

Useful patterns are the filter wrapper around an upstream `Translation`, preservation via `ShadowCandidate`, extracting committed context while skipping `raw` and `thru` records, model initialization validation, and failure returning the original translation. Its async engine also guards refreshes against changed composition and user navigation.

TinyRime does not adopt its seq2seq generation, novel candidate insertion, CTranslate2 dependency, or active asynchronous UI refresh. Even a carefully guarded refresh can visibly change a menu after presentation; TinyRime instead accepts only an on-time synchronous result and permanently discards late work for that composition.

## Cassotis

The source contains separate short-context residual/difference models, explicit promotion thresholds, contextual and non-contextual features, pairwise comparisons against the baseline top candidate, and fallback/abstention scores. The relevant product lesson is to optimize net wins and promotion precision, not reorder coverage. TinyRime adopts this conservative evaluation philosophy, not its GPL implementation, generated Pascal models, embedded evidence tables, Windows TSF code, dictionaries, or ONNX artifacts.

## rime-ice and Wanxiang/octagram

rime-ice is a strong dictionary baseline but its full schema includes Lua and presentation filters. The deterministic Gate 1 lane uses the same pinned dictionary with a minimal core schema. Wanxiang's grammar lane requires the octagram plugin plus a separately downloaded `.gram` release model; neither a score nor an equivalence claim is reported until that exact stack is pinned, checksummed, and run on the same split.

## OpenIME

OpenIME is an unconstrained neural pinyin-to-text system derived from OpenNMT and ships processed People’s Daily/TouchPal data references. It is architecturally outside this project because it generates text rather than permuting Rime candidates. No license file was found at the locked commit, so neither its code nor data is used.
