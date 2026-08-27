# Third-party software notices

No third-party source is vendored in this repository. The following projects were inspected or used externally during research; a future distribution must carry complete notices for anything it bundles.

| Component | License | Current use |
|---|---|---|
| librime | BSD-3-Clause | External build dependency and API reference |
| Squirrel | GPL-3.0 | Source analysis only |
| rime-ice | GPL-3.0-only | External baseline data/config; no dictionary vendored |
| RIME-LMDG | CC-BY-4.0 | External Wanxiang grammar baseline on AutoDL; no model vendored or copied to Mac |
| librime-octagram | GPL-3.0 | External AutoDL build for the measured grammar baseline; no source vendored |
| librime-ai-predict | BSD-3-Clause | Source analysis only |
| Cassotis IME | GPL-3.0 | Source analysis only; no code/model copied |
| OpenIME | No license file found at locked commit | Not used |

Transitive libraries in the external librime build retain their upstream licenses, including Boost, glog, LevelDB, marisa-trie, OpenCC, yaml-cpp, and GoogleTest. Consult the locked upstream tree before distributing binaries.
