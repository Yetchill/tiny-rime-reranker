# TinyRime 上下文候选重排器

TinyRime 是一个紧凑、完全本地运行的中文拼音候选重排研究项目。它接收 Rime 提供的前 8 个候选词，结合左侧上下文重新排序，但不会生成新文本：所有被接受的输出都只是原候选集合的稳定排列。如果模型选择放弃、超时、运行失败或给出非法结果，系统会原样保留 Rime 的候选顺序。

当前仓库是 **v0.1 研究版本**，包含冻结的 `TinyRime-Context-v1` 协议、评测与错误重叠分析工具、真实实验结果、原生运行时接口和测试。它还不是一个可直接安装的 macOS 输入法，也不宣称是“最好的开源中文输入法”。

## 架构

```text
Rime Top-8 候选 + 左侧上下文 + 拼音 + 候选元数据
                         |
                  紧凑残差打分模型
                         |
                 置信度门控 / 放弃
                         |
              对原 Top-8 候选做稳定重排
```

各学习模型共用确定性的汉字与拼音特征。Tiny-2M/4M/8M 使用小型 Transformer 编码器，MLP 是无注意力基线。修正后的代码使用 categorical vocabulary 表示候选类型。v0 检查点训练时还没有这一特征，因此正式复评使用明确的 `legacy_zero` 兼容模式，避免在不重新训练的情况下暗中改变模型输入。

## TinyRime-Context-v1 基准

数据来自 `fjcanyue/wikipedia-zh-cn` 的 2026-05-01 快照。流水线使用固定种子 `20260827` 扫描完整的 1,489,790 篇文档，并进行跨全文档级采样。所有文档先按 document ID 划分，再生成查询窗口。固定 query pool 包含 250,000 条训练查询、25,000 条验证查询和 25,000 条测试查询，三个 split 的文档互不重叠。

v1 修复了 pilot 阶段的 first-success 截断问题：

- candidate generation 覆盖每个完整 query split；
- val/test 保留全部可召回样本，分别为 22,066 和 22,080 条；
- 先生成全部 221,069 条可召回训练样本，再按稳定 SHA-256 优先级选择 100,000 条；
- `contested` 在完整固定 query pool 上统一定义并持久化；
- candidate miss 按 split 独立、确定性采样和诊断；
- candidate type 改为 categorical encoding；
- 完成 hash vocabulary collision audit：共观察到 8,499 个汉字与拼音 token，exact vocabulary 需要 8,501 个 embedding；32,768 桶的 hash 仍有 12.47% 的 unique-token collision，最高频 100 个 token 中有 26 个发生碰撞。

包含数据来源、上游版本和内容哈希的不可变清单位于 [`reports/TinyRime-Context-v1/benchmark_manifest.json`](reports/TinyRime-Context-v1/benchmark_manifest.json)。

## 实验结果

Candidate Recall 使用全部 25,000 条测试查询，包括无法召回的样本。排序指标使用 gold 位于原始 Rime Top-8 中的全部 22,080 条测试样本。

| Candidate generator | Recall@1 | Recall@3 | Recall@5 | Recall@8 | Recall@32 |
|---|---:|---:|---:|---:|---:|
| Rime + rime-ice | 73.704% | 85.068% | 87.396% | 88.320% | 88.868% |
| Rime + Wanxiang octagram | 78.148% | 86.396% | 88.112% | 88.808% | 89.232% |

| 排序方法 | Top-1 | Top-3 | MRR | Contested Top-1 |
|---|---:|---:|---:|---:|
| Rime | 83.451% | 96.318% | 0.90087 | 72.267% |
| MLP | 84.031% | 96.599% | 0.90461 | 73.411% |
| Tiny-2M | 85.951% | 96.952% | 0.91580 | 76.571% |
| Tiny-4M | 85.806% | 96.925% | 0.91493 | 76.355% |
| Tiny-8M | **86.621%** | **97.160%** | **0.92004** | **77.598%** |
| Wanxiang | **87.944%** | **97.251%** | **0.92738** | **80.083%** |

Tiny-8M 是当前最强的独立神经模型，但没有超过 Wanxiang：633 条测试样本只有 Tiny-8M 排对，925 条只有 Wanxiang 排对，净胜为 -292。两者的 Oracle hybrid 可以达到 90.811% Top-1。一个只在 val 上选择阈值的简单 hybrid 在 test 上达到 90.349%，相对 Wanxiang 有 697 wins、166 losses 和 +531 net wins。该结果说明 hybrid routing 值得继续研究，但还不是完成校准的产品规则。

Tiny-8M 有 7,430,338 个参数。经校验的 FP16 safetensors 文件大小为 14,865,276 bytes，SHA-256 为 `80da936a3e4616fbbb6172cbb37b208408101aab026cf484cb2f5187d288848a`；外部 Wanxiang grammar 文件为 420,250,668 bytes。由于训练权重的下游数据许可仍需单独审查，检查点只保留在本地研究环境中，不进入 Git，也不包含在公开版本里。

完整数字、结论边界和下一步判断见 [`reports/v0.1-release-report.md`](reports/v0.1-release-report.md)，资源对比见 [`reports/pareto-frontier.md`](reports/pareto-frontier.md)。

## 复现

先运行本地测试：

```bash
python -m pytest
cmake -S plugin -B build/plugin
cmake --build build/plugin
ctest --test-dir build/plugin --output-on-failure
```

完整 corpus、ranking dataset、candidate cache、外部 Wanxiang 模型和原始 checkpoint 均不会进入 Git。在已经按 [`DATA_LICENSES.md`](DATA_LICENSES.md) 准备好锁定上游依赖和数据文件的 AutoDL 数据盘上，可以运行：

```bash
bash scripts/remote/build_context_v1.sh ddba2f778f008813514368226f55a0e7a695c48d
bash scripts/remote/evaluate_context_v1.sh
```

任何 AutoDL → Mac 同步都必须先生成并检查包含文件大小与 SHA-256 的清单。禁止同步完整 corpus、完整 dataset、cache、checkpoint、原始日志或完整 prediction artifact。详细协议见 [`docs/benchmark-v1-protocol.md`](docs/benchmark-v1-protocol.md)，磁盘规则见 [`docs/data-and-disk-policy.md`](docs/data-and-disk-policy.md)。

## 安全约束

- 默认只重排 Top-8，不能引入候选列表之外的文本；
- 保守门控默认选择放弃重排；
- 打分同步执行，超过 deadline 时丢弃本次结果；
- 不包含网络请求、遥测、轮询、后台训练或用户历史上传路径；
- 用户词典、候选对象、注释、preedit 和引擎学习仍完全由 Rime 管理。

## 当前限制

- Benchmark 来自 Wikipedia，衡量的是离线上下文重排，不是真实用户输入体验；
- 当前结果只有一个训练种子和一个 100k 训练子集，不能说明 multi-seed 方差或 data scaling 趋势；
- 模型只能处理 Rime Top-8。即使合并 Rime/Wanxiang 候选，完整 query pool 上的 Union@32 Recall 也只有 89.260%；
- 当前 checkpoint 使用旧的 candidate-type 输入，categorical type 和 exact vocabulary 尚未完成受控训练 ablation；
- 训练模型的端到端延迟和 incremental RSS 尚未测量，现有 native microbenchmark 只是合成测试，不能作为产品证据；
- v0.1 不包含外部输入法 benchmark、Squirrel 系统集成、Core ML 产品优化或训练权重公开分发审查。

## 下一项研究问题

最值得继续验证的问题是：能否训练或校准一个 **Wanxiang → Tiny-8M residual router**，只在 Wanxiang 可能出错而 Tiny-8M 更可靠时提升候选，同时跨 seed 保持至少 95% promotion precision。现有 2.867 个百分点的 Oracle 空间和 val-selected hybrid 结果表明，routing 比继续扩大模型或数据更值得优先研究。v0.1 在启动该实验前停止。

## 致谢与借鉴

TinyRime 建立在开源输入法社区长期积累的工作之上，特别感谢：

- [rime/librime](https://github.com/rime/librime)：提供 Rime 核心引擎、候选接口和运行时契约，本项目的 headless candidate runner 基于其 API；
- [iDvel/rime-ice](https://github.com/iDvel/rime-ice)：提供本研究使用的 Rime 词典与配置基线；
- [amzxyz/RIME-LMDG](https://github.com/amzxyz/RIME-LMDG) 与 [lotem/librime-octagram](https://github.com/lotem/librime-octagram)：提供并支持 Wanxiang grammar 强基线；
- [rime/squirrel](https://github.com/rime/squirrel)：为 macOS 输入法生命周期和前端边界设计提供参考；
- [wyjrichhh/librime-ai-predict](https://github.com/wyjrichhh/librime-ai-predict)：其 filter、上下文和失败回退实现为运行时设计提供了参考；
- [shenmin/cassotis-ime](https://github.com/shenmin/cassotis-ime)：其保守式 residual ranking 思路为 TinyRime 的门控设计提供了借鉴；
- [fjcanyue/wikipedia-zh-cn](https://huggingface.co/datasets/fjcanyue/wikipedia-zh-cn) 与中文维基百科贡献者：提供本研究使用的公开中文语料来源。

研究过程中也阅读了 [cooelf/OpenIME](https://github.com/cooelf/OpenIME) 的论文与 benchmark 设计，用于了解已有研究版图；本项目没有使用其代码或数据。

以上项目及数据各自保留原许可证，TinyRime 的 BSD-3-Clause 许可证不会改变或覆盖它们。完整版本锁定与使用方式见 [`docs/upstream-lock.md`](docs/upstream-lock.md)、[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) 和 [`DATA_LICENSES.md`](DATA_LICENSES.md)。

TinyRime 自有源代码采用 BSD-3-Clause 许可证。
