# 实验 008：CAGUI 离线多步轨迹评测

## 目的

在固定 Validation-60 上以 AgentCPM-GUI 兼容口径评测 E1 与 E2，同时保留更严格的参数诊断指标。该实验用于真机闭环前的离线多步模型选择，不读取固定 Test。

## 指标

- TM：动作类型与 GT 一致。
- EM：类型正确且参数满足 AgentCPM-GUI 判定。
- Episode Success：一条 episode 的全部 step 均为 EM。
- Goal Progress：首次 EM 错误前的连续正确 step 数除以 episode step 数，再按 episode 平均。
- Strict Click：预测点落入未经扩大的目标 UI bbox；无框时沿用项目严格距离规则。
- Strict Type：预测文本与标注文本完全一致。

官方兼容 Click EM 使用 1.2 倍目标 UI 框；若预测点未命中目标框（包括 GT 不属于任何 UI 框），继续使用归一化欧氏距离 `<=0.14` 兜底。官方兼容 Type EM 忽略大小写和首尾空格，并接受预测与 GT 互为子串。

## 对照

- E1：Base → CAGUI SFT。
- E2：Base → UI-R1 Stage0 → CAGUI SFT。
- 数据：固定 Validation 60 episodes / 457 steps。
- 推理：history window 4、最多 30 个 UI boxes、最长边 448、greedy decoding、显式 EOS。
- 模型加载：与既有 SFT 评测保持一致，使用 4-bit 加载；batch size 固定为 1，避免多模态可变长度 batch 改变生成结果。

## 评测有效性检查

首次运行遗漏了旧评测脚本默认启用的 4-bit 加载，导致模型几乎退化为只输出 Click。固定前 64 条样本的对照结果如下：

| 加载方式 | TM | EM（修正 Click 兜底前） | TYPE TM | STOP TM |
| --- | ---: | ---: | ---: | ---: |
| BF16 全精度 | 71.88% | 18.75% | 0/7 | 0/10 |
| 4-bit | 81.25% | 37.50% | 5/7 | 3/10 |

这说明加载精度是实质性实验变量。此前未使用 4-bit 的完整输出作废，不进入 E1/E2 结论；正式结果必须在元数据中记录 `load_in_4bit=true`、`batch_size=1`。

## 结果

正式评测产物：`artifacts/eval/cagui-offline-e1-vs-e2-validation457-4bit-corrected-20260906/`。

| 指标 | E1 | E2 | E2 - E1 |
| --- | ---: | ---: | ---: |
| TM | **83.15%** | 79.65% | -3.50 pp |
| EM | **43.11%** | 42.45% | -0.66 pp |
| Episode Success | **1/60（1.67%）** | 0/60（0%） | -1 episode |
| Goal Progress | **13.55%** | 10.55% | -3.00 pp |
| Strict Click | **25.91%** | 22.26% | -3.65 pp |
| Strict Type | 44.44% | **57.41%** | +12.97 pp |

按动作拆分的官方兼容 EM：

| 动作 | 样本数 | E1 EM | E2 EM |
| --- | ---: | ---: | ---: |
| Click | 328 | **39.33%** | 33.84% |
| Type | 54 | 68.52% | **81.48%** |
| Scroll | 11 | 0% | 0% |
| Stop（含 impossible/wait） | 64 | 48.44% | **60.94%** |

## 结论

1. E1 是当前离线轨迹主指标最好的 checkpoint：TM、EM、Episode Success 与 Goal Progress 均高于 E2，应作为下一阶段起点。
2. E2 明显增强了 Type 和 Stop，但 Click TM/EM 退化；由于 Click 占 328/457（71.77%），收益被主动作退化抵消。这不是“多步能力整体提升”。
3. 当前最大瓶颈是 Click 参数而不是输出格式：格式有效率均为 100%，但 E1 Strict Click 仅 25.91%，官方宽松 Click EM 也只有 39.33%。
4. Type 仍有优化空间，但 E2 已证明专项训练可以把官方 Type EM 提升到 81.48%、严格文本准确率提升到 57.41%。下一轮需要保留这部分收益，同时避免遗忘 Click。
5. Scroll 为 0/11，说明少数动作覆盖不足；Episode Success 的连乘效应使单步 EM 约 43% 时只能达到 1/60，符合预期，也说明真机前必须先提高每一步的参数正确率。

## 下一步

先做一个以 E1 为起点的 action-balanced CAGUI SFT：提高 Type/Scroll/Stop 采样权重，但保留足量 Click replay，并以 Validation 的 Click EM、Type EM、整体 EM 和 Goal Progress 联合早停。只有新 SFT 同时不降低 Click 且提高少数动作后，再基于同一 AgentCPM-compatible reward 做 GRPO；当前不宜直接把 E2 当作多步 GRPO 起点。
