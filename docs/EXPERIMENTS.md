# GUI-Agent 实验总览

> 更新时间：2026-09-07
>
> 本页是实验状态的唯一总入口；各实验文档保留当时的配置、过程与完整结果。

## 1. 当前处于哪个阶段

项目目标是提高 CAGUI 多步任务执行成功率，最终在 Android 真机闭环验证。当前已经完成坐标修复、基础 SFT、UI-R1 预热消融、多步离线评测、外部 grounding 评测以及输入分辨率消融。

**当前阶段：Stage 1 基线与消融已完成，准备按统一动作协议重新训练正式 E1。尚未进入 GRPO，也尚未完成真机闭环评测。**

```text
[已完成] 数据/坐标审计
    ↓
[已完成] E1：Base → CAGUI SFT baseline
    ↓
[已完成] UI-R1 预热、二步轨迹、EOS、外部 grounding、分辨率消融
    ↓
[当前]   用统一动作协议重训 E1，并解决参数与长尾动作问题
    ↓
[待完成] 固定 Test 最终评测
    ↓
[待完成] Android 真机闭环与多步任务成功率
    ↓
[条件项] 有可靠环境奖励后再做 GRPO
```

当前主线只保留 **E1 路线（Base → CAGUI）**。E2（UI-R1 → CAGUI）保留为已完成消融，不再作为后续主模型。

## 2. 实验矩阵

| 编号 | 实验 | 状态 | 解决的问题 | 关键结论 / 决策 |
| --- | --- | --- | --- | --- |
| [001](EXPERIMENT-001-SFT-XY-BASELINE.md) | 坐标修正版 SFT Baseline | ✅ 完成 | 坐标语义修复后，CAGUI SFT 是否有效 | checkpoint-300 在 Test-439 显著超过 Base；证明 SFT 有效，但点击和长尾动作仍弱 |
| [002](EXPERIMENT-002-STAGE0-UI-R1-WARMUP.md) | UI-R1-136 基础能力预热 | ✅ 完成 | 少量 UI-R1 数据能否作为定位/格式预热 | 110 条可用样本完成 3 epoch；只能作为初始化，不能承担多步训练 |
| [003](EXPERIMENT-003-STAGE1-WARMSTART-CAGUI.md) | UI-R1 预热后 CAGUI SFT（E2） | ✅ 完成，停止扩展 | E2 是否优于直接 CAGUI SFT 的 E1 | E2 改善 Type/Stop 和外部 grounding，但损害 CAGUI Click 与整体轨迹；后续选择 E1 |
| [004](EXPERIMENT-004-EOS-TERMINATION.md) | EOS 与原始输出终止 | ✅ 完成 | 是否需要额外 EOS 训练 | E1 原始严格格式率和 EOS 结束率均为 99.22%；无需为 EOS 单独续训 |
| [005](EXPERIMENT-005-TRAJECTORY-SFT.md) | CAGUI 二步轨迹 SFT | ✅ 完成，当前方案不采用 | 两步多轮训练是否优于等预算单步续训 | 参数/Type 有局部收益，但 Click 与总体 reward 下降；训练/部署上下文不匹配，不能作为主模型 |
| [006](EXPERIMENT-006-SCREENSPOT-BENCHMARK.md) | ScreenSpot / V2 评测 | ✅ 完成，旁路指标 | 当前模型的外部分布单步 grounding 能力 | Grounding 仅 21.46% / 22.09%；只作诊断，不作为项目最终目标 |
| [007](EXPERIMENT-007-GROUNDING-TRANSFER-MATRIX.md) | Grounding 迁移矩阵 | ✅ 完成 | Base、Stage0、E1、E2 的迁移效果 | Stage0 单独没有提升 grounding；E2 外部评测略好，但不能替代 E1 |
| [008](EXPERIMENT-008-CAGUI-OFFLINE-TRAJECTORY-EVAL.md) | CAGUI 离线多步轨迹评测 | ✅ 完成 | 建立 Goal Progress / Episode Success 选型指标 | 评测协议已建立；确认 E1 优于 E2，并发现 4-bit 加载是实质变量 |
| [009](EXPERIMENT-009-RESOLUTION-ABLATION.md) | 448 / 672 输入分辨率消融 | ✅ 完成 | 更高分辨率能否改善点击参数与轨迹完成度 | checkpoint-1050@672 提升 Strict Click 与 Goal Progress，但 Type/Stop 下降；672 是候选配置，不是无条件替换 |
| 010 | 统一动作协议下的 E1 重训 | ⏳ 下一项 | 让训练、评测和部署使用同一动作空间 | 尚未启动；这是进入最终 Test 和真机前的必要实验 |
| 011 | Android 真机多步闭环 | ⬜ 未开始 | 测量真实 Task Success Rate 和错误恢复 | 最终项目验收；离线 Episode Success 不能替代该结果 |
| 012 | CAGUI/真机 GRPO | ⏸ 条件项 | 直接优化轨迹级成功奖励 | 只有闭环环境、奖励与 SFT baseline 稳定后才启动 |

状态说明：✅ 已完成；⏳ 下一项；⬜ 未开始；⏸ 前置条件未满足。

## 3. 当前最佳结果

当前综合表现最好的旧协议权重是：

```text
artifacts/train/e1-cagui-3epoch-total-448-2gpu-20260906/checkpoint-1050
```

固定 CAGUI Validation（60 episodes / 457 steps，history=4，4-bit，batch=1，greedy）结果：

| 指标 | checkpoint-1050 @448 | checkpoint-1050 @672 | 当前判断 |
| --- | ---: | ---: | --- |
| TM / 动作类型匹配 | **89.06%** | 87.31% | 448 更好 |
| EM / 动作与参数匹配 | 59.30% | **59.96%** | 672 小幅提高 |
| Click EM | 53.96% | **57.32%** | 672 提高 3.36 pp |
| Strict Click | 42.07% | **45.43%** | 672 提高 3.36 pp |
| Input Text EM | **85.19%** | 75.93% | 672 明显下降 |
| Strict Type | **62.96%** | 57.41% | 448 更好 |
| Stop EM | **80.00%** | 75.00% | 448 更好 |
| Goal Progress | 22.93% | **32.74%** | 672 提高 9.81 pp |
| Episode Success | 4/60 | **5/60** | 672 多完成 1 个 episode |
| Scroll / Long Press EM | 0% | 0% | 两者均未学会 |

结论：672 对当前最关键的点击定位和正确轨迹前缀有明显帮助，但存在 Type/Stop 能力退化。后续正式训练应把 448、672 或混合分辨率作为单一控制变量，并依据 Strict Click、Strict Type、Goal Progress、Episode Success 联合选型，不能只看总体 EM。

## 4. 已确认的关键结论

### 4.1 SFT 有效，但项目还没有达到最终展示状态

在固定 Test-439 上，坐标修正后的 checkpoint-300 相比 Base：格式率从 52.62% 提升到 90.43%，动作类型准确率从 26.88% 提升到 75.63%，点击命中率从 2.88% 提升到 21.47%。这证明 CAGUI SFT 确实学到了能力，而不是只有 loss 下降。

但初始 baseline 的 Click 命中率仍低，Scroll/Wait 等长尾动作接近 0；最新 checkpoint 虽继续改善，离线 Episode Success 仍只有 5/60。因此当前结果适合证明工程与实验能力，不应包装为已经解决 GUI 多步执行。

### 4.2 E1 是主线，E2 只保留为消融

E2 在 Input Text、Stop 和 ScreenSpot grounding 上有局部优势，但在 CAGUI 的 TM、Strict Click、Goal Progress 和 Episode Success 上不如 E1。后续不再增加 E2 训练预算。

### 4.3 输出格式与 EOS 已基本解决

显式 EOS 后，E1 在固定样本上的原始严格格式率与 EOS 结束率达到 99.22%；当前 checkpoint-1050 的完整 Validation 格式率和 EOS 结束率达到 100%。因此下一阶段的主要矛盾不是解析器，而是动作参数、长尾动作和多步决策。

### 4.4 离线多步指标不是闭环成功率

当前 CAGUI evaluator 使用数据集中每一步的真实截图进行离线预测，并计算连续正确前缀。它能用于 checkpoint 选择，但模型的错误不会真正改变下一张截图，因而不等价于 Android 环境中的闭环 Task Success Rate。最终结论必须来自真机或可复现模拟器。

### 4.5 ScreenSpot 不是最终目标

ScreenSpot 与 ScreenSpotV2 仅用于观察单步 grounding 的迁移能力。E1 分别取得 21.46% 和 22.09% Grounding Accuracy；E2 略高到 22.48% 和 22.41%，但这种小幅外部分布收益不足以抵消其 CAGUI 多动作退化。

## 5. 数据、动作空间与评测口径

### 5.1 固定数据划分

| 分区 | Episode | Step | 用途 |
| --- | ---: | ---: | --- |
| Train | 480 | 3,620 | 训练与重采样 |
| Validation | 60 | 457 | checkpoint、分辨率和训练方案选择 |
| Test | 60 | 439 | 最终方案确定后运行 |

版本化 manifest：`configs/splits/cagui-domestic-seed42-v1.json`。后续不得通过 Test 反复选择 checkpoint 或超参数。

### 5.2 统一训练动作协议

当前协议定义 7 类动作：Click、Long Press、Scroll、Input Text、Press、Wait、Stop。详细字段和映射以 [训练动作空间协议](TRAINING-ACTION-SPACE.md) 为唯一规范。

需要特别区分：

- `long_press` 不能降级为 `click`，必须保留坐标与 `duration`；
- `PRESS_BACK/HOME/ENTER` 统一为 `press(key=...)`；
- `wait` 使用 `duration=500`；
- `impossible` 不作为独立动作，数据边界归并为 `stop`；
- 模型每次只输出一个动作字典，并以 EOS 结束。

CAGUI domestic 的转换后分布仍高度不均衡：Click 3,237，Input Text 574，Stop 600，Scroll 79，Long Press 25，Wait 1，Press 0。总体准确率很容易被 Click 主导，因此必须报告分类指标。

### 5.3 协议版本边界

统一动作协议自提交 `d45c080` 后生效。现有 checkpoint-1050 是协议统一前训练的旧基线；它的结果仍可作为历史对照，但不能当作已经学会新版 `long_press/press/wait/stop` 定义的最终权重。

因此实验 010 必须在新版数据转换与输出协议下重新 SFT。最干净的主对照是继续沿用 E1 路线：Qwen3.5-0.8B Base → CAGUI Train，而不是从 E2 开始。

## 6. 已作废或仅供参考的结果

- 坐标 `[y,x]` 未转换为模型 `[x,y]` 时产生的点击参数指标：作废；格式和动作类型只能作为历史参考。
- CAGUI 离线评测中未启用 4-bit 加载的完整输出：作废；该变量导致模型明显退化。
- 修复“只取首动作”之前的解析率：仅用于说明评测器问题，不参与模型选型。
- ScreenSpot/V2：仅作旁路 grounding 诊断，不用于选择最终多步模型。
- 旧 10-step GRPO checkpoint：缺少完整日志与自包含指标，只能证明训练循环跑通，不能证明 GRPO 有效。

## 7. 下一步执行顺序

### P0：实验 010——统一协议重训 E1

1. 基于新版转换器重新生成并审计 CAGUI Train/Validation；
2. 确认 7 类动作格式、坐标顺序、EOS 和数据计数；
3. 从 Base 按 E1 路线训练，保留完整日志与 checkpoint；
4. 首轮以 448 为可比基线，再单独比较 672 或混合分辨率；
5. 对 Scroll、Long Press、Wait 做过采样或 loss 权重消融，同时监控 Click、Type、Stop 退化。

验收指标：格式率、TM、EM、Strict Click、Strict Type、各动作 EM、Goal Progress、Episode Success。必须同时给出总体与分类结果。

### P1：固定 Validation 选型

只在 Validation-457 比较新版 E1 的 checkpoint、分辨率与类别均衡策略。候选模型至少应超过旧 checkpoint-1050 的关键基准：Strict Click 45.43%、Goal Progress 32.74%、Episode Success 5/60，同时不能明显牺牲 Type/Stop。

### P2：一次性 Test 与真机闭环

方案固定后运行 Test-439，随后接入 Android 真机，报告 Task Success Rate、平均完成步数、非法动作率、首次错误位置与错误恢复率。真机结果才是秋招项目的最终核心指标。

### P3：条件式 GRPO

只有当真机/模拟器环境能稳定执行动作、返回新截图并给出可靠任务级奖励时，才在最佳新版 E1 上进行 GRPO。GRPO 必须与同预算 SFT 续训做控制变量对比。

## 8. 主要产物

| 内容 | 路径 |
| --- | --- |
| 固定数据划分 | `configs/splits/cagui-domestic-seed42-v1.json` |
| 动作空间规范 | `docs/TRAINING-ACTION-SPACE.md` |
| 初始 SFT baseline | `artifacts/train/sft-xy-baseline-ddp-bf16-20260903-102135/checkpoint-300` |
| 当前旧协议最佳权重 | `artifacts/train/e1-cagui-3epoch-total-448-2gpu-20260906/checkpoint-1050` |
| Base vs SFT Test-439 | `artifacts/eval/base-vs-sft-test439-20260903-152122/` |
| checkpoint-600 vs 1050 | `artifacts/eval/cagui-e1-longtrain-cp600-vs-cp1050-validation457-20260907/` |
| 448/672 分辨率结果 | `artifacts/eval/e1-checkpoint1050-resolution672-validation457-20260907/` |
| ScreenSpot 迁移矩阵 | `artifacts/eval/grounding-transfer-matrix-20260906/` |

## 9. 文档索引

- [001：坐标修正版 SFT Baseline](EXPERIMENT-001-SFT-XY-BASELINE.md)
- [002：UI-R1-136 基础能力预热](EXPERIMENT-002-STAGE0-UI-R1-WARMUP.md)
- [003：UI-R1 预热后 CAGUI SFT](EXPERIMENT-003-STAGE1-WARMSTART-CAGUI.md)
- [004：EOS 监督与原始输出终止](EXPERIMENT-004-EOS-TERMINATION.md)
- [005：CAGUI 二步轨迹 SFT](EXPERIMENT-005-TRAJECTORY-SFT.md)
- [006：ScreenSpot / ScreenSpotV2](EXPERIMENT-006-SCREENSPOT-BENCHMARK.md)
- [007：Grounding 迁移矩阵](EXPERIMENT-007-GROUNDING-TRANSFER-MATRIX.md)
- [008：CAGUI 离线多步轨迹评测](EXPERIMENT-008-CAGUI-OFFLINE-TRAJECTORY-EVAL.md)
- [009：CAGUI 输入分辨率消融](EXPERIMENT-009-RESOLUTION-ABLATION.md)
