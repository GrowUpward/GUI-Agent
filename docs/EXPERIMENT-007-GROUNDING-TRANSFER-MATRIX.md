# 实验 007：Grounding 迁移矩阵

## 目的

在完全相同的 ScreenSpot / ScreenSpotV2 评测协议下比较四个训练阶段，判断 UI-R1 grounding 是否有效，以及后续 CAGUI SFT 是否造成定位能力遗忘。

| 组别 | 权重 | 研究问题 |
| --- | --- | --- |
| Base | Qwen3.5-0.8B | 原始模型定位能力是多少？ |
| Stage0 | UI-R1 warm-up | UI-R1 是否直接改善外部分布定位？ |
| E1 | CAGUI SFT | 单独学习 CAGUI 后的定位能力（已有） |
| E2 | UI-R1 → CAGUI SFT | CAGUI SFT 是否保留 UI-R1 能力？ |

ScreenSpot 和 ScreenSpotV2 标签只用于离线评测，不进入训练。E1 复用实验 006 的完整结果；其余三组顺序运行，避免权重之间争抢 GPU。

## 决策规则

- Stage0 高于 E1、E2 下降：采用 UI-R1/CAGUI 混合 SFT 或 replay，解决灾难性遗忘。
- Stage0 本身没有提升：检查 UI-R1 数据转换、训练规模、分辨率和 LoRA 容量。
- E2 最优：以 E2 为起点进入后续多步训练。

## 结果

评测于 2026-09-06 完成。Base、Stage0 和 E2 依次运行，每组同时评测完整 ScreenSpot 与 ScreenSpotV2；每套数据均为 1272 条。运行目录：

```text
artifacts/eval/grounding-transfer-matrix-20260906
```

### 总体对照

| 权重 | ScreenSpot 格式 | ScreenSpot 动作 | ScreenSpot Grounding | V2 格式 | V2 动作 | V2 Grounding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 90.17% | 95.05% | 21.07%（268/1272） | 87.89% | 94.26% | 21.07%（268/1272） |
| Stage0 | 99.37% | 93.95% | 20.05%（255/1272） | 98.98% | 93.79% | 20.44%（260/1272） |
| E1 | **100.00%** | 99.76% | 21.46%（273/1272） | **100.00%** | **99.84%** | 22.09%（281/1272） |
| E2 | **100.00%** | **99.92%** | **22.48%（286/1272）** | **100.00%** | **99.84%** | **22.41%（285/1272）** |

这里的格式指未经截断后处理的原始严格格式率；动作指标表示是否输出 `click`，ScreenSpot 系列没有多动作 gold 标签；Grounding 是坐标落入 GT bbox 的比例，解析失败或坐标缺失均计错。

### 定位结果拆分

| 权重 | ScreenSpot Icon | ScreenSpot Text | V2 Icon | V2 Text |
| --- | ---: | ---: | ---: | ---: |
| Base | 13.91% | 26.97% | 15.52% | 25.35% |
| Stage0 | 13.22% | 25.68% | 15.16% | 24.51% |
| E1 | 14.96% | 26.83% | 15.34% | **27.30%** |
| E2 | **15.48%** | **28.26%** | **16.25%** | 27.16% |

### 增量分析

- Stage0 相对 Base：ScreenSpot `-1.02 pp`，ScreenSpotV2 `-0.63 pp`。它将严格格式提高约 9–11 个百分点，但没有提升外部分布定位。
- E1 相对 Base：ScreenSpot `+0.39 pp`，ScreenSpotV2 `+1.02 pp`，主要收益仍是格式与动作协议。
- E2 相对 E1：ScreenSpot `+1.02 pp`，ScreenSpotV2 `+0.31 pp`；E2 是两套外部测试上的最佳权重，但提升很小。
- E2 相对 Base：ScreenSpot `+1.42 pp`，ScreenSpotV2 `+1.34 pp`。说明两阶段训练产生了微弱正向迁移，但不足以证明 110 条 UI-R1 Stage0 数据已经建立通用 grounding 能力。

## 结论与决策

实验结果不支持“UI-R1 Stage0 单独改善通用定位”的假设。Stage0 只有 110 条可无损转换训练样本、42 个优化 step，更像格式预热；它显著改善输出规范，却没有提高 ScreenSpot bbox 命中率。因此，不能把 Stage0 的负增量解释为 UI-R1 方法本身无效，更准确的结论是当前数据量和训练设置不足以验证该方法。

E2 在两个 ScreenSpot benchmark 上最好，但此前固定 CAGUI Validation 表明 E2 相比 E1 的动作类型准确率下降 3.50 个百分点、Click 命中率下降 3.65 个百分点、平均 reward 下降 0.0631。故 E2 也不能直接替代 E1 作为总体最佳权重：E1 更适合当前 CAGUI 多动作策略，E2 只在外部单步 grounding 上略优。

下一步暂不进入 GRPO。优先建立规模更大的 grounding 训练集，并进行一次受控混合 SFT：CAGUI 保持 episode 固定划分，grounding 样本只加入训练分区，ScreenSpot/V2 严格保留为外部测试；训练时保留 CAGUI replay，避免只优化 click 格式而破坏 `input_text/stop/scroll/wait`。候选方案必须同时超过 E1 的 CAGUI Validation 指标和 E2 的 ScreenSpot 指标，才进入多步 GRPO。

## 结果文件 SHA256

```text
Base ScreenSpot:      6e0be2cfb36d8279020f59c6795f774917624b6fb125825f396a19e2b7540196
Base ScreenSpotV2:    fbf02a35dc462c27ed2fa00eb553b895778005a13afeb6b2b971ed413e2f8af2
Stage0 ScreenSpot:    f8b752a5bebe53511a156175abb59b17b78a7d63a1134c224f8faf177e0a9298
Stage0 ScreenSpotV2:  cef2500f2e25f2dfedd8d2f7c316f68e8368f9151a8ee20476091b8eefc05748
E2 ScreenSpot:        c73d323e4fa21f4b9b04a69038dccff4121a7d6c1be6c8608410671224b1b0a2
E2 ScreenSpotV2:      a8aafa3cb136017ed4f06526f8f13b40e333ed4d9d3276c2aa8ab6aabebabb47
```
