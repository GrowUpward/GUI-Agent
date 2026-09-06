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

待评测完成后写入。
