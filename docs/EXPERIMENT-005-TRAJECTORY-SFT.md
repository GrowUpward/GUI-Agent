# 实验 005：CAGUI 二步轨迹 SFT

## 1. 实验问题

检验在相同额外训练预算下，将独立 next-action 样本改为包含连续截图与动作的多轮轨迹窗口，能否改善多步行为，同时不损害现有 E1 的单步动作与参数能力。

比较对象：

```text
E3-Control：E1 checkpoint-300 -> 50-step 单步 SFT
E3-Trajectory：E1 checkpoint-300 -> 50-step 二步轨迹 SFT
```

本实验是 Step 2 多步 SFT，不包含 GRPO。

## 2. 数据与泄漏控制

固定划分：`configs/splits/cagui-domestic-seed42-v1.json`

- 训练严格使用 480 个 Train episode、3620 个动作；
- Validation 只用于评测与模型选择；
- Test 保持封存；
- 轨迹窗口不跨 episode，也不会遗漏或重复动作。

3620 个 Train 动作被组成 1936 个非重叠窗口，每个窗口最多包含 2 个连续 step。episode 末尾允许一个单步窗口。

## 3. 轨迹样本格式

每个二步训练样本使用真实多轮 chat template：

```text
system
user: 截图 1 + 指令 + 当前状态
assistant: 动作 1 + EOS
user: 截图 2 + 指令 + 当前状态
assistant: 动作 2 + EOS
```

只对两个 assistant 动作及各自 EOS 计算 loss；system、user、图片 token 和 padding 均被 mask。

## 4. 控制变量

| 参数 | E3-Control | E3-Trajectory |
|---|---:|---:|
| 初始化 adapter | E1 checkpoint-300 | E1 checkpoint-300 |
| 训练 step | 50 | 50 |
| GPU | 1 | 1 |
| 每卡 batch | 1 | 1 |
| 梯度累积 | 8 | 4 |
| 每次更新监督动作数 | 约 8 | 约 8 |
| 学习率 | `2e-5` | `2e-5` |
| 精度 | BF16 + 4-bit NF4 | BF16 + 4-bit NF4 |
| LoRA | 复用 E1 rank 16/alpha 32 | 复用 E1 rank 16/alpha 32 |
| history window | 4 | 4 |
| 最大图像边长 | 448 | 448 |

两组并行运行，每组独占一张 Quadro RTX 6000。轨迹组通过减半梯度累积，使每次 optimizer update 监督的动作数量与单步组近似一致。

## 5. Smoke test

真实数据检查结果：

- 1936 个窗口包含且仅包含 3620 个 Train 动作；
- 窗口不跨 episode；
- 两个二步窗口组成的 batch 正确处理 4 张图片；
- 每个动作末尾恰好有一个受监督 EOS；
- 1-step 完整前向、反向、Validation loss 和 adapter 保存成功；
- 单步轨迹 smoke 耗时约 12.7 秒，没有 OOM 或 NaN。

## 6. 验收方式

训练完成后，先在固定 Validation 前 128 条比较动作、参数、reward、Click 命中率和原始格式；再增加 episode 级连续正确前缀、首次错误位置和整段动作正确率。只有轨迹组优于额外单步训练控制组，才进入完整 457 条 Validation。

## 7. 结果

待两组 50-step 训练与 Validation 评测完成后填写。

