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

### 7.1 训练结果

两组训练均正常完成，未出现 OOM、NaN 或无穷值；最终 adapter 的 24 个张量全部有限。

| 指标 | E3-Control | E3-Trajectory |
|---|---:|---:|
| 训练耗时 | 2230.9 s | 1952.5 s |
| 平均 train loss | 0.5076 | 0.5318 |
| Step 50 Validation loss | **0.5077** | 0.5203 |
| 最终 checkpoint | `checkpoint-50` | `checkpoint-50` |

运行目录：

```text
artifacts/train/e3-control-step-sft-50-20260905-1730
artifacts/train/e3-trajectory2-sft-50-20260905-1730
```

### 7.2 固定 Validation-128 结果

评测目录：

```text
artifacts/eval/e3-control-vs-trajectory-validation128-20260905-1830
```

| 指标 | E1 | E3-Control | E3-Trajectory |
|---|---:|---:|---:|
| 原始严格格式率 | 99.22% | **100.00%** | 96.09% |
| EOS 结束率 | 99.22% | **100.00%** | 96.09% |
| 多余内容率 | 0.78% | **0.00%** | 3.91% |
| 首动作提取成功率 | 100.00% | 100.00% | 100.00% |
| 动作类型准确率 | **82.81%** | 79.69% | 76.56% |
| 参数准确率 | 13.28% | 16.41% | **17.19%** |
| 平均 reward | **0.4897** | 0.4583 | 0.3943 |
| Click 命中率 | **29.35%** | **29.35%** | 21.74% |
| Click 平均距离（越低越好） | 257.46 | **222.61** | 251.38 |
| Input text 完全匹配率 | 52.94% | 52.94% | **58.82%** |
| 分动作宏平均准确率 | 43.77% | 47.03% | **48.08%** |

Validation-128 中没有 Scroll，宏平均按该子集出现的 Click、Input text、Stop、Wait、Impossible 五类等权计算。

分动作类型结果：

| Gold 动作 | 样本数 | E1 | E3-Control | E3-Trajectory |
|---|---:|---:|---:|---:|
| Click | 92 | **92.39%** | 83.70% | 77.17% |
| Input text | 17 | 76.47% | 76.47% | **88.24%** |
| Stop | 16 | 50.00% | **75.00%** | **75.00%** |
| Wait | 2 | 0.00% | 0.00% | 0.00% |
| Impossible | 1 | 0.00% | 0.00% | 0.00% |

### 7.3 对照分析

相对 E1，额外 50-step 单步 SFT 已经出现能力权衡：参数准确率提高 3.13 个百分点、Click 距离减少约 34.86，但动作准确率下降 3.12 个百分点、平均 reward 下降 0.0314。这说明结果不能简单归因于“多训练 50 step 会更好”。

在相同额外预算下，E3-Trajectory 相对 E3-Control：参数准确率提高 0.78 个百分点、Input text 完全匹配率提高 5.88 个百分点，但动作准确率下降 3.13 个百分点、平均 reward 下降 0.0640、Click 命中率下降 7.61 个百分点，原始严格格式率也下降 3.91 个百分点。

## 8. 当前结论

**当前二步轨迹 SFT 不能替代 E1，也不进入完整 457 条 Validation。** 它对 Input text 和 Stop 有正向信号，但占多数的 Click 明显退化，导致总体 reward 更差。

该结果仍受一项协议差异影响：E3-Trajectory 在包含连续截图和 assistant 动作的多轮上下文中训练，而本轮 Validation 使用当前部署端的单轮提示词加文本动作历史。因而当前结论准确表述为“轨迹模型不适合直接放入现有单轮部署协议”，还不能断言它在匹配的多轮推理上下文中同样无效。

下一步应先增加与训练格式一致的轨迹级 Validation evaluator，报告 episode 连续正确前缀和首次错误位置。如果匹配上下文后仍无法保住 Click，则停止当前轨迹方案；如果多步指标改善，再考虑让部署端维护多轮图片上下文。Test 继续封存。
