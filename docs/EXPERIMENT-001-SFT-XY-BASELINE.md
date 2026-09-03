# EXPERIMENT-001：坐标修正版 SFT Baseline

## 1. 实验状态

- 状态：10-step BF16 冒烟实验已通过；300-step 正式 baseline 运行中
- 优先级：P0
- 训练阶段：SFT
- 后续阶段：只有 SFT baseline 可信且优于基座模型后，才进入 GRPO
- 代码版本：以 `run_config.json` 中记录的启动参数和对应 Git commit 为准

## 2. 研究问题

在修正 CAGUI `[y, x]` 到模型动作 `[x, y]` 的坐标转换后，Qwen3.5-0.8B 通过 LoRA/QLoRA SFT 能否稳定学习多种 GUI 动作，并形成一个可复现、可公平比较的 baseline？

本实验不验证 GRPO 收益，也不使用旧 adapter 继续训练，避免继承旧坐标语义。

## 3. 实验假设

1. 修正后的目标点击点与截图中的真实控件位置一致；
2. 从 Qwen3.5-0.8B 基座模型初始化新 LoRA，能够学习 CAGUI 的结构化动作格式；
3. SFT 后的动作类型准确率、文本输入准确率和点击指标应优于未微调基座模型；
4. 按 episode 划分数据可避免同一任务轨迹跨训练集和测试集泄漏。

## 4. 数据

数据路径：

```text
/data/data1/zhaozhiqing/project/UI-R1/dataset/CAGUI/CAGUI_agent/domestic
```

已检查数据规模：

| 项目 | 数量 |
| --- | ---: |
| Episode | 600 |
| Step | 4,516 |
| 点击 | 3,237 |
| 输入文本 | 574 |
| 任务完成 | 575 |
| 滑动 | 79 |
| 其他 | 51 |

## 5. 数据划分

采用 episode 级别划分，固定随机种子 `42`：

| 分区 | 比例 | 预计 Episode 数 | 用途 |
| --- | ---: | ---: | --- |
| Train | 80% | 480 | 参数更新 |
| Validation | 10% | 60 | 训练过程评估与 checkpoint 选择 |
| Test | 10% | 60 | 最终生成评测 |

每次运行在输出目录保存 `split_manifest.json`，其中记录三个分区的 episode ID、样本数、随机种子和坐标约定。最终实验必须复用同一 manifest。

## 6. 坐标约定

- CAGUI 原始数据：归一化 `[y, x]`；
- 模型标签、模型输出、评测与 API：归一化 `[x, y]`，范围 0–1000；
- ADB：由 API 将归一化 `[x, y]` 映射为截图像素坐标；
- `LEGACY_YX_OUTPUT=1` 仅用于部署旧 adapter，本实验禁止启用。

### 6.1 CAGUI 为什么使用 `[y, x]`

CAGUI 的 `[y, x]` 是有意采用的数据约定，不是原始标注错误。图像和张量通常按 `image[row, column]` 访问，即先指定纵向的 `y`，再指定横向的 `x`。CAGUI 因此将触摸点保存为：

```text
result_touch_yx = [y, x]
result_lift_yx  = [y, x]
```

候选 UI 区域也使用相同顺序：

```text
ui_positions = [y, x, height, width]
```

这种表示让触摸点和 UI 框保持统一，也便于直接与图像矩阵、检测结果和数据处理代码对齐。坐标使用 0–1 归一化值，从而与截图的实际分辨率解耦。CAGUI 官方数据格式和 AgentCPM-GUI 处理代码均明确保留这一约定：

- [CAGUI 数据格式](https://huggingface.co/datasets/openbmb/CAGUI)
- [AgentCPM-GUI 数据处理代码](https://github.com/OpenBMB/AgentCPM-GUI/blob/main/eval/eval_data/process_ac.py)

### 6.2 为什么必须转换成 `[x, y]`

本项目的模型动作、常规几何坐标和 Android ADB 均采用 `[x, y]`。例如 ADB 点击命令为：

```bash
adb shell input tap x y
```

因此，CAGUI 数据进入模型训练管线时必须交换坐标顺序。假设原始标注为：

```text
[y, x] = [0.20, 0.80]
```

它表示屏幕高度 20%、宽度 80% 的位置，正确的模型动作坐标应为：

```text
[x, y] = [800, 200]
```

旧实现只将两个值分别乘以 1000，没有交换顺序，错误地产生了 `[200, 800]`。这会把原本位于屏幕右上区域的目标映射到左下区域。

### 6.3 修正对象与影响范围

本次修正的对象是“CAGUI 数据格式与模型动作接口之间的转换”，而不是修改 CAGUI 原始数据。原始 JSON 和截图保持不变，只在数据加载边界执行一次：

```python
source_yx = [y, x]
model_xy = [round(x * 1000), round(y * 1000)]
```

旧坐标问题会影响：

- 点击与长按的训练标签；
- 目标框命中率和动作参数奖励；
- 模型输出映射到真实 ADB 点击位置时的端到端正确性。

动作类型判断和文本输入标签不依赖点击坐标，因此仍具有历史参考价值。平均坐标距离在预测和标签同时被交换时可能仍能衡量“对旧标签的拟合程度”，但不能证明真实屏幕位置正确。

修正后的实现统一保证：模型标签、模型输出、评测和部署接口均使用 `[x, y]`，并通过回归测试固定这一数据边界。由于旧 adapter 已经学习了历史坐标语义，只允许在演示旧模型时使用 `LEGACY_YX_OUTPUT=1` 临时交换输出；新的可信 baseline 必须从修正后的标签重新训练。

## 7. 模型与训练策略

- 基座模型：Qwen3.5-0.8B；
- 参数高效微调：LoRA；
- 量化：4-bit NF4；
- LoRA rank：16；
- LoRA alpha：32；
- LoRA dropout：0.05；
- 目标模块：`q_proj,v_proj`；
- 最大图像边长：448；
- 历史窗口：4；
- 最大候选 UI 框数：30；
- 注意力实现：SDPA；
- 数值精度：BF16。虽然服务器 GPU 为 Quadro RTX 6000（Turing），当前 PyTorch/CUDA 环境的 `torch.cuda.is_bf16_supported()` 返回 `True`，并且 BF16 探针梯度稳定；因此以实际环境检测和探针结果为准。

## 8. 阶段 A：10-step 冒烟实验

目的：验证数据加载、坐标转换、模型加载、反向传播、保存和评测链路可以完整运行。

| 参数 | 值 |
| --- | ---: |
| Episode 上限 | 30 |
| 最大训练步数 | 10 |
| 单卡 batch size | 1 |
| 梯度累积 | 4 |
| 学习率 | 1e-4 |
| 保存间隔 | 10 |
| 评测间隔 | 10 |
| GPU | 0 |

启动命令：

```bash
BF16=1 FP16=0 INIT_ADAPTER= \
RUN_NAME=sft-xy-smoke-$(date +%Y%m%d-%H%M) \
MAX_EPISODES=30 MAX_STEPS=10 \
SAVE_STEPS=10 EVAL_STEPS=10 \
BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=1e-4 MAX_IMAGE_SIDE=448 \
scripts/train_sft.sh
```

通过标准：

- 无 OOM、NaN 或数据读取异常；
- 能完成 10 个优化 step；
- 生成 `run_config.json`、`split_manifest.json`、训练日志和 checkpoint；
- 输出格式可被动作解析器读取；
- 随机抽样确认目标坐标方向正确。

### 已执行探针

第一次尝试使用 FP16，运行目录为：

```text
artifacts/train/sft-xy-smoke-20260903-095014
```

第 1 和第 5 step 均出现 `grad_norm=NaN`，loss scaler 跳过了有效更新，因此该运行被终止并保留为失败记录，不能用于模型效果判断。

第二次改用 BF16，运行目录为：

```text
artifacts/train/sft-xy-smoke-bf16-20260903-095315
```

10 个训练 step 已完成：第 10 step 的 loss 为 0.8629、梯度范数为 2.19，最终验证 loss 为 0.8632，`checkpoint-10` 和根目录 adapter 均已正常保存，未出现 NaN 或 OOM。batch size 1 时 GPU 峰值显存约 23.8 GB，接近单卡 24 GB 上限。

25 条测试样本的短程结果为：格式解析率 100%、动作类型准确率 80%、点击命中率 5%、平均点击距离 489.70、文本输入完全匹配率 0%、平均奖励 0.2894。模型把 25 条样本全部预测成点击动作，说明 10 step 只验证了工程链路，不能证明模型效果。

## 9. 阶段 B：300-step 正式 Baseline

只有阶段 A 通过后才启动。建议参数：

| 参数 | 值 |
| --- | ---: |
| Episode | 全部 600 |
| 最大训练步数 | 300 |
| GPU 数量 | 2 |
| 每卡 batch size | 1 |
| 梯度累积 | 4 |
| 有效 batch size | 8 |
| 学习率 | 1e-4 |
| 保存间隔 | 100 |
| 评测间隔 | 100 |
| 最大图像边长 | 448 |
| DataLoader workers（每进程） | 2 |

正式训练前必须先使用固定测试集评测未微调基座模型，从而建立真正可比较的 Base 指标。

第一次正式运行以单卡、梯度累积 8 启动：

```text
运行目录：artifacts/train/sft-xy-baseline-bf16-20260903-095945
进程 PID：1313029
训练集：480 episodes / 3,620 steps
原始验证集：60 episodes / 457 steps
原始测试集：60 episodes / 439 steps
训练步数：300
batch size：1
梯度累积：8
精度：BF16 + 4-bit NF4
```

第 1 step 的 loss 为 0.9006、梯度范数为 8.00，进程状态正常。训练期间的验证和训练结束时的在线生成评测暂时各限制为 256 条样本；最终报告前仍需基于 manifest 对完整 60-episode 测试集运行独立评测，并补齐未微调 Base 结果。

该运行在第 20 step 后主动终止，用作性能对照而非模型结果：平均约 39 秒/step，GPU 0 利用率多次采样仅为 24%–55%，GPU 1 空闲，且尚未到达第一个 checkpoint。已保留完整日志，不将该运行与最终 baseline 混用。

随后增加 `torch.distributed.run` 双卡 DDP 支持，并执行 5-step 探针：

```text
运行目录：artifacts/train/sft-xy-ddp-speed-probe-20260903-101612
GPU 数量：2
每卡 batch size：1
梯度累积：4
有效 batch size：8
平均速度：22.6 秒/step
最终 train loss：0.8575
结果：无 NaN、无 OOM，checkpoint 与根目录 adapter 均正常保存
```

双卡探针与单卡正式配置具有相同的有效 batch size，因而不改变优化步的样本规模。正式 baseline 改用 2 卡 DDP；验证与保存统一调整为每 100 step 一次，以减少短训练中的评测停顿，最终测试规模仍保持 256 条。

加速后的正式运行已启动：

```text
运行目录：artifacts/train/sft-xy-baseline-ddp-bf16-20260903-102135
进程 PID：1321369
GPU 数量：2
每卡 batch size：1
梯度累积：4
有效 batch size：8
精度：BF16 + 4-bit NF4
```

第 5 step 时平均速度约为 19.2 秒/step，loss 为 0.8624、梯度范数为 4.53；两卡显存采样约为 18–19 GB，未出现 NaN 或 OOM。按训练主体估算约 1.6 小时，计入三次验证和最终生成评测后，预计总耗时约 2.5–3 小时。

## 10. 评测指标

必须同时报告整体指标与各动作类别指标：

- 格式/解析成功率；
- 动作类型准确率；
- 动作参数准确率；
- 点击命中率；
- 平均点击距离；
- 输入文本完全匹配率；
- 每类动作准确率与混淆矩阵；
- 平均奖励；
- 典型成功与失败案例。

旧 adapter 的坐标相关指标只作为历史记录，不能作为修正版模型的直接对照。

## 11. 产物要求

每个实验输出目录必须包含：

```text
run_config.json
split_manifest.json
train_log.jsonl
metrics.json
adapter_config.json
adapter_model.safetensors
checkpoint-*/
```

实验结果不得写入 Git，只提交代码、实验方案和去除机器隐私后的指标摘要。

## 12. 停止条件

出现以下任一情况时停止并诊断，不直接扩大训练预算：

- loss 为 NaN 或持续发散；
- 连续出现 CUDA OOM；
- 格式解析率明显下降；
- 修正版坐标可视化仍与控件位置不一致；
- SFT 相比 Base 没有稳定收益；
- 日志、配置或数据划分无法复现。
