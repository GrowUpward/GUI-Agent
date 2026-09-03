# EXPERIMENT-001：坐标修正版 SFT Baseline

## 1. 实验状态

- 状态：10-step BF16 训练与 checkpoint 保存已通过，最终测试生成进行中
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

10 个训练 step 已完成：第 10 step 的 loss 为 0.8629、梯度范数为 2.19，最终验证 loss 为 0.8632，`checkpoint-10` 已正常保存，未出现 NaN 或 OOM。batch size 1 时 GPU 峰值显存约 23.8 GB，接近单卡 24 GB 上限。最终是否完整通过，还需等待 25 条测试样本的生成评测和根目录 adapter 保存完成。

## 9. 阶段 B：300-step 正式 Baseline

只有阶段 A 通过后才启动。建议参数：

| 参数 | 值 |
| --- | ---: |
| Episode | 全部 600 |
| 最大训练步数 | 300 |
| 单卡 batch size | 1 |
| 梯度累积 | 8 |
| 有效 batch size | 8 |
| 学习率 | 1e-4 |
| 保存间隔 | 100 |
| 评测间隔 | 50 |
| 最大图像边长 | 448 |

正式训练前必须先使用固定测试集评测未微调基座模型，从而建立真正可比较的 Base 指标。

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
