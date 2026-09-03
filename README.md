# GUI-Agent-Qwen3.5

一个面向移动端 GUI 操作的轻量级多模态智能体项目。项目基于 Qwen3.5-0.8B，研究在两张 24 GB GPU 的资源约束下，能否通过分阶段 SFT 与 GRPO，将小参数量视觉语言模型从单步定位扩展到结构化的 CAGUI 动作预测。

## 项目定位

这是从早期 UI-R1 实验工作区中整理出的独立研究仓库。仓库只保留源码、配置模板、测试与可复现实验入口；数据集、模型权重、缓存、日志和历史 checkpoint 均不会提交到 Git。

本项目不是对 UI-R1 的简单复现。它以 UI-R1 的规则奖励强化学习思路为基础，重点完成 Qwen3.5 小模型适配、CAGUI 多动作建模、训练评测闭环和移动端执行原型。

## 当前状态

以下是坐标顺序审计前，旧版 SFT adapter 在固定 128 条 CAGUI 样本上的历史结果：

| 指标 | 结果 |
| --- | ---: |
| 格式/解析成功率 | 88.28% |
| 动作类型准确率 | 73.44% |
| 动作参数准确率 | 9.38% |
| 点击命中率 | 16.13% |
| 平均点击距离（0–1000 归一化坐标） | 275.10 |
| 输入文本完全匹配率 | 43.75% |
| 平均奖励 | 0.3317 |

仓库整理过程中发现：旧训练流程将 CAGUI 的 `[y, x]` 坐标直接标记成了模型动作所使用的 `[x, y]`。因此，动作类型和文本指标仍具有历史参考价值，但点击命中、参数奖励等坐标相关指标必须通过修正后的流程重新建立基线，不能直接作为最终结论。

当前推理服务可通过 `LEGACY_YX_OUTPUT=1` 兼容旧 adapter；新训练默认使用正确的 `[x, y]` 语义。

GRPO 训练代码和短程 checkpoint 已存在，但目前还没有完成严格控制变量的 SFT 与 SFT+GRPO 对比，暂不能声称 GRPO 带来了稳定提升。详细记录见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

当前坐标修正版 SFT baseline 的完整实验方案见 [`docs/EXPERIMENT-001-SFT-XY-BASELINE.md`](docs/EXPERIMENT-001-SFT-XY-BASELINE.md)。

## 本仓库的主要工作

- 将 Qwen3.5-0.8B 适配到 GUI 动作预测任务，并支持 LoRA/QLoRA；
- 按 episode 解析 CAGUI 轨迹，构造带历史动作的逐步预测样本；
- 修正并测试 CAGUI `[y, x]` 到动作 `[x, y]` 的坐标转换边界；
- 按 episode 划分训练集、验证集和测试集，避免同轨迹数据泄漏；
- 实现 CAGUI 桥接 SFT，以及基于生成结果的分动作评测；
- 实现格式、动作类型、参数三部分结构化奖励和轻量级 GRPO 循环；
- 提供环境诊断、数据统计、无模型 dry-run 和统一训练脚本；
- 提供 FastAPI 推理服务、ADB 执行器和 Gradio 调试界面原型。

上游项目和许可证归属见 [`NOTICE`](NOTICE)。

## 系统流程

```text
CAGUI 轨迹
  -> 带历史信息的逐步样本
  -> Qwen3.5-0.8B + QLoRA SFT
  -> 固定样本生成评测
  -> 分组采样 + 结构化奖励 + KL 约束 GRPO
  -> 指标与 checkpoint

屏幕截图 -> 推理服务 -> 结构化动作 -> 人工确认 -> ADB 执行
```

详细设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 仓库结构

```text
.
├── configs/                  # 环境配置模板，不保存密钥和机器路径
├── docs/                     # 架构设计、实验状态和结果说明
├── scripts/                  # 稳定的训练、检查和部署入口
├── src/gui_agent/
│   ├── data/                 # 数据统计与检查
│   ├── deployment/           # FastAPI、ADB 和 Gradio 原型
│   ├── training/             # 当前维护的 SFT 与 GRPO 实现
│   └── doctor.py             # Python、CUDA、依赖和路径诊断
└── tests/                    # 解析、奖励和数据集回归测试
```

## 快速开始

建议使用 Python 3.10 和支持 CUDA 的 PyTorch 环境。

```bash
cd /data/data1/zhaozhiqing/project/GUI-Agent
python -m pip install -e '.[dev,demo]'
cp configs/local.env.example configs/local.env
# 修改 configs/local.env 后执行：
set -a
. configs/local.env
set +a
gui-agent-doctor --strict
```

当前服务器可直接使用已有 Python 环境：

```bash
export PYTHON=/data/data0/zhaozhiqing/anaconda3/envs/python3.10/bin/python
$PYTHON -m pip install -e . --no-deps
```

`--no-deps` 可以避免安装过程意外替换当前 CUDA/PyTorch 依赖。当前环境已经满足训练依赖，但尚未安装 `pytest`、`fastapi`、`uvicorn` 和 `gradio`。训练和轻量 smoke test 可直接运行；使用完整测试或演示功能前，需要安装 `dev`/`demo` 可选依赖。

## 环境、数据和 dry-run 检查

```bash
scripts/doctor.sh --strict
scripts/test.sh
scripts/inspect_data.sh --json
scripts/dry_run.sh
```

`scripts/test.sh` 始终运行不依赖 pytest 的核心断言；如果环境中已安装 pytest，还会继续执行完整测试集。`scripts/dry_run.sh` 会读取少量真实 episode，检查动作转换和奖励计算，但不会加载模型权重。

如需检查移动端演示依赖：

```bash
scripts/doctor.sh --strict --demo
```

## SFT 训练

SFT 默认使用仓库内版本化的 `configs/splits/cagui-domestic-seed42-v1.json`，固定 CAGUI domestic 的 480/60/60 episode 划分。启动时会校验数据集是否与 manifest 一致；如需开展新的划分实验，应通过 `SPLIT_MANIFEST` 显式指定另一份版本化文件。

```bash
RUN_NAME=sft-cagui-$(date +%Y%m%d-%H%M) \
MAX_STEPS=300 \
LEARNING_RATE=1e-4 \
BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=2 \
MAX_IMAGE_SIDE=448 \
scripts/train_sft.sh
```

继续训练已有 adapter 时，通过 `INIT_ADAPTER` 指定路径，并为每次实验设置新的 `RUN_NAME`，避免覆盖历史结果。

## GRPO 训练

先运行确定性生成检查：

```bash
SFT_ADAPTER=/path/to/sft-adapter \
RUN_NAME=grpo-smoke-$(date +%Y%m%d-%H%M) \
scripts/train_grpo.sh --smoke_generate 30 --no_do_sample
```

再执行带日志的 10-step 探针实验：

```bash
SFT_ADAPTER=/path/to/sft-adapter \
RUN_NAME=grpo-probe-$(date +%Y%m%d-%H%M) \
MAX_STEPS=10 \
SAVE_STEPS=10 \
LOGGING_STEPS=1 \
NUM_GENERATIONS=4 \
scripts/train_grpo.sh
```

在扩大 GRPO 训练预算前，必须使用相同的数据划分、样本、随机种子、提示词、解码配置和指标实现，对 SFT 与 SFT+GRPO 进行公平对比。

## 移动端执行原型

推理服务支持加载基座模型和可选的 PEFT adapter：

```bash
MODEL_PATH="$BASE_MODEL" \
ADAPTER_PATH="$SFT_ADAPTER" \
scripts/serve_model.sh
```

只有使用坐标修复前训练的旧 adapter 时，才应设置 `LEGACY_YX_OUTPUT=1`；所有新训练模型均应保持该选项关闭。

ADB 控制器需要运行在能够访问 Android 真机或模拟器的机器上。默认情况下，每个预测动作都需要人工确认后才会执行。

## 已知限制

- 当前奖励来自离线标注动作，而不是执行后的环境状态或任务成功信号；
- CAGUI 中点击动作约占 71.7%，整体准确率可能掩盖少数动作的失败；
- 当前最佳模型的坐标和动作参数能力仍然较弱；
- 历史 adapter 存在坐标顺序问题，推理兼容开关不能替代重新训练和评测；
- 尚未通过严格对照实验证明 GRPO 的收益；
- 移动端链路仍是原型，完成闭环验证前不能声称已具备可靠的端到端任务成功率。

## 引用与许可证

本仓库采用 Apache-2.0 许可证，并保留对 UI-R1 的上游归属说明。如使用 UI-R1 的方法，请引用 [`NOTICE`](NOTICE) 中列出的 AAAI 2026 正式论文。
