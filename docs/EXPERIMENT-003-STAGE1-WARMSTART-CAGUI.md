# 实验 003：Stage 0 预热后进行 CAGUI SFT

## 1. 实验问题

检验 UI-R1-low Stage 0 单步预热是否能改善后续 CAGUI 轨迹学习。比较对象：

```text
E1：Qwen3.5-0.8B Base -> CAGUI SFT
E2：Qwen3.5-0.8B Base -> UI-R1 Stage 0 -> CAGUI SFT
```

方法选择只使用固定 Validation；Test 留到方案固定后再做一次最终报告。

## 2. 数据一致性

固定划分：`configs/splits/cagui-domestic-seed42-v1.json`

| 分区 | Episode | Step sample |
|---|---:|---:|
| Train | 480 | 3620 |
| Validation | 60 | 457 |
| Test | 60 | 439 |

E1 运行时生成的旧 manifest 与版本化 manifest 的文件 SHA256 不同，是因为记录字段不同；三个分区的 episode ID 集合和样本数量逐项完全相同，因此 E1/E2 可直接比较。

## 3. 控制变量

E2 除初始化 adapter 外完全复用 E1 配置：

| 参数 | E1/E2 |
|---|---:|
| Base model | Qwen3.5-0.8B |
| 训练 step | 300 |
| GPU | 2 × Quadro RTX 6000 |
| 精度 | BF16 + 4-bit NF4 |
| 每卡 batch | 1 |
| 梯度累积 | 4 |
| 有效 batch | 8 |
| 学习率 | `1e-4` |
| warmup ratio | 0.03 |
| LoRA | rank 16, alpha 32, q/v projection |
| history window | 4 |
| 最大图像边长 | 448 |
| Validation loss 样本 | 固定前 256 条 |
| eval/save 间隔 | 100 step |

唯一实验变量：

- E1 从新 LoRA 初始化；
- E2 从 `artifacts/train/stage0-ui-r1-warmup-20260904-2330` 初始化同结构 LoRA。

## 4. E1 已有结果

运行目录：

```text
artifacts/train/sft-xy-baseline-ddp-bf16-20260903-102135
```

| Step | Validation loss |
|---:|---:|
| 100 | 0.57242 |
| 200 | 0.54549 |
| 300 | 0.53695 |

训练耗时约 6719.7 秒，平均 train loss 0.5823。

## 5. E2 启动命令

```bash
cd /data/data1/zhaozhiqing/project/GUI-Agent

export PYTHON=/data/data0/zhaozhiqing/anaconda3/envs/python3.10/bin/python
export BASE_MODEL=/data/data0/zhaozhiqing/.cache/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export CAGUI_ROOT=/data/data1/zhaozhiqing/project/UI-R1/dataset/CAGUI
export INIT_ADAPTER=/data/data1/zhaozhiqing/project/GUI-Agent/artifacts/train/stage0-ui-r1-warmup-20260904-2330
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export RUN_NAME=stage1-cagui-after-ui-r1-warmup-20260905
export MAX_STEPS=300
export LEARNING_RATE=1e-4
export GRADIENT_ACCUMULATION_STEPS=4
export MAX_EVAL_SAMPLES=256
export MAX_TEST_SAMPLES=256
export EVAL_STEPS=100
export SAVE_STEPS=100
export DATALOADER_NUM_WORKERS=2
bash scripts/train_sft.sh
```

## 6. Validation 生成评测

训练完成后，分别对 E1 和 E2 的 checkpoint-300 运行相同的完整 457 条 Validation 生成评测：

```bash
export EVAL_PARTITION=validation
export MAX_TEST_SAMPLES=0
bash scripts/eval_sft.sh
```

比较指标：格式成功率、动作类型准确率、参数准确率、点击命中率、点击距离、输入文本完全匹配率和各动作宏平均准确率。

## 7. 结果

待 E2 训练与 E1/E2 Validation 评测完成后填写。
