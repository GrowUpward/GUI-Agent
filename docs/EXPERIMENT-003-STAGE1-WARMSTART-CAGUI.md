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
export SKIP_FINAL_EVAL=1
export TEST_GENERATE_SAMPLES=0
bash scripts/train_sft.sh
```

`SKIP_FINAL_EVAL=1` 复用 step 300 已产生的 Validation 指标，避免训练结束后重复执行同一批 loss 评测；`TEST_GENERATE_SAMPLES=0` 禁止训练进程读取 Test 生成结果。完整 Validation 生成比较由独立评测进程完成。

## 6. Validation 生成评测

训练完成后，分别对 E1 和 E2 的 checkpoint-300 运行相同的完整 457 条 Validation 生成评测：

```bash
export EVAL_PARTITION=validation
export MAX_TEST_SAMPLES=0
bash scripts/eval_sft.sh
```

比较指标：格式成功率、动作类型准确率、参数准确率、点击命中率、点击距离、输入文本完全匹配率和各动作宏平均准确率。

## 7. 训练结果

E2 已于 2026-09-05 完成 300 step 训练，进程正常退出；训练期间未出现 OOM 或 NaN，最终 adapter 的 24 个张量均为有限值。

运行目录：

```text
artifacts/train/stage1-cagui-after-ui-r1-warmup-20260905-1221
```

| 指标 | E1：直接 CAGUI SFT | E2：UI-R1 预热后 CAGUI SFT | E2 相对 E1 |
|---|---:|---:|---:|
| 训练耗时 | 6719.7 s | 6646.6 s | -73.1 s |
| 平均 train loss | 0.582293 | **0.568318** | -0.013975（-2.40%） |
| Step 100 Validation loss | 0.572421 | **0.560471** | -0.011950（-2.09%） |
| Step 200 Validation loss | 0.545494 | **0.542738** | -0.002756（-0.51%） |
| Step 300 Validation loss | 0.536952 | **0.533921** | -0.003032（-0.56%） |

三次固定 Validation loss 上 E2 均优于 E1，说明 UI-R1 Stage 0 预热没有破坏 CAGUI 学习，并带来了小幅、一致的优化。不过，最终差距只有约 0.56%，仅凭 teacher-forcing loss 还不能证明多步执行能力得到提升，必须以完整生成指标和后续真机任务成功率作为主要结论依据。

另外，E2 训练结束后没有再次执行重复的 `trainer.evaluate()`，也没有在训练进程内生成 Test，说明本实验新增的评测解耦逻辑已生效。

## 8. 完整 Validation 生成结果

E1/E2 分别使用一张 GPU，对完全相同的 457 条 Validation 样本完成了独立生成评测。该阶段只使用 Validation，未读取 Test。E1 耗时 1895.8 秒，E2 耗时 1934.8 秒，两个进程均正常结束。

评测目录：

```text
artifacts/eval/e1-vs-e2-validation457-20260905-1432
```

| 指标 | E1 | E2 | 差值（E2 - E1） |
|---|---:|---:|---:|
| 解析成功率 | 88.84% | 88.84% | 0.00 pp |
| 格式有效率 | 88.84% | 88.84% | 0.00 pp |
| 动作类型准确率 | **73.09%** | 68.49% | **-4.60 pp** |
| 参数准确率 | 11.16% | **12.47%** | +1.31 pp |
| 平均 reward | **0.4013** | 0.3202 | -0.0811 |
| Click 命中率 | **19.82%** | 18.60% | -1.22 pp |
| Click 平均距离（越低越好） | 299.20 | **289.95** | -9.26 |
| Input text 完全匹配率 | **37.04%** | 33.33% | -3.71 pp |
| 分动作宏平均准确率 | 32.30% | **33.12%** | +0.82 pp |

宏平均按 Validation 中出现的六种 gold 动作类型等权计算。由于 `impossible`、`scroll` 和 `wait` 在两组中均为 0%，该指标容易被少量类别波动影响，不能替代总体动作准确率。

分动作类型结果：

| Gold 动作 | 样本数 | E1 | E2 | 差值（E2 - E1） |
|---|---:|---:|---:|---:|
| Click | 328 | **82.93%** | 74.09% | -8.84 pp |
| Input text | 54 | 57.41% | 57.41% | 0.00 pp |
| Stop | 58 | 53.45% | **67.24%** | +13.79 pp |
| Scroll | 11 | 0.00% | 0.00% | 0.00 pp |
| Wait | 4 | 0.00% | 0.00% | 0.00 pp |
| Impossible | 2 | 0.00% | 0.00% | 0.00 pp |

E2 的主要变化是更倾向输出 `stop`：正确 Stop 从 31 条增加到 39 条，但 Click 被预测为 Stop 的数量也从 15 条增加到 41 条。这造成占比最高的 Click 动作准确率下降 8.84 个百分点，进而使总体动作准确率和平均 reward 明显下降。参数准确率与 Click 平均距离的小幅改善不足以抵消动作选择退化。

## 9. 当前结论与决策规则

结论：**不采用 E2 作为下一阶段主基线，继续保留 E1。**

UI-R1 Stage 0 预热虽然将最终 Validation loss 降低约 0.56%，但没有转化为更好的整体生成行为：动作类型准确率下降 4.60 个百分点，平均 reward 下降 0.0811，Click 命中率和 Input text 完全匹配率也出现退化。因此，本实验不能将更低的 loss 表述为能力提升。

这次实验仍然有价值：它形成了一个清晰的负向消融结果，证明直接顺序微调会改变动作先验，并暴露出当前系统的三个关键问题：Click 占绝对多数、`scroll/wait/impossible` 完全没有学会、生成偶尔续写出多轮内容导致约 11.16% 的解析失败。

下一步应以 E1 为对照，先修复严格单动作生成和解析，再针对动作不均衡设计采样或 loss 权重消融；每次只改变一个变量，并继续只在固定 Validation 上选择方案。Test 保持封存，待方案固定后只运行一次，随后进入真机多步任务评测。
