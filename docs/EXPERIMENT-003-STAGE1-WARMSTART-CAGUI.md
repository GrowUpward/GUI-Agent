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

E1/E2 正在分别使用一张 GPU，对完全相同的 457 条 Validation 样本进行独立生成评测。该阶段只使用 Validation，不读取 Test；待两个进程均正常结束后，将在此处补充总体指标、分动作指标和结论。

评测目录：

```text
artifacts/eval/e1-vs-e2-validation457-20260905-1432
```

| 指标 | E1 | E2 | 差值（E2 - E1） |
|---|---:|---:|---:|
| 解析成功率 | 待填写 | 待填写 | 待填写 |
| 格式有效率 | 待填写 | 待填写 | 待填写 |
| 动作类型准确率 | 待填写 | 待填写 | 待填写 |
| 参数准确率 | 待填写 | 待填写 | 待填写 |
| 平均 reward | 待填写 | 待填写 | 待填写 |
| Click 命中率 | 待填写 | 待填写 | 待填写 |
| Click 平均距离 | 待填写 | 待填写 | 待填写 |
| Input text 完全匹配率 | 待填写 | 待填写 | 待填写 |
| 分动作宏平均准确率 | 待填写 | 待填写 | 待填写 |

## 9. 当前结论与决策规则

当前可以确认的是：Stage 0 预热对训练收敛有稳定但较小的正向作用。是否将 E2 作为下一阶段基线，按以下规则决定：

1. 如果 E2 的动作类型准确率、参数准确率和平均 reward 整体优于 E1，采用 E2；
2. 如果总体接近，但 E2 明显改善 Click 或 Input text 等核心基础动作，保留 E2 并进入轨迹级训练；
3. 如果 E2 的生成指标退化，则不以 loss 改善作为成功证据，回到 Stage 0 数据配比、动作格式或训练步数做消融；
4. Test 继续封存，待方案固定后只运行一次，随后进入真机多步任务评测。
