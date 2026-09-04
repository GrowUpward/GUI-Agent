# 实验 002：UI-R1-136 基础能力预热

## 1. 目标

在不触碰 CAGUI Validation/Test 的前提下，用 UI-R1-low 的少量单步样本对 Qwen3.5-0.8B 进行 Stage 0 SFT，预热视觉定位、动作格式和基础动作判断能力。随后将该 LoRA adapter 作为 CAGUI 轨迹 SFT 的初始化，与“Base 直接进行 CAGUI SFT”做严格对照。

本阶段不声称学习多步规划；最终目标仍是提升 CAGUI 多步任务执行成功率。

## 2. 原始数据审计

服务器数据：

```text
/data/data1/zhaozhiqing/project/UI-R1/dataset/train_ground.json
/data/data1/zhaozhiqing/project/UI-R1/dataset/train_imgs
```

原始文件包含 136 条记录、115 张截图：

| 动作 | 数量 | Stage 0 处理 |
|---|---:|---|
| click | 101 | 保留；bbox 中心转换为归一化 `[x,y]` |
| open_app | 19 | 剔除；当前真机执行器不支持原子 `open_app` |
| navigate_back | 9 | 保留；映射为 `press_back` |
| scroll | 5 | 剔除；原始标签没有结构化方向参数 |
| input_text | 2 | 剔除；原始标签没有独立结构化文本参数 |

因此正式训练使用 110 条可无损转换样本。程序会把原始文件 SHA256、动作分布和全部剔除原因写入 `dataset_manifest.json`，禁止静默猜测缺失参数。

## 3. 坐标与目标协议

UI-R1 点击标签是像素坐标 `bbox=[x1,y1,x2,y2]`。Stage 0 使用框中心，并按原图尺寸归一化：

```text
x = round(((x1 + x2) / 2) / image_width  * 1000)
y = round(((y1 + y2) / 2) / image_height * 1000)
```

模型目标与 CAGUI 阶段完全一致：

```python
{'action': 'click', 'coordinate': [x, y]}
{'action': 'press_back'}
```

## 4. 训练配置

计划配置：

| 项目 | 值 |
|---|---|
| Base | Qwen3.5-0.8B |
| 方法 | 4-bit NF4 LoRA SFT |
| GPU | 2 卡 DDP |
| Epoch | 3 |
| 每卡 batch | 1 |
| 梯度累积 | 4 |
| 有效 batch | 8 |
| 学习率 | `1e-5` |
| 最大图像边长 | 448 |
| LoRA | rank 16, alpha 32, q/v projection |
| 验证集 | 无；本阶段只生成初始化 adapter |

启动命令：

```bash
cd /data/data1/zhaozhiqing/project/GUI-Agent

export PYTHON=/data/data0/zhaozhiqing/anaconda3/envs/python3.10/bin/python
export BASE_MODEL=/data/data0/zhaozhiqing/.cache/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export UI_R1_WARMUP_JSON=/data/data1/zhaozhiqing/project/UI-R1/dataset/train_ground.json
export UI_R1_WARMUP_IMAGES=/data/data1/zhaozhiqing/project/UI-R1/dataset/train_imgs
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export RUN_NAME=stage0-ui-r1-warmup-$(date +%Y%m%d-%H%M%S)
bash scripts/train_stage0_ui_r1.sh
```

## 5. 验收标准

1. 数据 manifest 显示 `raw_samples=136`、`usable_samples=110`。
2. 训练无 NaN、OOM 和 DDP 错误。
3. 最终目录包含 adapter、processor、`run_config.json`、`dataset_manifest.json` 和 `train_log.jsonl`。
4. Stage 1 只在固化的 CAGUI Train 上继续训练；方法选择只查看 CAGUI Validation。
5. 最终对照实验保持相同 CAGUI 训练预算：
   - E1：Base → CAGUI SFT；
   - E2：Base → UI-R1 Stage 0 → CAGUI SFT。

## 6. 结果

### 6.1 工程校验

- 代码提交：`fca051b feat: add UI-R1 Stage 0 SFT warmup`
- 完整测试：10 项通过；新增文件 Ruff 检查通过。
- 真实数据 dry-run：`raw_samples=136`、`usable_samples=110`。
- 2-step 双卡 smoke test：成功，34.46 秒，无 NaN/OOM。
- Smoke 目录：

```text
/data/data1/zhaozhiqing/project/GUI-Agent/artifacts/train/stage0-ui-r1-smoke-20260904
```

### 6.2 正式训练

正式运行目录：

```text
/data/data1/zhaozhiqing/project/GUI-Agent/artifacts/train/stage0-ui-r1-warmup-20260904-2330
```

结果：

| 指标 | 数值 |
|---|---:|
| Epoch | 3 |
| 优化 step | 42/42 |
| 训练耗时 | 909.9 秒（约 15 分 10 秒） |
| 每秒训练样本 | 0.363 |
| 每秒优化 step | 0.046 |
| 平均 train loss | 0.7929 |
| 首步 loss | 0.8480 |
| 末步 loss | 0.7621 |
| 最低单步 loss | 0.6608 |

训练退出码为 0，无 NaN、OOM、数据读取错误或 DDP 同步错误。最终 adapter 为约 2.56 MB；`safetensors` 中 24 个张量全部通过有限值校验。由于 `save_total_limit=2`，保留 `checkpoint-40`、`checkpoint-42` 及根目录最终 adapter。

训练结束时 PyTorch 输出“未显式调用 `destroy_process_group()`”的资源清理警告，但进程正常退出、GPU 显存已释放，不影响本次训练产物。

### 6.3 结论边界

本实验只证明 Stage 0 数据转换、双卡训练和 adapter 保存链路有效，不能仅凭训练 loss 宣称 CAGUI 能力提升。

下一步必须在相同的 CAGUI Train/Validation、训练预算和随机种子下比较：

```text
E1: Qwen3.5 Base -> CAGUI SFT
E2: Qwen3.5 Base -> UI-R1 Stage 0 adapter -> CAGUI SFT
```

只有 E2 在 CAGUI Validation 的动作、点击和参数指标上优于 E1，才能将 Stage 0 作为最终方案保留。
