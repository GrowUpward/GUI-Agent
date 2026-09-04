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
source /data/data1/zhaozhiqing/project/UI-R1/.venv/bin/activate  # 按服务器实际环境调整

export BASE_MODEL=/data/data1/zhaozhiqing/project/UI-R1/ckpt/Qwen/Qwen3.5-0.8B
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

待训练完成后补充运行目录、耗时、loss 曲线和 adapter 校验结果。
