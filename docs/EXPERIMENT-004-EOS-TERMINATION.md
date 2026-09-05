# 实验 004：EOS 监督与原始输出终止

## 1. 实验问题

验证在 SFT assistant 动作末尾加入模型原生 EOS，并让 EOS 参与 loss，能否使模型原始输出自然停止在一个动作之后，而不依赖部署端截取。

本实验不修改 CAGUI 原始 JSON。特殊 token 属于模型和 chat template 的实现细节，由训练 Collator 统一添加。

## 2. 已发现的问题

原 Collator 使用：

```text
prompt + action
```

监督序列末尾没有 Qwen3.5 的 `<|im_end|>`。这会训练模型生成动作内容，却没有显式训练“动作结束后停止”。此外，通用实现不应通过 token ID 屏蔽 padding，因为某些模型的 PAD 与 EOS 可能相同。

## 3. 修复方案

提交待实验完成后填写，改动包括：

1. 训练文本改为 `prompt + action + tokenizer.eos_token`；
2. 使用 `attention_mask == 0` 屏蔽 padding，确保 EOS 标签参与 loss；
3. 推理显式传入 `eos_token_id` 和 `pad_token_id`；
4. 同时保留 `raw_completion` 与首动作提取结果；
5. 新增原始严格格式率、EOS 结束率和多余内容率。

token 级检查已经确认，两条真实 CAGUI 样本的最后一个有效监督 token 均为 `<|im_end|>`，每条恰好包含一个受监督 EOS，padding 均被正确屏蔽。

## 4. 实验设计

该实验分为两个阶段：

### E1-raw：现有 E1 checkpoint

使用修正后的推理终止配置，在固定 Validation 前 128 条上测量原始输出指标，得到短程续训前基线。

### E3-short：E1 + 50-step EOS 续训

从 E1 checkpoint-300 初始化，在同一 CAGUI Train 划分上续训 50 step。该阶段仅验证 EOS 学习是否有效，不作为与 E1 公平比较的最终模型。

保持不变的主要参数：2 张 Quadro RTX 6000、BF16、4-bit NF4、每卡 batch 1、梯度累积 4、有效 batch 8、学习率 `1e-4`、LoRA rank 16/alpha 32、history window 4、最大图像边长 448。

## 5. 判定指标

| 指标 | 含义 | 目标 |
|---|---|---:|
| 原始严格格式率 | 原始输出仅包含一个合法动作 | 提高 |
| EOS 结束率 | 生成因模型输出 EOS 而结束 | 提高 |
| 多余内容率 | 首动作之后仍存在内容 | 降低 |
| 首动作提取成功率 | 部署兜底能否取得合法首动作 | 不下降 |
| 动作/参数准确率 | 输出内容是否正确 | 不明显下降 |

## 6. 结果

待 E1-raw 与 E3-short 评测完成后填写。

