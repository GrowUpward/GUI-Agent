# 系统架构

## 1. 训练目标

模型根据当前屏幕截图、用户任务、近期动作历史和候选 UI 区域，只预测下一步结构化动作，不输出额外推理文本。

```text
CAGUI episode JSON + 截图
              |
              v
     CaguiEpisodeDataset
     - 按 episode 和 step 排序
     - CAGUI [y,x] -> 动作 [x,y]
     - 有界历史动作窗口
     - 统一结构化动作标签
              |
              v
     Qwen3.5-0.8B Processor
              |
      +-------+-------+
      |               |
     SFT             GRPO
  assistant-only    分组采样
      loss          结构化奖励
      |               |
      +-------+-------+
              v
       PEFT LoRA adapter
```

SFT 与 GRPO 共用动作转换、提示词构造、输出解析和奖励实现，减少两个训练阶段之间的格式漂移。

## 2. 数据边界

CAGUI 原始点击字段使用归一化的 `[y, x]`，模型、评测 API 和 ADB 执行端统一使用 `[x, y]`，坐标范围为 0–1000。

转换只允许发生在数据加载边界，并由回归测试固定该语义。坐标修复前训练的历史 adapter 可能仍生成 `[y, x]`，部署时可临时启用 `LEGACY_YX_OUTPUT=1` 交换坐标；新模型不得启用该兼容开关。

## 3. 动作空间

当前模型支持以下结构化动作：

- `click`：点击归一化坐标；
- `scroll`：向上、下、左或右滚动；
- `input_text`：输入指定文本；
- `press`：系统按键，按键参数为 `BACK`、`HOME` 或 `ENTER`；
- `wait`：等待固定或标注时长；
- `stop`：终止任务；CAGUI 的 impossible 源标注在训练边界归并为 stop。

CAGUI 的 `NO_ACTION` 映射为固定 500ms wait。当前精简动作空间不单列 long press，少量 `LONG_POINT` 源标注降级为 click，并保留点击位置监督。

模型输出采用单个 Python 字典形式，例如：

```python
{'action': 'click', 'coordinate': [520, 310]}
{'action': 'press', 'key': 'HOME'}
{'action': 'wait', 'duration': 500}
```

## 4. 奖励设计

当前奖励由格式、动作类型和动作参数三部分组成：

```text
R = 0.2 × format + 0.3 × action + 0.5 × arguments
```

不同动作使用不同的参数奖励：

- 点击/长按：归一化坐标距离与目标框命中；
- 文本输入：完全匹配和序列相似度；
- 滑动：起点和方向；
- 系统按键/任务状态：严格值匹配；
- 等待：时长范围合法性。

该奖励仍然来自离线标签。后续闭环版本应在真实或模拟环境中执行动作，并使用页面状态或任务成功检测提供奖励。

## 5. 部署链路

```text
Android 截图
   -> FastAPI 推理服务
   -> Python 字典动作
   -> 归一化坐标转像素坐标
   -> 人工确认
   -> ADB 执行
```

推理服务负责加载 Qwen3.5 基座模型和可选 LoRA adapter，并将模型输出解析为统一动作。ADB 端负责截图、发送请求、可视化预测点和执行动作。

当前模型和移动端链路尚未达到无人值守要求，因此人工确认是默认安全边界。
