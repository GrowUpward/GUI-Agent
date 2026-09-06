# 训练动作空间协议

本文档是 GUI-Agent 当前训练输出协议的唯一规范。数据转换、SFT、GRPO、离线评测及后续新增数据必须使用相同的动作名称、字段和参数语义。

## 动作定义

| 动作类型 | 标准训练输出 | 参数约束 | 语义 |
| --- | --- | --- | --- |
| Click | `{'action':'click','coordinate':[x,y]}` | `x,y` 为 `[0,1000]` 内整数 | 点击指定位置 |
| Long Press | `{'action':'long_press','coordinate':[x,y],'duration':1000}` | 坐标同 Click；`duration` 单位为 ms，使用标注值 | 在指定位置持续按压 |
| Scroll | `{'action':'scroll','direction':'up'}` | `direction ∈ {up,down,left,right}` | 沿指定方向滚动 |
| Input Text | `{'action':'input_text','text':'内容'}` | `text` 为字符串，保留标注内容 | 向当前输入框输入文本 |
| Press | `{'action':'press','key':'HOME'}` | `key ∈ {BACK,HOME,ENTER}` | 按下系统按键 |
| Wait | `{'action':'wait','duration':500}` | 当前固定为 `500ms` | 等待界面响应 |
| Stop | `{'action':'stop'}` | 无 | 终止当前任务 |

模型每次只允许输出一个字典，不输出思考过程、解释或额外文本。字段名和动作名使用上表写法；解析器可以兼容部分历史格式，但新训练数据不得继续生成历史格式。

## CAGUI 原始动作映射

| CAGUI `result_action_type` | 原始语义 | 训练动作 | domestic 数量 |
| ---: | --- | --- | ---: |
| 0 | `LONG_POINT` | `long_press` | 25 |
| 1 | `NO_ACTION` | `wait(duration=500)` | 1 |
| 3 | `TYPE` | `input_text` | 574 |
| 4 | `DUAL_POINT` 且位移 `≤0.04` | `click` | 3237 |
| 4 | `DUAL_POINT` 且位移 `>0.04` | `scroll` | 79 |
| 5/6/7 | `PRESS_BACK/HOME/ENTER` | `press(key=...)` | 0 |
| 10 | `STATUS_TASK_COMPLETE` | `stop` | 575 |
| 11 | `STATUS_TASK_IMPOSSIBLE` | `stop` | 25 |

完整 `domestic` 划分共有 600 个 episode、4516 个 step。`impossible` 不作为独立训练动作，但保留来源信息并在训练边界归并为 `stop`。

## 转换后分布

| 训练动作 | 样本数 | 占比 |
| --- | ---: | ---: |
| Click | 3237 | 71.68% |
| Long Press | 25 | 0.55% |
| Scroll | 79 | 1.75% |
| Input Text | 574 | 12.71% |
| Press | 0 | 0% |
| Wait | 1 | 0.02% |
| Stop | 600 | 13.29% |
| 合计 | 4516 | 100% |

Press 虽未出现在 CAGUI domestic 中，仍属于统一动作空间，并可由 UI-R1 等外部训练数据提供监督。

## 版本约束

本协议自提交 `d45c080` 后生效。此前训练的 E1 checkpoint 使用旧标签协议；后续实验以 E1 为初始化权重时，必须重新进行 SFT，不能把旧 checkpoint 的输出能力当作已适配新协议。
