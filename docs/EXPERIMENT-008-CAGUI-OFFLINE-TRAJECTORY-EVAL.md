# 实验 008：CAGUI 离线多步轨迹评测

## 目的

在固定 Validation-60 上以 AgentCPM-GUI 兼容口径评测 E1 与 E2，同时保留更严格的参数诊断指标。该实验用于真机闭环前的离线多步模型选择，不读取固定 Test。

## 指标

- TM：动作类型与 GT 一致。
- EM：类型正确且参数满足 AgentCPM-GUI 判定。
- Episode Success：一条 episode 的全部 step 均为 EM。
- Goal Progress：首次 EM 错误前的连续正确 step 数除以 episode step 数，再按 episode 平均。
- Strict Click：预测点落入未经扩大的目标 UI bbox；无框时沿用项目严格距离规则。
- Strict Type：预测文本与标注文本完全一致。

官方兼容 Click EM 使用 1.2 倍目标 UI 框；若 GT 点击不属于任何 UI 框，则使用归一化欧氏距离 `<=0.14`。官方兼容 Type EM 忽略大小写和首尾空格，并接受预测与 GT 互为子串。

## 对照

- E1：Base → CAGUI SFT。
- E2：Base → UI-R1 Stage0 → CAGUI SFT。
- 数据：固定 Validation 60 episodes / 457 steps。
- 推理：history window 4、最多 30 个 UI boxes、最长边 448、greedy decoding、显式 EOS。

## 结果

待评测完成后写入。
