# 实验 006：ScreenSpot / ScreenSpotV2 评测

## 目的

使用当前最优 E1 权重，在完整 ScreenSpot 与 ScreenSpotV2 测试集上测量模型的单步视觉定位和输出协议稳定性。

## 权重与数据

- 基座：Qwen3.5-0.8B
- Adapter：`sft-xy-baseline-ddp-bf16-20260903-102135/checkpoint-300`（E1）
- ScreenSpot：`ScreenSpot/screenspot_test.json`，1272 条
- ScreenSpotV2：`ScreenSpotV2/screenspotv2_test.json`，1272 条
- 图像最长边：448
- 解码：greedy，最多 48 个新 token，显式 EOS
- 并行：每套数据拆成 8 个互斥分片，每张 GPU 运行 4 个进程；最终按原始样本顺序合并

## 指标口径

- 原始严格格式率：未经后处理的完整输出恰好是一个可解析字典。
- 首动作可解析率：允许截取第一个平衡字典后，动作可被解析。
- EOS 终止率：生成 token 中出现 EOS。
- 额外内容率：第一个字典之外仍存在内容。
- 动作类型准确率：预测动作是否为 `click`，解析失败计错。
- 坐标合法率：输出二维数值坐标且两个值都在 `[0, 1000]`。
- 参数正确率 / Grounding Accuracy：归一化坐标还原到原图后，是否落入 GT bbox；分母为全部样本。

ScreenSpot 系列全部是单步点击定位样本。因此，动作类型准确率只能验证模型是否稳定输出 `click`，不能衡量多动作分类；数据中也没有输入文本、滑动方向等参数标注。核心可比指标是 bbox 命中率。

## 结果

待全量评测完成后写入。
