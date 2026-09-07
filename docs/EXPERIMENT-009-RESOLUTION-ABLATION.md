# 实验 009：CAGUI 输入分辨率消融

## 实验设置

- E1：`artifacts/train/sft-xy-baseline-ddp-bf16-20260903-102135/checkpoint-300`
- 672 smoke：`artifacts/train/e1-param-672-smoke-20260906`
- 固定 Validation：60 episodes / 457 steps
- History window：4
- 推理：4-bit、BF16、batch 1、greedy decoding、显式 EOS
- 评测：AgentCPM-compatible TM/EM、Strict Click、Strict Type、Goal Progress、Episode Success

672 smoke 从 E1 出发训练20 step，有效 batch 8、学习率1e-4，耗时28分18秒；train loss 0.5083，64条 eval loss 0.5081，无 OOM/NaN。

## 完整结果

| 指标 | E1 @448 | E1 @672 | E1 + 672训练20步 @672 | checkpoint-1050 @448 | checkpoint-1050 @672 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TM | 76.37% | 78.99% | 84.25% | **89.06%** | 87.31% |
| EM | 35.45% | 39.82% | 45.08% | 59.30% | **59.96%** |
| Click EM | 37.50% | 42.07% | 42.99% | 53.96% | **57.32%** |
| Strict Click | 25.30% | 31.10% | 30.49% | 42.07% | **45.43%** |
| Input Text EM | 48.15% | 50.00% | 74.07% | **85.19%** | 75.93% |
| Strict Type | 29.63% | 27.78% | 44.44% | **62.96%** | 57.41% |
| Stop EM | 21.67% | 28.33% | 41.67% | **80.00%** | 75.00% |
| Scroll / Long Press EM | 0% | 0% | 0% | 0% | 0% |
| Goal Progress | 10.33% | 17.01% | 20.70% | 22.93% | **32.74%** |
| Episode Success | 0/60 | 1/60 | 0/60 | 4/60 | **5/60** |

## 结论

只把同一 E1 的推理分辨率从448提高到672，TM +2.62 pp、EM +4.37 pp、Click EM +4.57 pp、Strict Click +5.79 pp、Goal Progress +6.68 pp，说明高分辨率推理本身有明确定位收益。

在相同672推理下继续训练20步后，TM和EM均 +5.25 pp，Input Text EM +24.07 pp，Stop EM +13.34 pp；但Click EM仅 +0.91 pp，Strict Click -0.61 pp。短训收益主要来自动作选择、文本和停止判断，没有证明严格点击定位继续改善。

`checkpoint-1050` 的零训练分辨率对照进一步验证了定位收益：448→672 后 Click EM 与 Strict Click 均 +3.36 pp，Goal Progress +9.81 pp，Episode Success 从 4/60 提升到 5/60，整体 EM +0.66 pp。代价是 TM -1.75 pp、Input Text EM -9.26 pp、Strict Type -5.55 pp、Stop EM -5.00 pp。因此 672 不是全面提升，但对当前最关键的点击定位和轨迹前缀完成度有明显价值。

下一步不应直接进行长周期高分辨率重训。先从 `checkpoint-1050` 做短周期 672 继续训练，并以 Strict Click、Strict Type、Goal Progress 和 Episode Success 联合选型；如果文本与停止指标继续下降，则需要加入参数类别均衡采样或混合 448/672 训练。

## 产物

```text
artifacts/train/e1-param-672-smoke-20260906/
artifacts/logs/e1-param-672-smoke-20260906.log
artifacts/eval/cagui-e1-vs-672-smoke-validation457-20260907/
artifacts/eval/e1-checkpoint1050-resolution672-validation457-20260907/
```
