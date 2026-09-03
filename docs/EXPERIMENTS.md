# Experiment status

## Dataset snapshot

The recorded CAGUI domestic split contains 600 episodes and 4,516 steps:

| Action | Count |
| --- | ---: |
| click | 3,237 |
| input_text | 574 |
| stop | 575 |
| scroll | 79 |
| wait | 25 |
| impossible | 25 |
| unknown | 1 |

Click accounts for roughly 71.7% of the data. Always report per-action metrics alongside overall accuracy.

## Archived SFT result

Source artifact in the original workspace:

```text
/data/data1/zhaozhiqing/project/UI-R1/results/train/
continue_sft_qwen35_0.8b_cagui_agent_base_lora/lr_1e_4_plus300
```

Fixed 128-sample generation test:

| Metric | Result |
| --- | ---: |
| Eval loss | 0.5730 |
| Parse success | 0.8828 |
| Action-type accuracy | 0.7344 |
| Argument accuracy | 0.0938 |
| Click hit rate | 0.1613 |
| Mean click distance | 275.1046 |
| Input-text exact match | 0.4375 |
| Mean reward | 0.3317 |

Compared with its `lr_1e_4` parent experiment, continued training improved action accuracy by 3.13 percentage points, click hit rate by 5.38 points, and reduced mean click distance by approximately 39%. Parse success fell by 4.69 points, so the change is a trade-off rather than an unqualified improvement.

### Coordinate-order audit

During repository cleanup, the active loader was found to pass CAGUI's normalized `[y, x]` fields directly into an action schema documented as `[x, y]`. The new repository fixes that conversion and includes a regression test. Consequently:

- parse success, action-type accuracy, and input-text exact match remain useful historical evidence;
- argument reward, target-box hit rate, and end-to-end click behavior require a corrected baseline rerun;
- the archived adapter can still be demoed with `LEGACY_YX_OUTPUT=1`, which swaps its predicted coordinate before ADB execution;
- new SFT and GRPO runs must use the corrected default and must not enable the legacy flag.

This distinction should be disclosed in résumés and interviews rather than presenting the archived coordinate metrics as a final result.

## GRPO status

A 10-step checkpoint was produced in the original workspace on 2026-07-01, but that run did not persist a self-contained metrics file or train log. It is evidence that the loop executed, not evidence that GRPO improved the policy.

After producing a corrected SFT baseline, the next controlled experiment must use the same test samples, seed, prompt, decoding settings, and metric implementation for:

1. base model;
2. best SFT adapter;
3. SFT + 10-step GRPO;
4. SFT + 50-step GRPO.

Do not increase the GRPO budget unless click hit rate, argument accuracy, or verified task success improves without unacceptable format/action regression.
