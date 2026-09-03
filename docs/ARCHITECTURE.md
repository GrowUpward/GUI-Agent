# Architecture

## Training path

The active model predicts exactly one structured action from the current screenshot, user instruction, recent action history, and candidate UI boxes.

```text
raw CAGUI episode JSON + screenshots
                |
                v
       CaguiEpisodeDataset
       - episode ordering
       - explicit CAGUI [y,x] -> action [x,y] conversion
       - bounded history window
       - structured action target
                |
                v
       Qwen3.5-0.8B processor
                |
        +-------+-------+
        |               |
      SFT             GRPO
  assistant-only   grouped rollout
      loss          structured reward
        |               |
        +-------+-------+
                v
        PEFT LoRA adapter
```

SFT and GRPO share the same prompt builder, target conversion, parser, and reward implementation. This prevents format drift between the bridge stage and policy optimization.

The coordinate conversion is a tested data-boundary invariant. Internally and at the API/ADB boundary, every action coordinate is `[x, y]` in the normalized 0–1000 system. Only archived pre-audit adapters require the deployment-only `LEGACY_YX_OUTPUT=1` compatibility switch.

## Reward

The current reward decomposes into:

```text
R = 0.2 * format + 0.3 * action + 0.5 * arguments
```

Arguments are action-dependent:

- click/long press: normalized point distance and target-box containment;
- input text: exact match plus sequence similarity shaping;
- swipe: start point and direction;
- press/status: strict value match;
- wait: valid duration range.

This is an offline label-derived reward. A future closed-loop version should verify task completion from environment state.

## Deployment path

```text
Android screenshot
   -> FastAPI model server
   -> Python-style action dict
   -> normalized-to-pixel conversion
   -> human confirmation
   -> ADB execution
```

Human confirmation is part of the default design because the current model and mobile execution path are not yet reliable enough for unattended operation.
