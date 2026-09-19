# Vision-only tiny Transformer

Generated deterministically by `scripts/train_vision_transformer.py`.

## Input contract

`32×32 grayscale frame → 2×2 patches → 16×16 = 256 patch means → concatenate Δpatch to previous sampled frame → 512D inspectable visual feature → learned 512→24 frame token`.

The Transformer receives eight visual observations sampled every 0.06 s (a 0.42 s history span) and **never receives simulator state directly**. Its final hidden token estimates normalized `[x, x_dot, theta, theta_dot]`; the same transparent fixed state-feedback equation used elsewhere maps that estimate to force.

## Architecture

- sequence length: 8
- frame interval: 0.06 s
- visual history span: 0.42 s
- frame size: 32×32
- patch grid: 16×16
- patch feature width: 256
- token input width (patch + delta): 512
- model width: 24
- heads: 1
- transformer blocks: 1
- FFN width: 48

## Validation ablation

| Metric | 8-frame history | Latest frame repeated |
| --- | ---: | ---: |
| action MAE (normalized) | 0.1593 | 0.2155 |
| x MAE (m) | 0.0516 | 0.0541 |
| x_dot MAE (m/s) | 0.1449 | 0.1942 |
| theta MAE (rad) | 0.0084 | 0.0108 |
| theta_dot MAE (rad/s) | 0.1475 | 0.2197 |

Removing temporal information increases x_dot error by 1.34× and theta_dot error by 1.49×.

## Closed-loop evaluation

Randomized visual-only starts at zero velocity, followed by occasional ±4 N disturbances.

| Mode | Mean steps | Median | Minimum | 500-step success |
| --- | ---: | ---: | ---: | ---: |
| Full 8-frame visual history | 321.3 | 324.5 | 132 | 10.0% |
| Latest frame repeated 8× | 121.3 | 120.5 | 60 | 0.0% |

## Claim boundary

This is an intentionally small educational visual front-end, not a ViT benchmark. Patch averages are used because the frame→patch→token arithmetic stays fully inspectable.
