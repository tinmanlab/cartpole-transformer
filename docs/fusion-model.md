# State + vision typed-token fusion

Generated deterministically by `scripts/train_fusion_transformer.py`.

## Token contract

Each of eight aligned timestamps contributes two tokens:

1. state token: normalized `[x, x_dot, theta, theta_dot]` + availability bit
2. vision token: 256 patch means + 256 delta-patch values + availability bit

The model therefore runs one-head self-attention over 16 typed tokens. Its causal mask is based on **timestamp**, not token ordering, so state and vision tokens from the same timestamp can attend to each other while all future timestamps remain blocked.

No cross-attention module is used.

## Validation ablation

```json
{
  "clean": {
    "state_mae": 0.010551732033491135,
    "xdot_mae": 0.027253514155745506,
    "thetadot_mae": 0.04212873429059982
  },
  "state_only": {
    "state_mae": 0.010903911665081978,
    "xdot_mae": 0.02518252097070217,
    "thetadot_mae": 0.0456518679857254
  },
  "vision_only": {
    "state_mae": 0.03385325148701668,
    "xdot_mae": 0.14846962690353394,
    "thetadot_mae": 0.15393494069576263
  },
  "noisy_state": {
    "state_mae": 0.0801282525062561,
    "xdot_mae": 0.12381520122289658,
    "thetadot_mae": 0.30975574254989624
  },
  "noisy_state_plus_vision": {
    "state_mae": 0.05820506438612938,
    "xdot_mae": 0.13666002452373505,
    "thetadot_mae": 0.28500378131866455
  },
  "partial_vision": {
    "state_mae": 0.041514575481414795,
    "xdot_mae": 0.167983740568161,
    "thetadot_mae": 0.17615190148353577
  },
  "partial_vision_plus_state": {
    "state_mae": 0.008473418653011322,
    "xdot_mae": 0.02553199790418148,
    "thetadot_mae": 0.03119317814707756
  }
}
```

## Closed-loop evaluation

```json
{
  "clean": {
    "episodes": 40,
    "horizon_steps": 500,
    "mean_steps": 500.0,
    "median_steps": 500.0,
    "min_steps": 500,
    "success_rate_500": 1.0
  },
  "state_only": {
    "episodes": 40,
    "horizon_steps": 500,
    "mean_steps": 500.0,
    "median_steps": 500.0,
    "min_steps": 500,
    "success_rate_500": 1.0
  },
  "vision_only": {
    "episodes": 40,
    "horizon_steps": 500,
    "mean_steps": 328.70001220703125,
    "median_steps": 326.0,
    "min_steps": 90,
    "success_rate_500": 0.15
  },
  "noisy_state": {
    "episodes": 40,
    "horizon_steps": 500,
    "mean_steps": 146.1750030517578,
    "median_steps": 147.0,
    "min_steps": 124,
    "success_rate_500": 0.0
  },
  "noisy_state_plus_vision": {
    "episodes": 40,
    "horizon_steps": 500,
    "mean_steps": 116.94999694824219,
    "median_steps": 116.0,
    "min_steps": 82,
    "success_rate_500": 0.0
  },
  "partial_vision": {
    "episodes": 40,
    "horizon_steps": 500,
    "mean_steps": 177.14999389648438,
    "median_steps": 139.5,
    "min_steps": 31,
    "success_rate_500": 0.025
  },
  "partial_vision_plus_state": {
    "episodes": 40,
    "horizon_steps": 500,
    "mean_steps": 500.0,
    "median_steps": 500.0,
    "min_steps": 500,
    "success_rate_500": 1.0
  }
}
```

## Claim boundary

Fusion is not expected to outperform clean explicit state on Cart-Pole. The useful question is how the modalities interact under degradation. In this simple typed-token model, state strongly repairs partial vision. Vision lowers average state-estimation error under biased state noise, but does **not** improve noisy-state closed-loop control; that negative result is retained rather than hidden or patched with a more complex fusion mechanism.
