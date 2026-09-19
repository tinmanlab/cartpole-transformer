# Learned tiny Transformer

Generated deterministically by `scripts/train_tiny_transformer.py`.

- seed: `20260919`
- sequence length: `8`
- model width: `8`
- heads: `1`
- transformer blocks: `1`
- feed-forward width: `16`
- final behavior-cloning MSE: `0.0000394`

## Closed-loop evaluation

The evaluation uses the same Cart-Pole equations as the browser, randomized initial
states, and occasional ±4 N disturbance pulses. Horizon is 500 control steps (10 s).

| Controller | Mean steps | Median | Minimum | 500-step success |
| --- | ---: | ---: | ---: | ---: |
| Tiny learned Transformer | 500.0 | 500.0 | 500 | 100.0% |
| Linear state-feedback teacher | 500.0 | 500.0 | 500 | 100.0% |

The linear controller is intentionally reported because Cart-Pole does **not** require
a Transformer. The Transformer is used here to make sequence attention observable,
not to claim architectural superiority.

## Claim boundary

Attention weights are internal learned weighting values. They are not treated as a
causal explanation of the policy's behavior.
