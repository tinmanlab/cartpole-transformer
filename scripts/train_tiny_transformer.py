#!/usr/bin/env python3
"""Train a tiny causal Transformer policy for the Cart-Pole explainer.

The goal is educational transparency, not Cart-Pole SOTA.  The model is
intentionally tiny so every intermediate tensor can be recomputed in-browser.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20260919
SEQ_LEN = 8
D_MODEL = 8
D_FF = 16
TAU = 0.02
SCALE = np.array([2.4, 3.0, 0.38, 3.5], dtype=np.float32)
EXPERT_GAIN = np.array([1.5, 0.5, 8.0, 3.0], dtype=np.float32)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "public" / "model" / "tiny-transformer.json"
DOC_PATH = ROOT / "docs" / "learned-model.md"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


def physics_step(state: np.ndarray, control: float, disturbance: float = 0.0) -> np.ndarray:
    gravity, mass_cart, mass_pole, half_len = 9.8, 1.0, 0.1, 0.5
    total_mass = mass_cart + mass_pole
    pole_mass_length = mass_pole * half_len
    force = float(np.clip(control, -10.0, 10.0) + disturbance)
    x, xdot, theta, thetadot = [float(v) for v in state]
    costheta, sintheta = math.cos(theta), math.sin(theta)
    temp = (force + pole_mass_length * thetadot * thetadot * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        half_len * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass)
    )
    x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass
    return np.array(
        [
            x + TAU * xdot,
            xdot + TAU * x_acc,
            theta + TAU * thetadot,
            thetadot + TAU * theta_acc,
        ],
        dtype=np.float32,
    )


def terminal(state: np.ndarray) -> bool:
    return abs(float(state[0])) > 2.4 or abs(float(state[2])) > 0.38


def expert_force(state: np.ndarray) -> float:
    z = state / SCALE
    score = float(np.dot(EXPERT_GAIN, z))
    return 10.0 * math.tanh(score)


def initial_state(rng: np.random.Generator, hard: bool = False) -> np.ndarray:
    if hard:
        limits = np.array([0.35, 0.35, 0.11, 0.35], dtype=np.float32)
    else:
        limits = np.array([0.25, 0.22, 0.085, 0.25], dtype=np.float32)
    return rng.uniform(-limits, limits).astype(np.float32)


def make_dataset(episodes: int = 320, horizon: int = 220) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    samples, targets = [], []
    for episode in range(episodes):
        state = initial_state(rng, hard=(episode % 4 == 0))
        history = [state.copy() for _ in range(SEQ_LEN)]
        pulse_left = 0
        disturbance = 0.0
        for step in range(horizon):
            target = expert_force(state)
            samples.append(np.stack(history))
            targets.append(target / 10.0)

            if pulse_left <= 0 and rng.random() < 0.018:
                pulse_left = int(rng.integers(3, 9))
                disturbance = float(rng.choice([-6.0, 6.0]))
            if pulse_left > 0:
                pulse_left -= 1
            else:
                disturbance = 0.0

            state = physics_step(state, target, disturbance)
            if terminal(state):
                state = initial_state(rng, hard=False)
                history = [state.copy() for _ in range(SEQ_LEN)]
                pulse_left = 0
                disturbance = 0.0
                continue
            history = history[1:] + [state.copy()]
    return np.stack(samples).astype(np.float32), np.asarray(targets, dtype=np.float32)


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, D_MODEL)
        self.pos_embedding = nn.Parameter(torch.zeros(SEQ_LEN, D_MODEL))
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.action = nn.Linear(D_MODEL, 1)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, raw: torch.Tensor, return_intermediates: bool = False):
        scale = torch.as_tensor(SCALE, dtype=raw.dtype, device=raw.device)
        normalized = raw / scale
        embedded = self.embed(normalized) + self.pos_embedding
        norm1 = self.ln1(embedded)
        q, k, v = self.q(norm1), self.k(norm1), self.v(norm1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D_MODEL)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=raw.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        residual1 = embedded + self.o(context)
        norm2 = self.ln2(residual1)
        ff = self.ff2(F.gelu(self.ff1(norm2), approximate="tanh"))
        hidden = residual1 + ff
        logit = self.action(hidden[:, -1, :]).squeeze(-1)
        if return_intermediates:
            return logit, {
                "normalized": normalized,
                "embedded": embedded,
                "q": q,
                "k": k,
                "v": v,
                "scores": scores,
                "weights": weights,
                "context": context,
                "hidden": hidden,
            }
        return logit


def train_model(x: np.ndarray, y: np.ndarray) -> tuple[TinyTransformer, list[float]]:
    model = TinyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    generator = torch.Generator().manual_seed(SEED)
    losses: list[float] = []

    batch = 512
    epochs = 28
    for epoch in range(epochs):
        perm = torch.randperm(len(x_t), generator=generator)
        total = 0.0
        count = 0
        model.train()
        for start in range(0, len(x_t), batch):
            idx = perm[start : start + batch]
            pred = torch.tanh(model(x_t[idx]))
            loss = F.mse_loss(pred, y_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(idx)
            count += len(idx)
        mean = total / count
        losses.append(mean)
        print(f"epoch {epoch+1:02d}/{epochs}: mse={mean:.7f}")
    return model.eval(), losses


@torch.no_grad()
def learned_force(model: TinyTransformer, history: list[np.ndarray]) -> float:
    raw = torch.from_numpy(np.stack(history)[None, ...])
    score = float(model(raw)[0])
    return 10.0 * math.tanh(score)


def evaluate_controller(controller, episodes: int = 80, horizon: int = 500) -> dict:
    rng = np.random.default_rng(SEED + 77)
    lengths = []
    for episode in range(episodes):
        state = initial_state(rng, hard=True)
        history = [state.copy() for _ in range(SEQ_LEN)]
        pulse_left, disturbance = 0, 0.0
        steps = 0
        for step in range(horizon):
            if pulse_left <= 0 and step > 20 and rng.random() < 0.010:
                pulse_left = int(rng.integers(3, 7))
                disturbance = float(rng.choice([-4.0, 4.0]))
            if pulse_left > 0:
                pulse_left -= 1
            else:
                disturbance = 0.0
            force = controller(history)
            state = physics_step(state, force, disturbance)
            history = history[1:] + [state.copy()]
            steps = step + 1
            if terminal(state):
                break
        lengths.append(steps)
    a = np.asarray(lengths)
    return {
        "episodes": episodes,
        "horizon_steps": horizon,
        "mean_steps": float(a.mean()),
        "median_steps": float(np.median(a)),
        "min_steps": int(a.min()),
        "success_rate_500": float(np.mean(a >= horizon)),
    }


def serial_linear(layer: nn.Linear) -> dict:
    return {
        "weight": layer.weight.detach().cpu().tolist(),
        "bias": layer.bias.detach().cpu().tolist(),
    }


def save_artifact(model: TinyTransformer, losses: list[float], metrics: dict) -> None:
    artifact = {
        "format": "cartpole-tiny-transformer-v1",
        "seed": SEED,
        "sequence_length": SEQ_LEN,
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "normalization_scale": SCALE.tolist(),
        "training": {
            "objective": "behavior cloning from deterministic linear state-feedback teacher",
            "final_mse": losses[-1],
            "epochs": len(losses),
        },
        "evaluation": metrics,
        "weights": {
            "embed": serial_linear(model.embed),
            "pos_embedding": model.pos_embedding.detach().cpu().tolist(),
            "ln1": {
                "weight": model.ln1.weight.detach().cpu().tolist(),
                "bias": model.ln1.bias.detach().cpu().tolist(),
            },
            "q": serial_linear(model.q),
            "k": serial_linear(model.k),
            "v": serial_linear(model.v),
            "o": serial_linear(model.o),
            "ln2": {
                "weight": model.ln2.weight.detach().cpu().tolist(),
                "bias": model.ln2.bias.detach().cpu().tolist(),
            },
            "ff1": serial_linear(model.ff1),
            "ff2": serial_linear(model.ff2),
            "action": serial_linear(model.action),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")

    lm = metrics["learned"]
    linear = metrics["linear_baseline"]
    DOC_PATH.write_text(
        f"""# Learned tiny Transformer

Generated deterministically by `scripts/train_tiny_transformer.py`.

- seed: `{SEED}`
- sequence length: `{SEQ_LEN}`
- model width: `{D_MODEL}`
- heads: `1`
- transformer blocks: `1`
- feed-forward width: `{D_FF}`
- final behavior-cloning MSE: `{losses[-1]:.7f}`

## Closed-loop evaluation

The evaluation uses the same Cart-Pole equations as the browser, randomized initial
states, and occasional ±4 N disturbance pulses. Horizon is 500 control steps (10 s).

| Controller | Mean steps | Median | Minimum | 500-step success |
| --- | ---: | ---: | ---: | ---: |
| Tiny learned Transformer | {lm['mean_steps']:.1f} | {lm['median_steps']:.1f} | {lm['min_steps']} | {lm['success_rate_500']*100:.1f}% |
| Linear state-feedback teacher | {linear['mean_steps']:.1f} | {linear['median_steps']:.1f} | {linear['min_steps']} | {linear['success_rate_500']*100:.1f}% |

The linear controller is intentionally reported because Cart-Pole does **not** require
a Transformer. The Transformer is used here to make sequence attention observable,
not to claim architectural superiority.

## Claim boundary

Attention weights are internal learned weighting values. They are not treated as a
causal explanation of the policy's behavior.
""",
        encoding="utf-8",
    )


def main() -> None:
    x, y = make_dataset()
    print("dataset", x.shape, y.shape)
    model, losses = train_model(x, y)

    learned = evaluate_controller(lambda h: learned_force(model, h))
    linear = evaluate_controller(lambda h: expert_force(h[-1]))
    metrics = {"learned": learned, "linear_baseline": linear}
    print("learned", learned)
    print("linear", linear)

    if learned["mean_steps"] < 300:
        raise SystemExit("learned policy failed minimum closed-loop acceptance: mean_steps < 300")
    save_artifact(model, losses, metrics)
    print("wrote", MODEL_PATH)
    print("wrote", DOC_PATH)


if __name__ == "__main__":
    main()
