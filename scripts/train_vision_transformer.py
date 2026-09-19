#!/usr/bin/env python3
"""Train an inspectable pixels-only temporal Transformer for Cart-Pole.

32x32 grayscale frame -> 2x2 patch averages -> 16x16=256 visual features
-> learned 256->24 frame token -> 8-frame causal Transformer
-> inferred normalized [x, x_dot, theta, theta_dot]
-> fixed transparent state-feedback action.

The Transformer never receives simulator state as input. State is used only as
supervision/evaluation so the UI can compare visual inference with ground truth.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20260919
SEQ_LEN = 8
FRAME_STRIDE = 3
BUFFER_LEN = (SEQ_LEN - 1) * FRAME_STRIDE + 1
FRAME_SIZE = 32
PATCH_SIZE = 2
GRID_SIZE = FRAME_SIZE // PATCH_SIZE
FEATURE_DIM = GRID_SIZE * GRID_SIZE
D_MODEL = 24
D_FF = 48
TAU = 0.02
STATE_SCALE = np.array([2.4, 3.0, 0.38, 3.5], dtype=np.float32)
EXPERT_GAIN = np.array([1.5, 0.5, 8.0, 3.0], dtype=np.float32)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "public" / "model" / "vision-transformer.json"
DOC_PATH = ROOT / "docs" / "vision-model.md"

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
        [x + TAU * xdot, xdot + TAU * x_acc, theta + TAU * thetadot, thetadot + TAU * theta_acc],
        dtype=np.float32,
    )


def terminal(state: np.ndarray) -> bool:
    return abs(float(state[0])) > 2.4 or abs(float(state[2])) > 0.38


def expert_force(state: np.ndarray) -> float:
    z = state / STATE_SCALE
    return 10.0 * math.tanh(float(np.dot(EXPERT_GAIN, z)))


def js_round(value: float) -> int:
    """Match JavaScript Math.round for raster parity."""
    return int(math.floor(value + 0.5))


def set_pixel(frame: np.ndarray, x: int, y: int, value: float) -> None:
    if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE:
        frame[y, x] = max(float(frame[y, x]), float(value))


def render_frame(state: np.ndarray) -> np.ndarray:
    x, _, theta, _ = [float(v) for v in state]
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    frame[27, 1 : FRAME_SIZE - 1] = 0.15

    cx = js_round(16 + np.clip(x / 2.4, -1.0, 1.0) * 12)
    for py in range(23, 27):
        for px in range(cx - 4, cx + 5):
            set_pixel(frame, px, py, 0.72)

    pivot_x, pivot_y = cx, 23
    pole_length = 11
    tip_x = js_round(pivot_x + math.sin(theta) * pole_length)
    tip_y = js_round(pivot_y - math.cos(theta) * pole_length)
    for i in range(37):
        t = i / 36
        px = js_round(pivot_x + (tip_x - pivot_x) * t)
        py = js_round(pivot_y + (tip_y - pivot_y) * t)
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                if abs(ox) + abs(oy) <= 1:
                    set_pixel(frame, px + ox, py + oy, 1.0)
    set_pixel(frame, pivot_x, pivot_y, 0.9)
    return frame


def patch_features(frame: np.ndarray) -> np.ndarray:
    out = np.empty(FEATURE_DIM, dtype=np.float32)
    idx = 0
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            patch = frame[
                gy * PATCH_SIZE : (gy + 1) * PATCH_SIZE,
                gx * PATCH_SIZE : (gx + 1) * PATCH_SIZE,
            ]
            out[idx] = float(patch.mean())
            idx += 1
    return out


def visual_observation(state: np.ndarray) -> np.ndarray:
    return patch_features(render_frame(state))


def initial_state(rng: np.random.Generator, hard: bool = False, zero_velocity: bool = False) -> np.ndarray:
    x_lim = 0.34 if hard else 0.24
    theta_lim = 0.11 if hard else 0.085
    return np.array(
        [
            rng.uniform(-x_lim, x_lim),
            0.0 if zero_velocity else rng.uniform(-0.20, 0.20),
            rng.uniform(-theta_lim, theta_lim),
            0.0 if zero_velocity else rng.uniform(-0.22, 0.22),
        ],
        dtype=np.float32,
    )


def make_dataset(episodes: int = 200, horizon: int = 220):
    rng = np.random.default_rng(SEED)
    xs, action_targets, state_targets = [], [], []

    for episode in range(episodes):
        state = initial_state(rng, hard=(episode % 4 == 0), zero_velocity=(episode % 3 == 0))
        first = visual_observation(state)
        buffer = [first.copy() for _ in range(BUFFER_LEN)]
        pulse_left, disturbance = 0, 0.0

        for step in range(horizon):
            model_history = buffer[::FRAME_STRIDE]
            xs.append(np.stack(model_history))
            action_targets.append(expert_force(state) / 10.0)
            state_targets.append(np.clip(state / STATE_SCALE, -1.0, 1.0))

            if pulse_left <= 0 and step > 8 and rng.random() < 0.020:
                pulse_left = int(rng.integers(3, 9))
                disturbance = float(rng.choice([-6.0, 6.0]))
            if pulse_left > 0:
                pulse_left -= 1
            else:
                disturbance = 0.0

            state = physics_step(state, expert_force(state), disturbance)
            if terminal(state):
                state = initial_state(rng, hard=False, zero_velocity=True)
                first = visual_observation(state)
                buffer = [first.copy() for _ in range(BUFFER_LEN)]
                pulse_left, disturbance = 0, 0.0
                continue
            buffer = buffer[1:] + [visual_observation(state)]

    # float16 stores the deterministic visual features compactly; batches are
    # promoted to float32 before model execution.
    return (
        np.stack(xs).astype(np.float16),
        np.asarray(action_targets, dtype=np.float32),
        np.stack(state_targets).astype(np.float32),
    )


class TinyVisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(FEATURE_DIM, D_MODEL)
        self.pos_embedding = nn.Parameter(torch.zeros(SEQ_LEN, D_MODEL))
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.state = nn.Linear(D_MODEL, 4)
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.register_buffer("controller_gain", torch.tensor(EXPERT_GAIN, dtype=torch.float32))

    def forward(self, features: torch.Tensor):
        tokens = self.embed(features) + self.pos_embedding
        norm1 = self.ln1(tokens)
        q, k, v = self.q(norm1), self.k(norm1), self.v(norm1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D_MODEL)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=features.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        residual1 = tokens + self.o(context)
        norm2 = self.ln2(residual1)
        ff = self.ff2(F.gelu(self.ff1(norm2), approximate="tanh"))
        hidden = residual1 + ff
        state_norm = torch.tanh(self.state(hidden[:, -1, :]))
        action_score = (state_norm * self.controller_gain).sum(dim=-1)
        return action_score, state_norm


def train_model(x: np.ndarray, y_action: np.ndarray, y_state: np.ndarray):
    n = len(x)
    split = int(n * 0.88)
    train_x, val_x = x[:split], x[split:]
    train_a, val_a = y_action[:split], y_action[split:]
    train_s, val_s = y_state[:split], y_state[split:]

    model = TinyVisionTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)

    tx = torch.from_numpy(train_x)
    ta = torch.from_numpy(train_a)
    ts = torch.from_numpy(train_s)
    vx = torch.from_numpy(val_x)
    va = torch.from_numpy(val_a)
    vs = torch.from_numpy(val_s)

    losses = []
    batch = 320
    epochs = 30
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tx), generator=generator)
        total = 0.0
        count = 0
        for start in range(0, len(tx), batch):
            idx = perm[start : start + batch]
            action_score, state_norm = model(tx[idx].float())
            action_pred = torch.tanh(action_score)
            action_loss = F.mse_loss(action_pred, ta[idx])
            state_loss = F.mse_loss(state_norm, ts[idx])
            loss = action_loss + 0.65 * state_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
            count += len(idx)

        model.eval()
        with torch.no_grad():
            val_score, val_state = model(vx.float())
            val_action = torch.tanh(val_score)
            val_action_mse = float(F.mse_loss(val_action, va))
            val_state_mse = float(F.mse_loss(val_state, vs))
        mean = total / count
        losses.append((mean, val_action_mse, val_state_mse))
        print(f"epoch {epoch+1:02d}/{epochs}: train={mean:.7f} val_action={val_action_mse:.7f} val_state={val_state_mse:.7f}")

    return model.eval(), losses, (val_x, val_a, val_s)


@torch.no_grad()
def model_outputs(model: TinyVisionTransformer, history: list[np.ndarray]):
    x = torch.from_numpy(np.stack(history)[None, ...].astype(np.float32))
    action_score, state_norm = model(x)
    force = 10.0 * math.tanh(float(action_score[0]))
    inferred_state = state_norm[0].cpu().numpy() * STATE_SCALE
    return force, inferred_state


@torch.no_grad()
def validation_ablation(model: TinyVisionTransformer, val):
    x, y_action, y_state = val
    full = torch.from_numpy(x).float()
    repeated = torch.from_numpy(np.repeat(x[:, -1:, :], SEQ_LEN, axis=1).copy()).float()

    score_full, state_full = model(full)
    score_rep, state_rep = model(repeated)
    action_full = torch.tanh(score_full).cpu().numpy()
    action_rep = torch.tanh(score_rep).cpu().numpy()
    inferred_full = state_full.cpu().numpy() * STATE_SCALE
    inferred_rep = state_rep.cpu().numpy() * STATE_SCALE
    true_state = y_state * STATE_SCALE

    return {
        "action_mae_full": float(np.mean(np.abs(action_full - y_action))),
        "action_mae_repeat_latest": float(np.mean(np.abs(action_rep - y_action))),
        "x_mae_full": float(np.mean(np.abs(inferred_full[:, 0] - true_state[:, 0]))),
        "x_mae_repeat_latest": float(np.mean(np.abs(inferred_rep[:, 0] - true_state[:, 0]))),
        "xdot_mae_full": float(np.mean(np.abs(inferred_full[:, 1] - true_state[:, 1]))),
        "xdot_mae_repeat_latest": float(np.mean(np.abs(inferred_rep[:, 1] - true_state[:, 1]))),
        "theta_mae_full": float(np.mean(np.abs(inferred_full[:, 2] - true_state[:, 2]))),
        "theta_mae_repeat_latest": float(np.mean(np.abs(inferred_rep[:, 2] - true_state[:, 2]))),
        "thetadot_mae_full": float(np.mean(np.abs(inferred_full[:, 3] - true_state[:, 3]))),
        "thetadot_mae_repeat_latest": float(np.mean(np.abs(inferred_rep[:, 3] - true_state[:, 3]))),
    }


def evaluate_closed_loop(model: TinyVisionTransformer, repeat_latest: bool, episodes: int = 60, horizon: int = 500):
    rng = np.random.default_rng(SEED + 91)
    lengths = []
    for _ in range(episodes):
        state = initial_state(rng, hard=True, zero_velocity=True)
        first = visual_observation(state)
        buffer = [first.copy() for _ in range(BUFFER_LEN)]
        pulse_left, disturbance = 0, 0.0
        steps = 0

        for step in range(horizon):
            spaced_history = buffer[::FRAME_STRIDE]
            model_history = [spaced_history[-1].copy() for _ in range(SEQ_LEN)] if repeat_latest else spaced_history
            force, _ = model_outputs(model, model_history)

            if pulse_left <= 0 and step > 18 and rng.random() < 0.012:
                pulse_left = int(rng.integers(3, 7))
                disturbance = float(rng.choice([-4.0, 4.0]))
            if pulse_left > 0:
                pulse_left -= 1
            else:
                disturbance = 0.0

            state = physics_step(state, force, disturbance)
            buffer = buffer[1:] + [visual_observation(state)]
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


def serial_linear(layer: nn.Linear):
    return {"weight": layer.weight.detach().cpu().tolist(), "bias": layer.bias.detach().cpu().tolist()}


def save_artifact(model, losses, ablation, closed_full, closed_repeat):
    artifact = {
        "format": "cartpole-vision-transformer-v1",
        "seed": SEED,
        "sequence_length": SEQ_LEN,
        "frame_stride_steps": FRAME_STRIDE,
        "frame_stride_seconds": FRAME_STRIDE * TAU,
        "history_span_seconds": (SEQ_LEN - 1) * FRAME_STRIDE * TAU,
        "frame_size": FRAME_SIZE,
        "patch_size": PATCH_SIZE,
        "grid_size": GRID_SIZE,
        "feature_dim": FEATURE_DIM,
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "state_scale": STATE_SCALE.tolist(),
        "controller_gain": EXPERT_GAIN.tolist(),
        "training": {
            "objective": "pixels-only visual state inference + transparent state-feedback action",
            "epochs": len(losses),
            "final_train_loss": losses[-1][0],
            "final_val_action_mse": losses[-1][1],
            "final_val_state_mse": losses[-1][2],
        },
        "ablation": ablation,
        "closed_loop": {"full_history": closed_full, "repeat_latest_frame": closed_repeat},
        "weights": {
            "embed": serial_linear(model.embed),
            "pos_embedding": model.pos_embedding.detach().cpu().tolist(),
            "ln1": {"weight": model.ln1.weight.detach().cpu().tolist(), "bias": model.ln1.bias.detach().cpu().tolist()},
            "q": serial_linear(model.q),
            "k": serial_linear(model.k),
            "v": serial_linear(model.v),
            "o": serial_linear(model.o),
            "ln2": {"weight": model.ln2.weight.detach().cpu().tolist(), "bias": model.ln2.bias.detach().cpu().tolist()},
            "ff1": serial_linear(model.ff1),
            "ff2": serial_linear(model.ff2),
            "state": serial_linear(model.state),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")

    ratio_xdot = ablation["xdot_mae_repeat_latest"] / max(1e-9, ablation["xdot_mae_full"])
    ratio_thetadot = ablation["thetadot_mae_repeat_latest"] / max(1e-9, ablation["thetadot_mae_full"])
    DOC_PATH.write_text(
        f"""# Vision-only tiny Transformer

Generated deterministically by `scripts/train_vision_transformer.py`.

## Input contract

`32×32 grayscale frame → 2×2 patches → 16×16 = 256 patch-average features → learned 256→24 frame token`.

The Transformer receives eight visual observations sampled every {FRAME_STRIDE*TAU:.2f} s (a {(SEQ_LEN-1)*FRAME_STRIDE*TAU:.2f} s history span) and **never receives simulator state directly**. Its final hidden token estimates normalized `[x, x_dot, theta, theta_dot]`; the same transparent fixed state-feedback equation used elsewhere maps that estimate to force.

## Architecture

- sequence length: {SEQ_LEN}
- frame interval: {FRAME_STRIDE*TAU:.2f} s
- visual history span: {(SEQ_LEN-1)*FRAME_STRIDE*TAU:.2f} s
- frame size: {FRAME_SIZE}×{FRAME_SIZE}
- patch grid: {GRID_SIZE}×{GRID_SIZE}
- frame feature width: {FEATURE_DIM}
- model width: {D_MODEL}
- heads: 1
- transformer blocks: 1
- FFN width: {D_FF}

## Validation ablation

| Metric | 8-frame history | Latest frame repeated |
| --- | ---: | ---: |
| action MAE (normalized) | {ablation['action_mae_full']:.4f} | {ablation['action_mae_repeat_latest']:.4f} |
| x MAE (m) | {ablation['x_mae_full']:.4f} | {ablation['x_mae_repeat_latest']:.4f} |
| x_dot MAE (m/s) | {ablation['xdot_mae_full']:.4f} | {ablation['xdot_mae_repeat_latest']:.4f} |
| theta MAE (rad) | {ablation['theta_mae_full']:.4f} | {ablation['theta_mae_repeat_latest']:.4f} |
| theta_dot MAE (rad/s) | {ablation['thetadot_mae_full']:.4f} | {ablation['thetadot_mae_repeat_latest']:.4f} |

Removing temporal information increases x_dot error by {ratio_xdot:.2f}× and theta_dot error by {ratio_thetadot:.2f}×.

## Closed-loop evaluation

Randomized visual-only starts at zero velocity, followed by occasional ±4 N disturbances.

| Mode | Mean steps | Median | Minimum | 500-step success |
| --- | ---: | ---: | ---: | ---: |
| Full 8-frame visual history | {closed_full['mean_steps']:.1f} | {closed_full['median_steps']:.1f} | {closed_full['min_steps']} | {closed_full['success_rate_500']*100:.1f}% |
| Latest frame repeated 8× | {closed_repeat['mean_steps']:.1f} | {closed_repeat['median_steps']:.1f} | {closed_repeat['min_steps']} | {closed_repeat['success_rate_500']*100:.1f}% |

## Claim boundary

This is an intentionally small educational visual front-end, not a ViT benchmark. Patch averages are used because the frame→patch→token arithmetic stays fully inspectable.
""",
        encoding="utf-8",
    )


def main():
    x, action_y, state_y = make_dataset()
    print("dataset", x.shape, action_y.shape, state_y.shape)
    model, losses, val = train_model(x, action_y, state_y)
    ablation = validation_ablation(model, val)
    print("ablation", ablation)

    full = evaluate_closed_loop(model, repeat_latest=False)
    repeat = evaluate_closed_loop(model, repeat_latest=True)
    print("closed_full", full)
    print("closed_repeat", repeat)

    if full["mean_steps"] < 320:
        raise SystemExit("vision policy acceptance failed: mean_steps < 320")
    if ablation["xdot_mae_full"] >= ablation["xdot_mae_repeat_latest"] * 0.90:
        raise SystemExit("temporal ablation failed: xdot history advantage too small")
    if ablation["thetadot_mae_full"] >= ablation["thetadot_mae_repeat_latest"] * 0.90:
        raise SystemExit("temporal ablation failed: thetadot history advantage too small")

    save_artifact(model, losses, ablation, full, repeat)
    print("wrote", MODEL_PATH)
    print("wrote", DOC_PATH)


if __name__ == "__main__":
    main()
