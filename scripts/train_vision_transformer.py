#!/usr/bin/env python3
"""Train a tiny pixels-only temporal Transformer for Cart-Pole.

The visual front-end is intentionally inspectable:
32x32 discrete grayscale frame -> 4x4 patch averages -> 8x8=64 features
-> learned 64->16 frame token -> 8-frame causal Transformer.

The policy never receives x, x_dot, theta, or theta_dot directly.
Ground-truth velocities are used only as auxiliary training/evaluation targets.
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
FRAME_SIZE = 32
PATCH_SIZE = 4
GRID_SIZE = FRAME_SIZE // PATCH_SIZE
FEATURE_DIM = GRID_SIZE * GRID_SIZE
D_MODEL = 16
D_FF = 32
TAU = 0.02
STATE_SCALE = np.array([2.4, 3.0, 0.38, 3.5], dtype=np.float32)
MOTION_SCALE = np.array([3.0, 3.5], dtype=np.float32)
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
    z = state / STATE_SCALE
    score = float(np.dot(EXPERT_GAIN, z))
    return 10.0 * math.tanh(score)


def set_pixel(frame: np.ndarray, x: int, y: int, value: float) -> None:
    if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE:
        frame[y, x] = max(float(frame[y, x]), float(value))


def render_frame(state: np.ndarray) -> np.ndarray:
    x, _, theta, _ = [float(v) for v in state]
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)

    track_y = 27
    frame[track_y, 1 : FRAME_SIZE - 1] = 0.15

    cx = int(round(16 + np.clip(x / 2.4, -1.0, 1.0) * 12))
    for py in range(23, 27):
        for px in range(cx - 4, cx + 5):
            set_pixel(frame, px, py, 0.72)

    pivot_x, pivot_y = cx, 23
    pole_length = 11
    tip_x = int(round(pivot_x + math.sin(theta) * pole_length))
    tip_y = int(round(pivot_y - math.cos(theta) * pole_length))
    samples = 36
    for i in range(samples + 1):
        t = i / samples
        px = int(round(pivot_x + (tip_x - pivot_x) * t))
        py = int(round(pivot_y + (tip_y - pivot_y) * t))
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
    state = np.array(
        [
            rng.uniform(-x_lim, x_lim),
            0.0 if zero_velocity else rng.uniform(-0.20, 0.20),
            rng.uniform(-theta_lim, theta_lim),
            0.0 if zero_velocity else rng.uniform(-0.22, 0.22),
        ],
        dtype=np.float32,
    )
    return state


def make_dataset(episodes: int = 230, horizon: int = 220):
    rng = np.random.default_rng(SEED)
    xs, action_targets, motion_targets = [], [], []

    for episode in range(episodes):
        state = initial_state(rng, hard=(episode % 4 == 0), zero_velocity=(episode % 3 == 0))
        first = visual_observation(state)
        history = [first.copy() for _ in range(SEQ_LEN)]
        pulse_left, disturbance = 0, 0.0

        for step in range(horizon):
            xs.append(np.stack(history))
            action_targets.append(expert_force(state) / 10.0)
            motion_targets.append(
                np.clip(
                    np.array([state[1], state[3]], dtype=np.float32) / MOTION_SCALE,
                    -1.0,
                    1.0,
                )
            )

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
                history = [first.copy() for _ in range(SEQ_LEN)]
                pulse_left, disturbance = 0, 0.0
                continue
            history = history[1:] + [visual_observation(state)]

    return (
        np.stack(xs).astype(np.float32),
        np.asarray(action_targets, dtype=np.float32),
        np.stack(motion_targets).astype(np.float32),
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
        self.action = nn.Linear(D_MODEL, 1)
        self.motion = nn.Linear(D_MODEL, 2)
        nn.init.normal_(self.pos_embedding, std=0.02)

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
        final = hidden[:, -1, :]
        return self.action(final).squeeze(-1), self.motion(final)


def train_model(x: np.ndarray, y_action: np.ndarray, y_motion: np.ndarray):
    n = len(x)
    split = int(n * 0.88)
    train_x, val_x = x[:split], x[split:]
    train_a, val_a = y_action[:split], y_action[split:]
    train_m, val_m = y_motion[:split], y_motion[split:]

    model = TinyVisionTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=2.8e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)

    tx = torch.from_numpy(train_x)
    ta = torch.from_numpy(train_a)
    tm = torch.from_numpy(train_m)
    vx = torch.from_numpy(val_x)
    va = torch.from_numpy(val_a)
    vm = torch.from_numpy(val_m)

    losses = []
    batch = 384
    epochs = 28
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tx), generator=generator)
        total = 0.0
        count = 0
        for start in range(0, len(tx), batch):
            idx = perm[start : start + batch]
            action_logit, motion_logit = model(tx[idx])
            action_pred = torch.tanh(action_logit)
            motion_pred = torch.tanh(motion_logit)
            action_loss = F.mse_loss(action_pred, ta[idx])
            motion_loss = F.mse_loss(motion_pred, tm[idx])
            loss = action_loss + 0.40 * motion_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
            count += len(idx)

        model.eval()
        with torch.no_grad():
            val_action_logit, val_motion_logit = model(vx)
            val_action = torch.tanh(val_action_logit)
            val_motion = torch.tanh(val_motion_logit)
            val_action_mse = float(F.mse_loss(val_action, va))
            val_motion_mse = float(F.mse_loss(val_motion, vm))
        mean = total / count
        losses.append((mean, val_action_mse, val_motion_mse))
        print(
            f"epoch {epoch+1:02d}/{epochs}: train={mean:.7f} "
            f"val_action={val_action_mse:.7f} val_motion={val_motion_mse:.7f}"
        )

    return model.eval(), losses, (val_x, val_a, val_m)


@torch.no_grad()
def model_outputs(model: TinyVisionTransformer, history: list[np.ndarray]):
    x = torch.from_numpy(np.stack(history)[None, ...].astype(np.float32))
    action_logit, motion_logit = model(x)
    force = 10.0 * math.tanh(float(action_logit[0]))
    motion = np.tanh(motion_logit[0].cpu().numpy()) * MOTION_SCALE
    return force, motion


@torch.no_grad()
def validation_ablation(model: TinyVisionTransformer, val):
    x, y_action, y_motion = val
    full = torch.from_numpy(x)
    repeated_np = np.repeat(x[:, -1:, :], SEQ_LEN, axis=1).copy()
    repeated = torch.from_numpy(repeated_np)

    a_full, m_full = model(full)
    a_rep, m_rep = model(repeated)
    action_full = torch.tanh(a_full).cpu().numpy()
    action_rep = torch.tanh(a_rep).cpu().numpy()
    motion_full = np.tanh(m_full.cpu().numpy()) * MOTION_SCALE
    motion_rep = np.tanh(m_rep.cpu().numpy()) * MOTION_SCALE
    motion_true = y_motion * MOTION_SCALE

    return {
        "action_mae_full": float(np.mean(np.abs(action_full - y_action))),
        "action_mae_repeat_latest": float(np.mean(np.abs(action_rep - y_action))),
        "xdot_mae_full": float(np.mean(np.abs(motion_full[:, 0] - motion_true[:, 0]))),
        "xdot_mae_repeat_latest": float(np.mean(np.abs(motion_rep[:, 0] - motion_true[:, 0]))),
        "thetadot_mae_full": float(np.mean(np.abs(motion_full[:, 1] - motion_true[:, 1]))),
        "thetadot_mae_repeat_latest": float(np.mean(np.abs(motion_rep[:, 1] - motion_true[:, 1]))),
    }


def evaluate_closed_loop(model: TinyVisionTransformer, repeat_latest: bool, episodes: int = 60, horizon: int = 500):
    rng = np.random.default_rng(SEED + 91)
    lengths = []
    for episode in range(episodes):
        state = initial_state(rng, hard=True, zero_velocity=True)
        first = visual_observation(state)
        history = [first.copy() for _ in range(SEQ_LEN)]
        pulse_left, disturbance = 0, 0.0
        steps = 0

        for step in range(horizon):
            model_history = [history[-1].copy() for _ in range(SEQ_LEN)] if repeat_latest else history
            force, _ = model_outputs(model, model_history)

            if pulse_left <= 0 and step > 18 and rng.random() < 0.012:
                pulse_left = int(rng.integers(3, 7))
                disturbance = float(rng.choice([-4.0, 4.0]))
            if pulse_left > 0:
                pulse_left -= 1
            else:
                disturbance = 0.0

            state = physics_step(state, force, disturbance)
            history = history[1:] + [visual_observation(state)]
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
    return {
        "weight": layer.weight.detach().cpu().tolist(),
        "bias": layer.bias.detach().cpu().tolist(),
    }


def save_artifact(model, losses, ablation, closed_full, closed_repeat):
    artifact = {
        "format": "cartpole-vision-transformer-v1",
        "seed": SEED,
        "sequence_length": SEQ_LEN,
        "frame_size": FRAME_SIZE,
        "patch_size": PATCH_SIZE,
        "grid_size": GRID_SIZE,
        "feature_dim": FEATURE_DIM,
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "motion_scale": MOTION_SCALE.tolist(),
        "training": {
            "objective": "pixels-only behavior cloning + auxiliary velocity inference",
            "epochs": len(losses),
            "final_train_loss": losses[-1][0],
            "final_val_action_mse": losses[-1][1],
            "final_val_motion_mse": losses[-1][2],
        },
        "ablation": ablation,
        "closed_loop": {
            "full_history": closed_full,
            "repeat_latest_frame": closed_repeat,
        },
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
            "action": serial_linear(model.action),
            "motion": serial_linear(model.motion),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")

    ratio_x = ablation["xdot_mae_repeat_latest"] / max(1e-9, ablation["xdot_mae_full"])
    ratio_th = ablation["thetadot_mae_repeat_latest"] / max(1e-9, ablation["thetadot_mae_full"])
    DOC_PATH.write_text(
        f"""# Vision-only tiny Transformer

Generated deterministically by `scripts/train_vision_transformer.py`.

## Input contract

`32×32 grayscale frame → 4×4 patches → 8×8 = 64 patch-average features → learned 64→16 frame token`.

The policy receives eight visual observations and **never receives simulator state directly**.

## Architecture

- sequence length: {SEQ_LEN}
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
| x_dot MAE (m/s) | {ablation['xdot_mae_full']:.4f} | {ablation['xdot_mae_repeat_latest']:.4f} |
| theta_dot MAE (rad/s) | {ablation['thetadot_mae_full']:.4f} | {ablation['thetadot_mae_repeat_latest']:.4f} |

Removing temporal information increases x_dot error by {ratio_x:.2f}× and theta_dot error by {ratio_th:.2f}×.

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
    x, action_y, motion_y = make_dataset()
    print("dataset", x.shape, action_y.shape, motion_y.shape)
    model, losses, val = train_model(x, action_y, motion_y)
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
