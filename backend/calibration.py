"""Confidence calibration helpers.

The production calibration parameter must be fitted on an independent validation set.
This module intentionally does not invent a calibration parameter when none exists.
"""
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = ROOT / "calibration.json"


def load_temperature(default=1.0):
    if not CALIBRATION_PATH.exists():
        return float(default), False
    try:
        payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        temperature = float(payload.get("temperature", default))
        if temperature <= 0:
            raise ValueError
        return temperature, True
    except Exception:
        return float(default), False


def calibrate_logits(logits):
    temperature, configured = load_temperature()
    return logits / temperature, temperature, configured


def fit_temperature(logits, labels, max_steps=500, lr=0.01):
    """Fit temperature scaling on held-out validation logits/labels."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_steps, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = loss_fn(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())
