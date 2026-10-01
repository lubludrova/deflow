"""Read-only training telemetry and an exclusive capsule for a failed update."""

from collections import deque
import json
import math
from pathlib import Path
import random

import numpy as np
import torch


def tensor_stats(named_tensors):

    count = nan = inf = 0
    norms, maxima, nonfinite_names = [], [], []
    for name, value in named_tensors:
        value = value.detach()
        count += value.numel()
        n_nan = int(torch.isnan(value).sum().item())
        n_inf = int(torch.isinf(value).sum().item())
        nan += n_nan
        inf += n_inf
        if n_nan or n_inf:
            nonfinite_names.append(name)
        elif value.numel():
            converted = value.to(dtype=torch.float64)
            norms.append(float(torch.linalg.vector_norm(converted).item()))
            maxima.append(float(converted.abs().max().item()))
    finite = not (nan or inf)
    norm = math.hypot(*norms)
    return {"numel": count, "all_finite": finite, "nan_count": nan,
            "inf_count": inf, "nonfinite_names": nonfinite_names,
            "l2": norm if finite and math.isfinite(norm) else None,
            "max_abs": max(maxima, default=0.0) if finite else None}


def _snapshot(value):
    if isinstance(value, torch.nn.Module):
        return {"state_dict": _snapshot(value.state_dict()),
                "gradients": {name: _snapshot(p.grad) for name, p in value.named_parameters()},
                "training": value.training}
    if isinstance(value, torch.optim.Optimizer):
        return _snapshot(value.state_dict())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _scalar(value):
    if isinstance(value, dict):
        return {key: _scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    if isinstance(value, torch.Tensor):
        value = value.detach().item()
    if isinstance(value, np.generic):
        value = value.item()
    return value if not isinstance(value, float) or math.isfinite(value) else str(value)


class DiagnosticRecorder:
    def __init__(self, run_dir, metadata, interval=1000, history_size=32):
        if interval < 1 or history_size < 1:
            raise ValueError("diagnostic interval and history size must be positive")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = _snapshot(metadata)
        self.interval = interval
        self.recent = deque(maxlen=history_size)
        self._bucket = self._sample_step = None
        self.path = self.run_dir / "training_diagnostics.jsonl"
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": "docking_training_diagnostics_v1",
                                     "metadata": metadata, "interval": interval},
                                    allow_nan=False) + "\n")

    def due(self, step):
        bucket = int(step) // self.interval
        if bucket != self._bucket:
            self._bucket, self._sample_step = bucket, int(step)
        return int(step) == self._sample_step

    def observe(self, step, phase, scalars):

        row = {"step": int(step), "phase": phase,
               "scalars": {name: _scalar(value) for name, value in scalars.items()}}
        self.recent.append(row)
        return row

    def record(self, step, phase, modules=None, optimizers=None, tensors=None, force=False,
               scalars=None):
        if not force and not self.due(step):
            return None
        row = {"step": int(step), "phase": phase, "modules": {},
               "optimizers": {}, "tensors": {},
               "scalars": {name: _scalar(value) for name, value in (scalars or {}).items()}}
        for name, module in (modules or {}).items():
            parameters = list(module.named_parameters())
            gradients = [(key, p.grad) for key, p in parameters if p.grad is not None]
            row["modules"][name] = {
                "parameters": tensor_stats(parameters), "gradients": tensor_stats(gradients),
                "parameters_without_gradient": len(parameters) - len(gradients),
            }
        for name, optimizer in (optimizers or {}).items():
            if optimizer is None:
                continue
            groups = {}
            for index, state in enumerate(optimizer.state.values()):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        groups.setdefault(key, []).append((str(index), value))
            row["optimizers"][name] = {key: tensor_stats(values) for key, values in groups.items()}
        for name, tensor in (tensors or {}).items():
            if tensor is None:
                continue
            row["tensors"][name] = tensor_stats([(name, tensor)])
            if tensor.numel() == 1:
                row["tensors"][name]["value"] = _scalar(tensor)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.recent.append(row)
        return row

    def capture_failure(self, step, reason, payload):

        rng = {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch_cpu": torch.get_rng_state(),
               "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
        contract = self.run_dir / "run_contract.json"
        capsule = {"schema": "docking_training_failure_v1", "step": int(step),
                   "reason": str(reason), "metadata": self.metadata,
                   "recent": list(self.recent), "rng": _snapshot(rng),
                   "run_contract": json.loads(contract.read_text()) if contract.exists() else None,
                   "payload": _snapshot(payload)}
        path = self.run_dir / f"failure_step_{int(step):09d}.pt"
        with path.open("xb") as stream:
            torch.save(capsule, stream)
        return path
