"""Explicit numerical policy shared by training, evaluation, and audits."""
import hashlib
import json
import os
from pathlib import Path
import torch

POLICIES = ('legacy_tf32', 'strict_fp32')
OVERRIDES = ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')


def snapshot():
    return dict(matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                autocast_cuda=torch.is_autocast_enabled('cuda'),
                autocast_cpu=torch.is_autocast_enabled('cpu'),
                default_dtype=str(torch.get_default_dtype()),
                torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                backend_environment={k:os.environ[k] for k in OVERRIDES if k in os.environ})


def assert_policy(policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown precision policy: {policy}')
    state = snapshot()
    strict = policy == 'strict_fp32'
    if strict and (state['backend_environment'] or state['autocast_cuda'] or state['autocast_cpu']):
        raise RuntimeError(f'Strict FP32 rejects backend overrides/autocast: {state}')
    expected = ('highest', False, False) if strict else ('high', True, True)
    actual = (state['matmul_precision'], state['matmul_allow_tf32'], state['cudnn_allow_tf32'])
    if actual != expected or (strict and torch.get_default_dtype() != torch.float32):
        raise RuntimeError(f'Precision policy mismatch: {policy}: {state}')
    return state


def apply_policy(policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown precision policy: {policy}')
    if policy == 'strict_fp32' and any(k in os.environ for k in OVERRIDES):
        raise RuntimeError('Strict FP32 rejects backend override environment variables')
    strict = policy == 'strict_fp32'
    torch.set_float32_matmul_precision('highest' if strict else 'high')
    torch.backends.cuda.matmul.allow_tf32 = not strict
    torch.backends.cudnn.allow_tf32 = not strict
    return assert_policy(policy)


def assert_float32(policy, *tensors):
    assert_policy(policy)
    if policy == 'strict_fp32' and any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors):
        raise RuntimeError('Strict scientific execution requires float32 tensors')


def constructor_digest(kwargs):
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
