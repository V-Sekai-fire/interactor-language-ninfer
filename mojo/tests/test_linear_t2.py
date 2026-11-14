"""The GPU ternary linear against an FP64 reference, under the engine's own tolerance.

The engine's CUDA kernels cannot run on this machine, but Mojo's GPU target can: this runs the Mojo
kernel on whatever accelerator the host has (Apple silicon through Metal here) and holds it to the
criterion of `tests/ops/linear/linear_test_common.cpp` (relative L2 and the gross bound of
`tests/ops/op_check.h`, both in units of the BF16 roundoff 2^-8). Weights are random legal T2G128
planes: codes {00, 01, 11}, binary16 group scales, K not a multiple of 128 in one case."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOJO = shutil.which("mojo")
pytestmark = pytest.mark.skipif(MOJO is None, reason="needs mojo")

ROUNDOFF = 1.0 / 256.0
REL_L2, GROSS_ABS, GROSS_REL = ROUNDOFF, ROUNDOFF, 2.0 * ROUNDOFF


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return (((u + ((u >> 16) & 1) + 0x7FFF) >> 16)).astype(np.uint16)


def unbf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def make_weight(rng, n, k):
    groups = -(-k // 128)
    codes = rng.integers(0, 3, size=(n, groups * 128), dtype=np.uint8)
    codes[codes == 2] = 3
    codes[:, k:] = 0  # padding lanes past the logical K are zero
    packed = np.zeros((n, groups, 32), np.uint8)
    view = codes.reshape(n, groups, 32, 4)
    for i in range(4):
        packed |= view[..., i] << (2 * i)
    scales = rng.uniform(0.01, 0.2, size=(n, groups)).astype(np.float16)
    w = np.where(codes == 1, 1.0, np.where(codes == 3, -1.0, 0.0))[:, :k]
    dense = w * np.repeat(scales.astype(np.float64), 128, axis=1)[:, :k]
    return packed, scales, dense


def run_kernel(tmp_path, packed, scales, act, n, k, t):
    packed.tofile(tmp_path / "q.bin")
    scales.tofile(tmp_path / "s.bin")
    bf16(act).tofile(tmp_path / "a.bin")
    out = subprocess.run(
        [MOJO, "run", "-I", str(ROOT / "mojo"), str(HERE / "linear_t2_probe.mojo"), str(n), str(k), str(t),
         str(tmp_path / "q.bin"), str(tmp_path / "s.bin"), str(tmp_path / "a.bin")],
        capture_output=True, text=True, timeout=900,
    )
    assert out.returncode == 0, out.stderr[-2500:]
    rows = [[float(v) for v in line.split()] for line in out.stdout.strip().split("\n")[-t:]]
    return np.array(rows, np.float64)


def check(actual, reference, label):
    assert np.isfinite(actual).all(), f"{label}: non-finite output"
    err = actual - reference
    rel = np.sqrt((err ** 2).sum()) / max(np.sqrt((reference ** 2).sum()), 1e-30)
    gross = np.abs(err).max()
    limit = GROSS_ABS + GROSS_REL * np.abs(reference).max()
    assert rel <= REL_L2, f"{label}: relative L2 {rel:.3e} > {REL_L2:.3e}"
    assert gross <= limit, f"{label}: max abs error {gross:.3e} > {limit:.3e}"
    return rel


@pytest.mark.parametrize("n,k", [(1024, 5120), (130, 5120), (512, 200), (5120, 6144), (34816, 5120)])
def test_t2_linear_matches_fp64(tmp_path, n, k):
    rng = np.random.default_rng(n * 31 + k)
    packed, scales, dense = make_weight(rng, n, k)
    widest = 129 if n <= 5120 else 2
    act_all = unbf16(bf16(rng.normal(size=(widest, k)))).astype(np.float64)
    ts = [1, 2, 17, 65, 129] if n <= 5120 else [1, 2]
    for t in ts:
        if t > widest:
            continue
        got = run_kernel(tmp_path, packed, scales, act_all[:t], n, k, t)
        ref = act_all[:t] @ dense.T
        # The output is stored as BF16, so compare against the BF16-rounded activations' FP64 result.
        check(got, ref, f"n={n} k={k} t={t}")


def test_t2_linear_detects_a_wrong_code(tmp_path):
    """The comparison must be able to fail: flip the sign of every code in one row."""
    rng = np.random.default_rng(5)
    n, k, t = 64, 256, 3
    packed, scales, dense = make_weight(rng, n, k)
    act = unbf16(bf16(rng.normal(size=(t, k)))).astype(np.float64)
    wrong = packed.copy()
    wrong[7] = packed[7] ^ ((packed[7] & 0x55) << 1)  # 01 <-> 11 in every field of row 7
    got = run_kernel(tmp_path, wrong, scales, act, n, k, t)
    with pytest.raises(AssertionError):
        check(got, act @ dense.T, "mutated")
