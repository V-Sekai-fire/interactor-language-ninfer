"""The GPU ternary linear against a NumPy model of its own arithmetic, bit for bit.

The kernels fix their summation order in the source (see `ninfer/ops/linear_t2.mojo`), so the result
is a function of the inputs and not of the device or the schedule. The model here does the same
arithmetic with `float32`/`int32` NumPy operations in that order; the comparison is on the BF16 bit
patterns, with no tolerance. A second set of checks holds the same kernels to the engine's own
accuracy criterion (`tests/ops/linear/linear_test_common.cpp`) against an FP64 reference, which is
what ties the model to the mathematics rather than only to itself.

Controls (each must change the verdict): a weight with one row's signs flipped, and a reference with
no spread (a bit comparison of zeros proves nothing). The comparison is on BF16 output, which hides
a last-bit FP32 difference; the model is checked to tell summation orders apart in FP32.

Mojo's GPU target runs on whatever accelerator the host has (Apple silicon through Metal here). A
host without `mojo` fails this module at collection; nothing is skipped.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOJO = shutil.which("mojo")
assert MOJO is not None, "mojo is not on PATH: this module cannot run, and a skip would read as a pass"

ROUNDOFF = 1.0 / 256.0
G = 128


# --- number formats ----------------------------------------------------------------------------


def to_bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + ((u >> 16) & 1) + 0x7FFF) >> 16).astype(np.uint16)


def from_bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def bf16_round(x):
    return from_bf16(to_bf16(x))


# --- weights -----------------------------------------------------------------------------------


def make_weight(rng, n, k, legal=True):
    groups = -(-k // G)
    codes = rng.integers(0, 4, size=(n, groups * G), dtype=np.uint8)
    if legal:
        codes[codes == 2] = 3
    codes[:, k:] = 0  # padding lanes past the logical K are zero
    packed = np.zeros((n, groups, 32), np.uint8)
    view = codes.reshape(n, groups, 32, 4)
    for i in range(4):
        packed |= view[..., i] << (2 * i)
    scales = rng.uniform(0.01, 0.2, size=(n, groups)).astype(np.float16)
    mag, sign = codes & 1, (codes >> 1) & 1
    w = (mag.astype(np.int32) - 2 * (mag & sign).astype(np.int32))  # the engine's decode, 10 -> 0
    return packed, scales, w


def activations(rng, t, k):
    return bf16_round(rng.normal(size=(t, k)) * rng.choice([0.01, 1.0, 30.0], size=(t, 1)))


# --- the model ---------------------------------------------------------------------------------


def butterfly(arr):
    """(.., 32) -> (..): lane l adds lane l ^ 16, then ^ 8, ^ 4, ^ 2, ^ 1; every lane ends with the sum."""
    idx = np.arange(32)
    for off in (16, 8, 4, 2, 1):
        arr = arr + arr[..., idx ^ off]
    return arr[..., 0]


def model_a16(w, scales, x, rows, row_begin, order="butterfly"):
    n_all, kp = w.shape
    groups = kp // G
    t, k = x.shape
    xp = np.zeros((t, groups * G), np.float32)
    xp[:, :k] = x
    wr = w[row_begin:row_begin + rows].reshape(rows, groups, 32, 4).astype(np.float32)
    sc = scales[row_begin:row_begin + rows].astype(np.float32)
    xv = xp.reshape(t, groups, 32, 4)
    acc = np.zeros((t, rows, 32), np.float32)
    for g in range(groups):
        partial = np.zeros((t, rows, 32), np.float32)
        for j in range(4):
            partial = (partial + wr[None, :, g, :, j] * xv[:, None, g, :, j]).astype(np.float32)
        acc = (acc + (partial * sc[None, :, g, None]).astype(np.float32)).astype(np.float32)
    if order == "butterfly":
        return butterfly(acc)
    return acc.sum(axis=-1, dtype=np.float32)  # a different order: the control


def quantize(x, groups):
    t, k = x.shape
    xp = np.zeros((t, groups * G), np.float32)
    xp[:, :k] = x
    xg = xp.reshape(t, groups, G)
    amax = np.abs(xg).max(axis=-1)
    scale = (amax / np.float32(127)).astype(np.float16)
    s = scale.astype(np.float32)
    inv = np.where(s > 0, np.float32(1) / np.where(s > 0, s, np.float32(1)), np.float32(0)).astype(np.float32)
    q = np.clip(np.rint((xg * inv[..., None]).astype(np.float32)), -127, 127).astype(np.int8)
    return q.reshape(t, groups * G), scale


def model_a8(w, scales, x, rows, row_begin):
    groups = w.shape[1] // G
    t = x.shape[0]
    q, xs = quantize(x, groups)
    qg = q.reshape(t, groups, G).astype(np.int64)
    wg = w[row_begin:row_begin + rows].reshape(rows, groups, G).astype(np.int64)
    sw = scales[row_begin:row_begin + rows].astype(np.float32)
    sx = xs.astype(np.float32)
    acc = np.zeros((t, rows), np.float32)
    for g in range(groups):
        dot = (qg[:, None, g, :] * wg[None, :, g, :]).sum(axis=-1).astype(np.float32)  # exact
        acc = (acc + ((dot * sw[None, :, g]).astype(np.float32) * sx[:, None, g]).astype(np.float32)).astype(np.float32)
    return acc


def place(result_f32, t, rows, out_ld, out_off, accumulate, init_bf16):
    """What the kernel stores: the plane `init`, with columns [out_off, out_off + rows) replaced by
    the BF16 rounding of the result (plus the old value, in FP32, when accumulating)."""
    plane = from_bf16(init_bf16).reshape(t, out_ld).copy()
    cols = result_f32.astype(np.float32)
    if accumulate:
        cols = (cols + plane[:, out_off:out_off + rows]).astype(np.float32)
    plane[:, out_off:out_off + rows] = bf16_round(cols)
    return plane


# --- running the kernels -----------------------------------------------------------------------


def run(tmp_path, args):
    out = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(HERE / "linear_t2_probe.mojo"), *map(str, args)],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2500:]
    return out.stdout.strip("\n").split("\n")


def run_linear(tmp_path, op, packed, scales, x, k, row_begin, rows, out_ld, out_off, accumulate, init_bf16):
    t = x.shape[0]
    packed.tofile(tmp_path / "q.bin")
    scales.tofile(tmp_path / "s.bin")
    to_bf16(x).tofile(tmp_path / "a.bin")
    init_bf16.tofile(tmp_path / "i.bin")
    lines = run(tmp_path, [op, k, t, row_begin, rows, out_ld, out_off, int(accumulate),
                           tmp_path / "q.bin", tmp_path / "s.bin", tmp_path / "a.bin", tmp_path / "i.bin"])
    return np.array([[np.float32(v) for v in line.split()] for line in lines[-t:]], np.float32)


def assert_bits_equal(got, want, label):
    gb, wb = to_bf16(got), to_bf16(want)
    bad = np.argwhere(gb != wb)
    assert bad.size == 0, f"{label}: {len(bad)} of {gb.size} outputs differ; first at {bad[0].tolist()}: got {got[tuple(bad[0])]!r}, want {want[tuple(bad[0])]!r}"


def assert_informative(want, label):
    """A bit comparison of a constant or all-zero reference proves nothing."""
    assert np.count_nonzero(want) > 0.9 * want.size and np.unique(want).size > min(8, want.size), f"{label}: degenerate reference"


CASES = [  # (n_all, k, t, row_begin, rows, out_ld, out_off, accumulate)
    (64, 5120, 1, 0, 64, 64, 0, False),
    (96, 200, 5, 0, 96, 96, 0, False),                # K not a multiple of 128
    (130, 6144, 3, 17, 70, 70, 0, False),             # a row view of the parent
    (130, 5120, 4, 30, 40, 100, 25, False),           # written into a column range of a wider plane
    (96, 384, 6, 10, 48, 48, 0, True),                # linear_add: added into what is there
    (257, 17408, 2, 0, 257, 257, 0, False),
]


@pytest.mark.parametrize("n,k,t,row_begin,rows,out_ld,out_off,accumulate", CASES)
@pytest.mark.parametrize("op", ["a16", "a8"])
def test_linear_matches_model_bit_for_bit(tmp_path, op, n, k, t, row_begin, rows, out_ld, out_off, accumulate):
    rng = np.random.default_rng(n * 7 + k + t)
    packed, scales, w = make_weight(rng, n, k, legal=False)  # illegal 10 codes too: they decode to 0
    x = activations(rng, t, k)
    init = to_bf16(rng.normal(size=(t, out_ld)) * 3) if accumulate else np.zeros((t, out_ld), np.uint16)
    got = run_linear(tmp_path, op, packed, scales, x, k, row_begin, rows, out_ld, out_off, accumulate, init)
    result = model_a16(w, scales, x, rows, row_begin) if op == "a16" else model_a8(w, scales, x, rows, row_begin)
    want = place(result, t, rows, out_ld, out_off, accumulate, init)
    assert_informative(want[:, out_off:out_off + rows], f"{op} n={n} k={k}")
    assert_bits_equal(got, want, f"{op} n={n} k={k} t={t} rows={rows} [{row_begin}:+{rows}] -> ld {out_ld} +{out_off} acc={accumulate}")
    # the columns outside the written range keep what was there
    keep = np.ones(out_ld, bool)
    keep[out_off:out_off + rows] = False
    assert_bits_equal(got[:, keep], from_bf16(init).reshape(t, out_ld)[:, keep], "untouched columns")


@pytest.mark.parametrize("k,t", [(5120, 3), (200, 4), (300, 2)])
def test_quantization_matches_model_bit_for_bit(tmp_path, k, t):
    rng = np.random.default_rng(k + t)
    x = activations(rng, t, k)
    x[0, :G] = 0.0                  # a zero group keeps zero codes and a zero scale
    x[1, 5] = 1e4                   # one outlier squeezes the group's other codes
    x = bf16_round(x)               # the kernel sees BF16
    to_bf16(x).tofile(tmp_path / "a.bin")
    lines = run(tmp_path, ["quant", k, t, tmp_path / "a.bin"])
    groups = -(-k // G)
    q, xs = quantize(x, groups)
    got_q = np.array([[int(v) for v in line.split()] for line in lines[:t]], np.int8)
    got_s = np.array([[np.float32(v) for v in line.split()] for line in lines[t:2 * t]], np.float32)
    assert np.array_equal(got_q, q), f"codes differ at {np.argwhere(got_q != q)[:3].tolist()}"
    assert np.array_equal(got_s.astype(np.float16).view(np.uint16), xs.view(np.uint16)), "scales differ"
    assert (xs[0, 0] == 0) and not q[0, :G].any()
    assert np.abs(q).max() == 127 and np.count_nonzero(q) > 0.3 * q.size


# --- the engine's accuracy criterion -----------------------------------------------------------


def criterion(path):
    """`tests/ops/linear/linear_test_common.cpp`: A16 at one BF16 roundoff, A8 at 0.04."""
    return (ROUNDOFF, ROUNDOFF, 2 * ROUNDOFF) if path == "a16" else (0.04, ROUNDOFF, 1.5 * 0.04)


@pytest.mark.parametrize("op", ["a16", "a8"])
@pytest.mark.parametrize("n,k", [(1024, 5120), (130, 5120), (512, 200), (5120, 6144), (34816, 5120)])
def test_linear_meets_the_engines_criterion_against_fp64(tmp_path, op, n, k):
    rng = np.random.default_rng(n * 31 + k)
    packed, scales, w = make_weight(rng, n, k)
    dense = w[:, :k] * np.repeat(scales.astype(np.float64), G, axis=1)[:, :k]
    rel_l2, gross_abs, gross_rel = criterion(op)
    for t in ([1, 2, 17, 65, 129] if n <= 5120 else [1, 2]):
        x = bf16_round(rng.normal(size=(t, k)))
        got = run_linear(tmp_path, op, packed, scales, x, k, 0, n, n, 0, False, np.zeros((t, n), np.uint16)).astype(np.float64)
        ref = x.astype(np.float64) @ dense.T
        assert np.isfinite(got).all() and np.abs(ref).max() > 0
        err = got - ref
        rel = np.sqrt((err ** 2).sum()) / np.sqrt((ref ** 2).sum())
        assert rel <= rel_l2, f"{op} n={n} k={k} t={t}: relative L2 {rel:.3e} > {rel_l2:.3e}"
        limit = gross_abs + gross_rel * np.abs(ref).max()
        assert np.abs(err).max() <= limit, f"{op} n={n} k={k} t={t}: max abs error {np.abs(err).max():.3e} > {limit:.3e}"


# --- SwiGLU ------------------------------------------------------------------------------------


def test_swiglu_matches_model(tmp_path):
    rng = np.random.default_rng(11)
    rows, k, t = 192, 5120, 5
    packed, scales, w = make_weight(rng, rows, k)
    x = bf16_round(rng.normal(size=(t, k)))  # pre-activations near 1, where silu is not saturated
    packed.tofile(tmp_path / "q.bin"); scales.tofile(tmp_path / "s.bin"); to_bf16(x).tofile(tmp_path / "a.bin")
    lines = run(tmp_path, ["swiglu", k, t, rows, tmp_path / "q.bin", tmp_path / "s.bin", tmp_path / "a.bin"])
    got = np.array([[np.float32(v) for v in line.split()] for line in lines[-t:]], np.float32)
    plane = bf16_round(model_a16(w, scales, x, rows, 0))                        # the projection is bit-exact
    gate, up = plane[:, :rows // 2], plane[:, rows // 2:]
    want = bf16_round((gate / (np.float32(1) + np.exp(-gate))).astype(np.float32) * up)
    assert_informative(want, "swiglu")
    # exp() is the device's own, so allow one BF16 step on the rare element where the exponentials differ in the last bit
    step = np.abs(to_bf16(got).astype(np.int32) - to_bf16(want).astype(np.int32))
    assert step.max() <= 1 and (step == 0).mean() > 0.99, f"swiglu: max {step.max()} steps, {(step == 0).mean():.4f} exact"


# --- controls ----------------------------------------------------------------------------------


def test_the_bit_comparison_can_fail(tmp_path):
    """A flipped row and a constant reference must be caught; the model must tell the summation orders apart."""
    rng = np.random.default_rng(5)
    n, k, t = 64, 5120, 4
    packed, scales, w = make_weight(rng, n, k)
    x = activations(rng, t, k)
    init = np.zeros((t, n), np.uint16)
    got = run_linear(tmp_path, "a16", packed, scales, x, k, 0, n, n, 0, False, init)
    want = place(model_a16(w, scales, x, n, 0), t, n, n, 0, False, init)
    assert_bits_equal(got, want, "baseline")
    flipped = packed.copy()
    flipped[7] = packed[7] ^ ((packed[7] & 0x55) << 1)  # 01 <-> 11 in every field of row 7
    wrong = run_linear(tmp_path, "a16", flipped, scales, x, k, 0, n, n, 0, False, init)
    with pytest.raises(AssertionError):
        assert_bits_equal(wrong, want, "flipped row")
    # The output is BF16, which hides a last-bit difference in the FP32 sum: the order shows in the
    # FP32 results of the model, not in the stored output. Say so by checking the model sees it.
    f32_butterfly, f32_plain = model_a16(w, scales, x, n, 0), model_a16(w, scales, x, n, 0, order="sum")
    assert not np.array_equal(f32_butterfly, f32_plain), "the order control cannot see a different order"
    with pytest.raises(AssertionError):
        assert_informative(np.zeros((t, n), np.float32), "zeros")
