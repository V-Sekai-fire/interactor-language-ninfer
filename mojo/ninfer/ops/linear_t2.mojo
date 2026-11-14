"""Ternary (T2G128_F16S) linear on the GPU; mirrors the contracts of `src/ops/linear/t2`,
`wrapper/linear_add` and `wrapper/linear_swiglu`.

Weights are row-split planes, row-major per plane: for output row `n` and group `g` of 128 inputs,
32 code bytes (four 2-bit codes per byte, lane `i` in bits `(i % 4) * 2` of byte `i / 4`) and one
binary16 scale. A code decodes as `magnitude - 2 * (magnitude & sign)` from its two bit planes
(`t2_rowsplit_storage.cuh`): 00 is 0, 01 is +1, 11 is -1, and the illegal 10 is 0.

Two routes, as in the engine:
- A16: BF16 activations `[t, k]`, FP32 accumulation, BF16 output `[t, n]`.
- A8 (`t2_a8_quantize` then `t2_a8_linear`): activations are quantized once per token and 128-wide
  group to int8 with one binary16 scale (`amax / 127`, codes `clamp(rne(x / scale), +-127)`, a zero
  group keeps zero codes and a zero scale); the product is an exact int32 dot per group, scaled by
  the weight and activation scales in FP32.

The summation order is fixed in the source, so the result is a function of the inputs and not of
the device: a lane adds its four products left to right, scales the group's partial and adds it to
its accumulator group after group, and the 32 lanes combine in a butterfly (offsets 16, 8, 4, 2, 1).
`tests/test_linear_t2.py` holds the kernels to a NumPy model of exactly that order, bit for bit.

Every projection takes a row range of its weight (`row_begin`, `rows`: the engine's row views) and
writes into a column range of a wider output (`out_ld`, `out_offset`: the split projections of the
GDN and attention wrappers), optionally adding into what is there (`accumulate`: `linear_add`).
`t2_linear_swiglu` projects the gate/up plane and applies SwiGLU as the engine's separate pass.

One warp computes one output element, the plain schedule rather than the engine's tuned decode,
small-T and prefill variants. Kernel scalars are `Int32`: `Int` is not device passable.
"""

from std.math import ceildiv, exp, floor, round

from max.gpu import block_idx, thread_idx, lane_id
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.primitives import warp

comptime T2_GROUP_K = 128
comptime T2_CODE_BYTES_PER_GROUP = 32


@always_inline
def _decode(code: Int) -> Int:
    """`magnitude - 2 * (magnitude & sign)` of a two-bit code."""
    var magnitude = code & 1
    var sign = (code >> 1) & 1
    return magnitude - 2 * (magnitude & sign)


@always_inline
def _round_even(x: Float32) -> Float32:
    """`__float2int_rn`: round to nearest, ties to even."""
    var f = floor(x)
    if x - f == 0.5:
        if floor(f * 0.5) * 2.0 == f:
            return f
        return f + 1.0
    return round(x)


@always_inline
def _butterfly_sum(value: Float32) -> Float32:
    """The 32 lanes' sum in a fixed order: lane l adds lane l ^ 16, then ^ 8, ^ 4, ^ 2, ^ 1."""
    var v = value
    v += warp.shuffle_xor(v, UInt32(16))
    v += warp.shuffle_xor(v, UInt32(8))
    v += warp.shuffle_xor(v, UInt32(4))
    v += warp.shuffle_xor(v, UInt32(2))
    v += warp.shuffle_xor(v, UInt32(1))
    return v


# --- A16 ---------------------------------------------------------------------------------------


def t2_linear_kernel(
    qdata: ImmPointer[UInt8, ImmUntrackedOrigin],
    scales: ImmPointer[Float16, ImmUntrackedOrigin],
    act: ImmPointer[BFloat16, ImmUntrackedOrigin],
    dst: MutPointer[BFloat16, MutUntrackedOrigin],
    k: Int32,
    groups: Int32,
    row_begin: Int32,
    out_ld: Int32,
    out_offset: Int32,
    accumulate: Int32,
):
    var local = Int(block_idx.x)
    var row = Int(row_begin) + local
    var t = Int(block_idx.y)
    var lane = Int(lane_id())
    var logical_k = Int(k)
    var acc = Float32(0)
    for g in range(Int(groups)):
        var byte = qdata[unsafe_offset=(row * Int(groups) + g) * T2_CODE_BYTES_PER_GROUP + lane]
        var base = g * T2_GROUP_K + lane * 4
        var partial = Float32(0)
        for j in range(4):
            if base + j < logical_k:  # codes past the logical K are padding
                var w = _decode(Int((byte >> UInt8(2 * j)) & 3))
                partial += Float32(w) * act[unsafe_offset=t * logical_k + base + j].cast[DType.float32]()
        var scale = scales[unsafe_offset=row * Int(groups) + g].cast[DType.float32]()
        acc += partial * scale
    var total = _butterfly_sum(acc)
    if lane == 0:
        var at = t * Int(out_ld) + Int(out_offset) + local
        if accumulate != 0:
            total += dst[unsafe_offset=at].cast[DType.float32]()
        dst[unsafe_offset=at] = total.cast[DType.bfloat16]()


def t2_linear(
    ctx: DeviceContext,
    qdata: DeviceBuffer[DType.uint8],
    scales: DeviceBuffer[DType.float16],
    act: DeviceBuffer[DType.bfloat16],
    dst: DeviceBuffer[DType.bfloat16],
    k: Int,
    t: Int,
    row_begin: Int,
    rows: Int,
    out_ld: Int,
    out_offset: Int = 0,
    accumulate: Bool = False,
) raises:
    """`dst[t, out_offset + r] (+)= act[t, :k] . W[row_begin + r, :k]` for r in [0, rows). The planes
    hold `ceildiv(k, 128)` groups per row."""
    ctx.enqueue_function[t2_linear_kernel](
        qdata,
        scales,
        act,
        dst,
        Int32(k),
        Int32(ceildiv(k, T2_GROUP_K)),
        Int32(row_begin),
        Int32(out_ld),
        Int32(out_offset),
        Int32(1 if accumulate else 0),
        grid_dim=(rows, t),
        block_dim=32,
    )


# --- A8 ----------------------------------------------------------------------------------------


def t2_a8_quantize_kernel(
    act: ImmPointer[BFloat16, ImmUntrackedOrigin],
    codes: MutPointer[Int8, MutUntrackedOrigin],
    xscales: MutPointer[Float16, MutUntrackedOrigin],
    k: Int32,
    groups: Int32,
):
    """One block per (group, token), a lane per four inputs. The codes are padded to whole groups
    with zeros; the scale plane is `[token][group]`."""
    var g = Int(block_idx.x)
    var t = Int(block_idx.y)
    var lane = Int(lane_id())
    var logical_k = Int(k)
    var base = g * T2_GROUP_K + lane * 4
    var f = SIMD[DType.float32, 4](0)
    for j in range(4):
        if base + j < logical_k:
            f[j] = act[unsafe_offset=t * logical_k + base + j].cast[DType.float32]()
    var amax = Float32(0)
    for j in range(4):
        amax = max(amax, abs(f[j]))
    amax = warp.max(amax)
    var scale = (amax / 127.0).cast[DType.float16]()
    if lane == 0:
        xscales[unsafe_offset=t * Int(groups) + g] = scale
    var s = scale.cast[DType.float32]()
    var inv = Float32(0)
    if s > 0.0:
        inv = 1.0 / s
    for j in range(4):
        var q = _round_even(f[j] * inv)
        q = max(Float32(-127), min(Float32(127), q))
        codes[unsafe_offset=t * Int(groups) * T2_GROUP_K + base + j] = Int8(Int(q))


def t2_a8_quantize(
    ctx: DeviceContext,
    act: DeviceBuffer[DType.bfloat16],
    codes: DeviceBuffer[DType.int8],
    xscales: DeviceBuffer[DType.float16],
    k: Int,
    t: Int,
) raises:
    """Quantizes `act[t, k]` once; the planes serve any number of `t2_a8_linear` calls over the
    same activations. `codes` holds `t * ceildiv(k, 128) * 128` bytes, `xscales` one binary16 per
    token and group."""
    var groups = ceildiv(k, T2_GROUP_K)
    ctx.enqueue_function[t2_a8_quantize_kernel](
        act, codes, xscales, Int32(k), Int32(groups), grid_dim=(groups, t), block_dim=32
    )


def t2_a8_linear_kernel(
    qdata: ImmPointer[UInt8, ImmUntrackedOrigin],
    scales: ImmPointer[Float16, ImmUntrackedOrigin],
    codes: ImmPointer[Int8, ImmUntrackedOrigin],
    xscales: ImmPointer[Float16, ImmUntrackedOrigin],
    dst: MutPointer[BFloat16, MutUntrackedOrigin],
    groups: Int32,
    row_begin: Int32,
    out_ld: Int32,
    out_offset: Int32,
    accumulate: Int32,
):
    var local = Int(block_idx.x)
    var row = Int(row_begin) + local
    var t = Int(block_idx.y)
    var lane = Int(lane_id())
    var acc = Float32(0)
    for g in range(Int(groups)):
        var byte = qdata[unsafe_offset=(row * Int(groups) + g) * T2_CODE_BYTES_PER_GROUP + lane]
        var base = t * Int(groups) * T2_GROUP_K + g * T2_GROUP_K + lane * 4
        var partial = Int32(0)
        for j in range(4):
            var w = _decode(Int((byte >> UInt8(2 * j)) & 3))
            partial += Int32(w) * Int32(codes[unsafe_offset=base + j])
        var dot = warp.sum(partial)  # integers: any order gives the same exact sum
        var sw = scales[unsafe_offset=row * Int(groups) + g].cast[DType.float32]()
        var sx = xscales[unsafe_offset=t * Int(groups) + g].cast[DType.float32]()
        acc += Float32(dot) * sw * sx
    if lane == 0:
        var at = t * Int(out_ld) + Int(out_offset) + local
        if accumulate != 0:
            acc += dst[unsafe_offset=at].cast[DType.float32]()
        dst[unsafe_offset=at] = acc.cast[DType.bfloat16]()


def t2_a8_linear(
    ctx: DeviceContext,
    qdata: DeviceBuffer[DType.uint8],
    scales: DeviceBuffer[DType.float16],
    codes: DeviceBuffer[DType.int8],
    xscales: DeviceBuffer[DType.float16],
    dst: DeviceBuffer[DType.bfloat16],
    k: Int,
    t: Int,
    row_begin: Int,
    rows: Int,
    out_ld: Int,
    out_offset: Int = 0,
    accumulate: Bool = False,
) raises:
    """The integer route over activations from `t2_a8_quantize`; rows and outputs as `t2_linear`."""
    ctx.enqueue_function[t2_a8_linear_kernel](
        qdata,
        scales,
        codes,
        xscales,
        dst,
        Int32(ceildiv(k, T2_GROUP_K)),
        Int32(row_begin),
        Int32(out_ld),
        Int32(out_offset),
        Int32(1 if accumulate else 0),
        grid_dim=(rows, t),
        block_dim=32,
    )


# --- fused wrappers ----------------------------------------------------------------------------


def t2_silu_mul_kernel(
    gate_up: ImmPointer[BFloat16, ImmUntrackedOrigin],
    dst: MutPointer[BFloat16, MutUntrackedOrigin],
    inter: Int32,
    total: Int32,
):
    """`dst[t, r] = silu(gate_up[t, r]) * gate_up[t, inter + r]`, `silu(x) = x / (1 + exp(-x))`."""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i >= Int(total):
        return
    var t = i // Int(inter)
    var r = i % Int(inter)
    var g = gate_up[unsafe_offset=t * 2 * Int(inter) + r].cast[DType.float32]()
    var u = gate_up[unsafe_offset=t * 2 * Int(inter) + Int(inter) + r].cast[DType.float32]()
    dst[unsafe_offset=i] = (g / (1.0 + exp(-g)) * u).cast[DType.bfloat16]()


def t2_linear_swiglu(
    ctx: DeviceContext,
    qdata: DeviceBuffer[DType.uint8],
    scales: DeviceBuffer[DType.float16],
    act: DeviceBuffer[DType.bfloat16],
    plane: DeviceBuffer[DType.bfloat16],
    dst: DeviceBuffer[DType.bfloat16],
    k: Int,
    t: Int,
    gate_up_rows: Int,
) raises:
    """Gate and up are the two halves of the weight's rows: project into `plane[t, gate_up_rows]`
    (A16, as the engine's ternary wrapper does), then SwiGLU into `dst[t, gate_up_rows / 2]`."""
    t2_linear(ctx, qdata, scales, act, plane, k, t, 0, gate_up_rows, gate_up_rows)
    var inter = gate_up_rows // 2
    var total = t * inter
    ctx.enqueue_function[t2_silu_mul_kernel](
        plane, dst, Int32(inter), Int32(total), grid_dim=ceildiv(total, 256), block_dim=256
    )
