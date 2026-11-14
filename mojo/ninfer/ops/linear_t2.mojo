"""Ternary (T2G128_F16S) linear on the GPU; mirrors the contract of `src/ops/linear/t2`.

Weights are row-split planes, row-major per plane: for output row `n` and group `g` of 128
inputs, 32 code bytes (four 2-bit two's-complement codes per byte, lane `i` in bits `(i % 4) * 2`
of byte `i / 4`: 00 is 0, 01 is +1, 11 is -1) and one binary16 scale. Activations are BF16
`[t, k]`, the output is BF16 `[t, n]`, accumulation is FP32.

One warp computes one output element: lane `l` decodes byte `l` of each group (four codes), scales
the lane's partial sum by the group scale, and a warp reduction adds the lanes. This is the
straightforward schedule, not the engine's tuned ones; the tests hold it to the engine's tolerance
against an FP64 reference.

Kernel scalars are `Int32`: `Int` is not device passable.
"""

from std.math import ceildiv

from max.gpu import block_idx, lane_id
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.primitives import warp

comptime T2_GROUP_K = 128
comptime T2_CODE_BYTES_PER_GROUP = 32


def t2_linear_kernel(
    qdata: ImmPointer[UInt8, ImmUntrackedOrigin],
    scales: ImmPointer[Float16, ImmUntrackedOrigin],
    act: ImmPointer[BFloat16, ImmUntrackedOrigin],
    dst: MutPointer[BFloat16, MutUntrackedOrigin],
    n: Int32,
    k: Int32,
    groups: Int32,
):
    var row = Int(block_idx.x)
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
                var code = Int((byte >> UInt8(2 * j)) & 3)
                var w = Float32(0)
                if code == 1:
                    w = 1.0
                elif code == 3:
                    w = -1.0
                partial += w * act[unsafe_offset=t * logical_k + base + j].cast[DType.float32]()
        acc += partial * scales[unsafe_offset=row * Int(groups) + g].cast[DType.float32]()
    var total = warp.sum(acc)
    if lane == 0:
        dst[unsafe_offset=t * Int(n) + row] = total.cast[DType.bfloat16]()


def t2_linear(
    ctx: DeviceContext,
    qdata: DeviceBuffer[DType.uint8],
    scales: DeviceBuffer[DType.float16],
    act: DeviceBuffer[DType.bfloat16],
    dst: DeviceBuffer[DType.bfloat16],
    n: Int,
    k: Int,
    t: Int,
) raises:
    """Enqueues `dst[t, n] = act[t, k] @ W[n, k]^T` for a T2G128_F16S weight; `k` is the logical
    input width, the planes hold `ceildiv(k, 128)` groups per row."""
    var groups = ceildiv(k, T2_GROUP_K)
    ctx.enqueue_function[t2_linear_kernel](
        qdata,
        scales,
        act,
        dst,
        Int32(n),
        Int32(k),
        Int32(groups),
        grid_dim=(n, t),
        block_dim=32,
    )
