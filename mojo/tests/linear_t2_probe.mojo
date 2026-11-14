"""Runs the GPU ternary linear ops for tests/test_linear_t2.py.

    quant   k t act                                   codes (ints) then scales, one token per line
    a16     k t row_begin rows out_ld out_off acc q s act init   the output plane, one token per line
    a8      (same arguments as a16)                   quantizes act, then the integer route
    swiglu  k t gate_up_rows q s act                  the SwiGLU output, one token per line

Planes are little-endian binary files (q: bytes, s: binary16, act/init: BF16). Values print as
decimal Float32, which is exact for BF16 and binary16.
"""

from std.math import ceildiv
from std.pathlib import Path
from std.sys import argv

from max.gpu.host import DeviceContext

from ninfer.ops.linear_t2 import (
    T2_GROUP_K,
    t2_a8_linear,
    t2_a8_quantize,
    t2_linear,
    t2_linear_swiglu,
)


def _u16(data: List[UInt8], i: Int) -> UInt16:
    return UInt16(data[2 * i]) | (UInt16(data[2 * i + 1]) << 8)


def main() raises:
    var args = argv()
    var op = String(args[1])
    var ctx = DeviceContext()
    if op == "quant":
        var k = atol(String(args[2]))
        var t = atol(String(args[3]))
        var a_bytes = Path(String(args[4])).read_bytes()
        var groups = ceildiv(k, T2_GROUP_K)
        var a = ctx.enqueue_create_buffer[DType.bfloat16](t * k)
        var codes = ctx.enqueue_create_buffer[DType.int8](t * groups * T2_GROUP_K)
        var xs = ctx.enqueue_create_buffer[DType.float16](t * groups)
        with a.map_to_host() as h:
            for i in range(t * k):
                h[i] = BFloat16(from_bits=_u16(a_bytes, i))
        codes.enqueue_fill(Int8(0))
        xs.enqueue_fill(Float16(0))
        t2_a8_quantize(ctx, a, codes, xs, k, t)
        ctx.synchronize()
        with codes.map_to_host() as h:
            for row in range(t):
                var line = String()
                for i in range(groups * T2_GROUP_K):
                    line += String(Int(h[row * groups * T2_GROUP_K + i])) + " "
                print(line)
        with xs.map_to_host() as h:
            for row in range(t):
                var line = String()
                for g in range(groups):
                    line += String(h[row * groups + g].cast[DType.float32]()) + " "
                print(line)
        return

    if op == "swiglu":
        var k = atol(String(args[2]))
        var t = atol(String(args[3]))
        var rows = atol(String(args[4]))
        var q_bytes = Path(String(args[5])).read_bytes()
        var s_bytes = Path(String(args[6])).read_bytes()
        var a_bytes = Path(String(args[7])).read_bytes()
        var q = ctx.enqueue_create_buffer[DType.uint8](len(q_bytes))
        var s = ctx.enqueue_create_buffer[DType.float16](len(s_bytes) // 2)
        var a = ctx.enqueue_create_buffer[DType.bfloat16](t * k)
        var plane = ctx.enqueue_create_buffer[DType.bfloat16](t * rows)
        var inter = rows // 2
        var o = ctx.enqueue_create_buffer[DType.bfloat16](t * inter)
        with q.map_to_host() as h:
            for i in range(len(q_bytes)):
                h[i] = q_bytes[i]
        with s.map_to_host() as h:
            for i in range(len(s_bytes) // 2):
                h[i] = Float16(from_bits=_u16(s_bytes, i))
        with a.map_to_host() as h:
            for i in range(t * k):
                h[i] = BFloat16(from_bits=_u16(a_bytes, i))
        plane.enqueue_fill(BFloat16(0.0))
        o.enqueue_fill(BFloat16(0.0))
        t2_linear_swiglu(ctx, q, s, a, plane, o, k, t, rows)
        ctx.synchronize()
        with o.map_to_host() as h:
            for row in range(t):
                var line = String()
                for col in range(inter):
                    line += String(h[row * inter + col].cast[DType.float32]()) + " "
                print(line)
        return

    var k = atol(String(args[2]))
    var t = atol(String(args[3]))
    var row_begin = atol(String(args[4]))
    var rows = atol(String(args[5]))
    var out_ld = atol(String(args[6]))
    var out_off = atol(String(args[7]))
    var accumulate = atol(String(args[8])) != 0
    var q_bytes = Path(String(args[9])).read_bytes()
    var s_bytes = Path(String(args[10])).read_bytes()
    var a_bytes = Path(String(args[11])).read_bytes()
    var init_bytes = Path(String(args[12])).read_bytes()
    var groups = ceildiv(k, T2_GROUP_K)
    var q = ctx.enqueue_create_buffer[DType.uint8](len(q_bytes))
    var s = ctx.enqueue_create_buffer[DType.float16](len(s_bytes) // 2)
    var a = ctx.enqueue_create_buffer[DType.bfloat16](t * k)
    var o = ctx.enqueue_create_buffer[DType.bfloat16](t * out_ld)
    with q.map_to_host() as h:
        for i in range(len(q_bytes)):
            h[i] = q_bytes[i]
    with s.map_to_host() as h:
        for i in range(len(s_bytes) // 2):
            h[i] = Float16(from_bits=_u16(s_bytes, i))
    with a.map_to_host() as h:
        for i in range(t * k):
            h[i] = BFloat16(from_bits=_u16(a_bytes, i))
    with o.map_to_host() as h:
        for i in range(t * out_ld):
            h[i] = BFloat16(from_bits=_u16(init_bytes, i))
    if op == "a16":
        t2_linear(ctx, q, s, a, o, k, t, row_begin, rows, out_ld, out_off, accumulate)
    else:
        var codes = ctx.enqueue_create_buffer[DType.int8](t * groups * T2_GROUP_K)
        var xs = ctx.enqueue_create_buffer[DType.float16](t * groups)
        t2_a8_quantize(ctx, a, codes, xs, k, t)
        t2_a8_linear(ctx, q, s, codes, xs, o, k, t, row_begin, rows, out_ld, out_off, accumulate)
    ctx.synchronize()
    with o.map_to_host() as h:
        for row in range(t):
            var line = String()
            for col in range(out_ld):
                line += String(h[row * out_ld + col].cast[DType.float32]()) + " "
            print(line)
