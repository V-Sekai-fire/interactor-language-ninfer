"""Runs the GPU ternary linear. argv: n k t qdata.bin scales.bin act.bin; prints the BF16 output as
one `[t, n]` row of decimal values (exact BF16) per line."""

from std.pathlib import Path
from std.sys import argv

from max.gpu.host import DeviceContext

from ninfer.ops.linear_t2 import t2_linear


def main() raises:
    var args = argv()
    var n = atol(String(args[1]))
    var k = atol(String(args[2]))
    var t = atol(String(args[3]))
    var q_bytes = Path(String(args[4])).read_bytes()
    var s_bytes = Path(String(args[5])).read_bytes()
    var a_bytes = Path(String(args[6])).read_bytes()
    var ctx = DeviceContext()
    var q = ctx.enqueue_create_buffer[DType.uint8](len(q_bytes))
    var s = ctx.enqueue_create_buffer[DType.float16](len(s_bytes) // 2)
    var a = ctx.enqueue_create_buffer[DType.bfloat16](len(a_bytes) // 2)
    var o = ctx.enqueue_create_buffer[DType.bfloat16](t * n)
    with q.map_to_host() as h:
        for i in range(len(q_bytes)):
            h[i] = q_bytes[i]
    with s.map_to_host() as h:
        for i in range(len(s_bytes) // 2):
            h[i] = Float16(from_bits=UInt16(s_bytes[2 * i]) | (UInt16(s_bytes[2 * i + 1]) << 8))
    with a.map_to_host() as h:
        for i in range(len(a_bytes) // 2):
            h[i] = BFloat16(from_bits=UInt16(a_bytes[2 * i]) | (UInt16(a_bytes[2 * i + 1]) << 8))
    o.enqueue_fill(BFloat16(0.0))
    t2_linear(ctx, q, s, a, o, n, k, t)
    ctx.synchronize()
    with o.map_to_host() as h:
        for row in range(t):
            var line = String()
            for col in range(n):
                line += String(h[row * n + col].cast[DType.float32]()) + " "
            print(line)
