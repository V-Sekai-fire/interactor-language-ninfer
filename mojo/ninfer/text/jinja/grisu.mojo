"""Grisu2 shortest-digit generation exactly as nlohmann::json (`detail::dtoa_impl`) does it.

Grisu2 is not always the shortest representation (1e23 prints as 9.999999999999999e+22), so
matching the C++ output means porting the algorithm and its table, not using Mojo's own float
formatting."""

from std.collections import List

# `f e k` of the 79 cached powers of ten, from nlohmann/json.hpp (kCachedPowers).
comptime _CACHED = """ab70fe17c79ac6ca -1060 -300
ff77b1fcbebcdc4f -1034 -292
be5691ef416bd60c -1007 -284
8dd01fad907ffc3c -980 -276
d3515c2831559a83 -954 -268
9d71ac8fada6c9b5 -927 -260
ea9c227723ee8bcb -901 -252
aecc49914078536d -874 -244
823c12795db6ce57 -847 -236
c21094364dfb5637 -821 -228
9096ea6f3848984f -794 -220
d77485cb25823ac7 -768 -212
a086cfcd97bf97f4 -741 -204
ef340a98172aace5 -715 -196
b23867fb2a35b28e -688 -188
84c8d4dfd2c63f3b -661 -180
c5dd44271ad3cdba -635 -172
936b9fcebb25c996 -608 -164
dbac6c247d62a584 -582 -156
a3ab66580d5fdaf6 -555 -148
f3e2f893dec3f126 -529 -140
b5b5ada8aaff80b8 -502 -132
87625f056c7c4a8b -475 -124
c9bcff6034c13053 -449 -116
964e858c91ba2655 -422 -108
dff9772470297ebd -396 -100
a6dfbd9fb8e5b88f -369 -92
f8a95fcf88747d94 -343 -84
b94470938fa89bcf -316 -76
8a08f0f8bf0f156b -289 -68
cdb02555653131b6 -263 -60
993fe2c6d07b7fac -236 -52
e45c10c42a2b3b06 -210 -44
aa242499697392d3 -183 -36
fd87b5f28300ca0e -157 -28
bce5086492111aeb -130 -20
8cbccc096f5088cc -103 -12
d1b71758e219652c -77 -4
9c40000000000000 -50 4
e8d4a51000000000 -24 12
ad78ebc5ac620000 3 20
813f3978f8940984 30 28
c097ce7bc90715b3 56 36
8f7e32ce7bea5c70 83 44
d5d238a4abe98068 109 52
9f4f2726179a2245 136 60
ed63a231d4c4fb27 162 68
b0de65388cc8ada8 189 76
83c7088e1aab65db 216 84
c45d1df942711d9a 242 92
924d692ca61be758 269 100
da01ee641a708dea 295 108
a26da3999aef774a 322 116
f209787bb47d6b85 348 124
b454e4a179dd1877 375 132
865b86925b9bc5c2 402 140
c83553c5c8965d3d 428 148
952ab45cfa97a0b3 455 156
de469fbd99a05fe3 481 164
a59bc234db398c25 508 172
f6c69a72a3989f5c 534 180
b7dcbf5354e9bece 561 188
88fcf317f22241e2 588 196
cc20ce9bd35c78a5 614 204
98165af37b2153df 641 212
e2a0b5dc971f303a 667 220
a8d9d1535ce3b396 694 228
fb9b7cd9a4a7443c 720 236
bb764c4ca7a44410 747 244
8bab8eefb6409c1a 774 252
d01fef10a657842c 800 260
9b10a4e5e9913129 827 268
e7109bfba19c0c9d 853 276
ac2820d9623bf429 880 284
80444b5e7aa7cf85 907 292
bf21e44003acdd2d 933 300
8e679c2f5e44ff8f 960 308
d433179d9c8cb841 986 316
9e19db92b4e31ba9 1013 324
"""

comptime _ALPHA = -60
comptime _GAMMA = -32


struct DiyFp(Copyable, Movable):
    var f: UInt64
    var e: Int

    def __init__(out self, f: UInt64, e: Int):
        self.f = f
        self.e = e


def _mul(x: DiyFp, y: DiyFp) -> DiyFp:
    var u_lo = x.f & 0xFFFFFFFF
    var u_hi = x.f >> 32
    var v_lo = y.f & 0xFFFFFFFF
    var v_hi = y.f >> 32
    var p0 = u_lo * v_lo
    var p1 = u_lo * v_hi
    var p2 = u_hi * v_lo
    var p3 = u_hi * v_hi
    var p0_hi = p0 >> 32
    var p1_lo = p1 & 0xFFFFFFFF
    var p1_hi = p1 >> 32
    var p2_lo = p2 & 0xFFFFFFFF
    var p2_hi = p2 >> 32
    var q = p0_hi + p1_lo + p2_lo
    q += UInt64(1) << 31
    var h = p3 + p2_hi + p1_hi + (q >> 32)
    return DiyFp(h, x.e + y.e + 64)


def _normalize(x: DiyFp) -> DiyFp:
    var f = x.f
    var e = x.e
    while (f >> 63) == 0:
        f <<= 1
        e -= 1
    return DiyFp(f, e)


def _cached_index(e: Int) -> Int:
    var f = _ALPHA - e - 1
    var num = f * 78913
    # C++ integer division truncates toward zero
    var k = num // (1 << 18) if num >= 0 else -((-num) // (1 << 18))
    if f > 0:
        k += 1
    return (300 + k + 7) // 8


def _atoi(s: StringSpan) -> Int:
    var neg = False
    var v = 0
    for b in s.as_bytes():
        if b == 0x2D:
            neg = True
        else:
            v = v * 10 + (Int(b) - 0x30)
    return -v if neg else v


def _cached_power(index: Int) -> DiyFp:
    var n = 0
    for line in _CACHED.split("\n"):
        if line.byte_length() == 0:
            continue
        if n == index:
            var parts = line.split(" ")
            return DiyFp(_hex64(parts[0]), _atoi(parts[1]))
        n += 1
    return DiyFp(0, 0)


def _hex64(s: StringSpan) -> UInt64:
    var v: UInt64 = 0
    for b in s.as_bytes():
        var c = Int(b)
        v = (v << 4) | UInt64(c - 0x30 if c <= 0x39 else c - 0x61 + 10)
    return v


def _pow10(n: UInt32) -> Tuple[Int, UInt32]:
    if n >= 1000000000:
        return (10, UInt32(1000000000))
    if n >= 100000000:
        return (9, UInt32(100000000))
    if n >= 10000000:
        return (8, UInt32(10000000))
    if n >= 1000000:
        return (7, UInt32(1000000))
    if n >= 100000:
        return (6, UInt32(100000))
    if n >= 10000:
        return (5, UInt32(10000))
    if n >= 1000:
        return (4, UInt32(1000))
    if n >= 100:
        return (3, UInt32(100))
    if n >= 10:
        return (2, UInt32(10))
    return (1, UInt32(1))


def _round(mut buf: List[UInt8], dist: UInt64, delta: UInt64, rest_in: UInt64, ten_k: UInt64):
    var rest = rest_in
    while (
        rest < dist
        and delta - rest >= ten_k
        and (rest + ten_k < dist or dist - rest > rest + ten_k - dist)
    ):
        buf[len(buf) - 1] -= 1
        rest += ten_k


def _digit_gen(
    mut buf: List[UInt8], mut decimal_exponent: Int, m_minus: DiyFp, w: DiyFp, m_plus: DiyFp
):
    var delta = m_plus.f - m_minus.f
    var dist = m_plus.f - w.f
    var one_e = m_plus.e
    var one_f = UInt64(1) << UInt64(-one_e)
    var p1 = UInt32(m_plus.f >> UInt64(-one_e))
    var p2 = m_plus.f & (one_f - 1)
    var pk = _pow10(p1)
    var n = pk[0]
    var pow10 = pk[1]
    while n > 0:
        var d = p1 // pow10
        var r = p1 % pow10
        buf.append(UInt8(0x30 + Int(d)))
        p1 = r
        n -= 1
        var rest = (UInt64(p1) << UInt64(-one_e)) + p2
        if rest <= delta:
            decimal_exponent += n
            _round(buf, dist, delta, rest, UInt64(pow10) << UInt64(-one_e))
            return
        pow10 //= 10
    var m = 0
    while True:
        p2 *= 10
        var d = p2 >> UInt64(-one_e)
        var r = p2 & (one_f - 1)
        buf.append(UInt8(0x30 + Int(d)))
        p2 = r
        m += 1
        delta *= 10
        dist *= 10
        if p2 <= delta:
            break
    decimal_exponent -= m
    _round(buf, dist, delta, p2, one_f)


def _append_exponent(mut out: String, e_in: Int):
    var e = e_in
    if e < 0:
        e = -e
        out += "-"
    else:
        out += "+"
    if e < 10:
        out += "0" + String(e)
    elif e < 100:
        out += String(e)
    else:
        out += String(e)


def dump_double(value_in: Float64) -> String:
    """`nlohmann::json(double).dump()` for a finite value."""
    var negative = value_in < 0.0 or (value_in == 0.0 and String(value_in).startswith("-"))
    var value = -value_in if negative else value_in
    var sign = String("-" if negative else "")
    if value == 0.0:
        return sign + "0.0"
    var bits = UInt64(value.to_bits())
    var E = Int((bits >> 52) & 0x7FF)
    var F = bits & ((UInt64(1) << 52) - 1)
    var is_denormal = E == 0
    var v = DiyFp(F, -1074) if is_denormal else DiyFp(F + (UInt64(1) << 52), E - 1075)
    var lower_closer = F == 0 and E > 1
    var plus = DiyFp(2 * v.f + 1, v.e - 1)
    var minus = DiyFp(4 * v.f - 1, v.e - 2) if lower_closer else DiyFp(2 * v.f - 1, v.e - 1)
    var w_plus = _normalize(plus)
    var delta_e = minus.e - w_plus.e
    var w_minus = DiyFp(minus.f << UInt64(delta_e), w_plus.e)
    var w = _normalize(v)
    var index = _cached_index(w_plus.e)
    var cached = _cached_power(index)
    var decimal_exponent = 300 - index * 8  # -k of the cached power
    var wm = _mul(w, cached)
    var wmm = _mul(w_minus, cached)
    var wpp = _mul(w_plus, cached)
    var buf = List[UInt8]()
    _digit_gen(buf, decimal_exponent, DiyFp(wmm.f + 1, wmm.e), wm, DiyFp(wpp.f - 1, wpp.e))
    var digits = String(unsafe_from_utf8=buf)
    var k = len(buf)
    var n = k + decimal_exponent
    var out = String()
    if k <= n and n <= 15:
        out = digits + "0" * (n - k) + ".0"
    elif 0 < n and n <= 15:
        out = String(digits[byte=0:n]) + "." + String(digits[byte=n:])
    elif -4 < n and n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        out = String(digits[byte=0:1])
        if k > 1:
            out += "." + String(digits[byte=1:])
        out += "e"
        _append_exponent(out, n - 1)
    return sign + out
