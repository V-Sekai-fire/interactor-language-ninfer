"""UTF-8 decoding, Unicode categories and NFC; mirrors `src/text/unicode.{h,cpp}`.

The C++ engine gets these from utf8proc. The tables here are generated from that same library
(`mojo/tools/gen_unicode_tables.py`), and the UTF-8 decoder follows `utf8proc_iterate` rule for
rule, so both engines split, classify and normalize every input identically.
"""

from std.collections import Dict, List

from ninfer.text.unicode_tables import (
    CLASS_RANGES,
    CCC_RANGES,
    COMPOSITIONS,
    DECOMPOSITIONS,
)

comptime CLASS_NONE = 0
comptime CLASS_LETTER = 1
comptime CLASS_MARK = 2
comptime CLASS_NUMBER = 3
comptime CLASS_SEPARATOR = 4

comptime INVALID_UTF8 = "Invalid UTF-8 string"

# Hangul syllable arithmetic (Unicode ch. 3.12).
comptime _S_BASE = 0xAC00
comptime _L_BASE = 0x1100
comptime _V_BASE = 0x1161
comptime _T_BASE = 0x11A7
comptime _L_COUNT = 19
comptime _V_COUNT = 21
comptime _T_COUNT = 28
comptime _N_COUNT = 588
comptime _S_COUNT = 11172


struct CodepointSpan(Copyable, Movable):
    var value: Int
    var offset: Int
    var length: Int

    def __init__(out self, value: Int, offset: Int, length: Int):
        self.value = value
        self.offset = offset
        self.length = length


def utf8_codepoint_at(
    text: List[UInt8], offset: Int, context: String
) raises -> CodepointSpan:
    """Decode one code point. Validation is the standard library's (`String(from_utf8=)`), which
    rejects what `utf8proc_iterate` rejects: overlong forms, surrogates, values past U+10FFFF
    and truncated sequences; the parity tests pin that equivalence."""
    if offset >= len(text):
        raise Error("UTF-8 codepoint offset exceeds " + context)
    var lead = Int(text[offset])
    if lead < 0x80:
        return CodepointSpan(lead, offset, 1)
    var bad = "invalid UTF-8 in " + context + ": " + INVALID_UTF8
    if lead < 0xC2 or lead > 0xF4:
        raise Error(bad)
    var width = 2 if lead < 0xE0 else (3 if lead < 0xF0 else 4)
    if len(text) - offset < width:
        raise Error(bad)
    var piece = List[UInt8](capacity=width)
    for i in range(width):
        piece.append(text[offset + i])
    try:
        var s = String(from_utf8=piece)
        return CodepointSpan(Int(ord(s)), offset, width)
    except:
        raise Error(bad)


def utf8_codepoints(text: List[UInt8], context: String) raises -> List[CodepointSpan]:
    """Validate the whole buffer with the standard library, then walk its code points."""
    var out = List[UInt8]()
    var s: String
    try:
        s = String(from_utf8=text)
    except:
        # find the offending sequence so the message matches the C++ loader's
        var offset = 0
        while offset < len(text):
            var span = utf8_codepoint_at(text, offset, context)
            offset += span.length
        raise Error("invalid UTF-8 in " + context + ": " + INVALID_UTF8)
    var spans = List[CodepointSpan]()
    var offset = 0
    for cp in s.codepoints():
        var width = cp.utf8_byte_length()
        spans.append(CodepointSpan(Int(cp.to_u32()), offset, width))
        offset += width
    return spans^


def append_utf8(mut out: List[UInt8], codepoint: Int) raises:
    """`codepoint_to_utf8`: rejects surrogates and values past U+10FFFF; encoding is `chr`."""
    if codepoint < 0 or codepoint > 0x10FFFF or (codepoint >= 0xD800 and codepoint <= 0xDFFF):
        raise Error("invalid Unicode codepoint")
    for b in chr(codepoint).as_bytes():
        out.append(b)


def codepoint_to_utf8(codepoint: Int) raises -> List[UInt8]:
    var out = List[UInt8]()
    append_utf8(out, codepoint)
    return out^


def _hex(s: StringSpan) -> Int:
    var v = 0
    for b in s.as_bytes():
        var c = Int(b)
        v = v * 16 + (c - 0x30 if c <= 0x39 else c - 0x61 + 10)
    return v


struct _Ranges(Movable):
    var first: List[Int32]
    var last: List[Int32]
    var value: List[Int32]

    def __init__(out self, text: StaticString):
        self.first = List[Int32]()
        self.last = List[Int32]()
        self.value = List[Int32]()
        for line in text.split("\n"):
            if line.byte_length() == 0:
                continue
            var f = line.split(" ")
            self.first.append(Int32(_hex(f[0])))
            self.last.append(Int32(_hex(f[1])))
            self.value.append(Int32(_hex(f[2])))

    def lookup(self, cp: Int) -> Int:
        var lo = 0
        var hi = len(self.first) - 1
        while lo <= hi:
            var mid = (lo + hi) // 2
            if cp < Int(self.first[mid]):
                hi = mid - 1
            elif cp > Int(self.last[mid]):
                lo = mid + 1
            else:
                return Int(self.value[mid])
        return 0


struct Unicode(Movable):
    """Parsed tables. Build once (the tokenizer owns one) and reuse."""

    var classes: _Ranges
    var ccc: _Ranges
    var decomp_index: Dict[Int, Int]  # cp -> start in decomp_parts
    var decomp_parts: List[Int32]  # [count, p0, p1, ...] per entry
    var compose: Dict[Int, Int]  # (first << 21 | second) -> composite

    def __init__(out self):
        self.classes = _Ranges(CLASS_RANGES)
        self.ccc = _Ranges(CCC_RANGES)
        self.decomp_index = Dict[Int, Int]()
        self.decomp_parts = List[Int32]()
        for line in DECOMPOSITIONS.split("\n"):
            if line.byte_length() == 0:
                continue
            var halves = line.split(":")
            var parts = halves[1].split(" ")
            self.decomp_index[_hex(halves[0])] = len(self.decomp_parts)
            self.decomp_parts.append(Int32(len(parts)))
            for p in parts:
                self.decomp_parts.append(Int32(_hex(p)))
        self.compose = Dict[Int, Int]()
        for line in COMPOSITIONS.split("\n"):
            if line.byte_length() == 0:
                continue
            var f = line.split(" ")
            self.compose[(_hex(f[0]) << 21) | _hex(f[1])] = _hex(f[2])

    def is_letter(self, cp: Int) -> Bool:
        return self.classes.lookup(cp) == CLASS_LETTER

    def is_mark(self, cp: Int) -> Bool:
        return self.classes.lookup(cp) == CLASS_MARK

    def is_number(self, cp: Int) -> Bool:
        return self.classes.lookup(cp) == CLASS_NUMBER

    def is_whitespace(self, cp: Int) -> Bool:
        # ASCII whitespace plus every Z* category (utf8proc has no U+0009..000D in Z).
        if cp == 0x20 or (cp >= 0x09 and cp <= 0x0D):
            return True
        return self.classes.lookup(cp) == CLASS_SEPARATOR

    def combining_class(self, cp: Int) -> Int:
        return self.ccc.lookup(cp)

    def _decompose(self, cp: Int, mut out: List[Int]):
        var s = cp - _S_BASE
        if s >= 0 and s < _S_COUNT:
            out.append(_L_BASE + s // _N_COUNT)
            out.append(_V_BASE + (s % _N_COUNT) // _T_COUNT)
            if s % _T_COUNT != 0:
                out.append(_T_BASE + s % _T_COUNT)
            return
        var found = self.decomp_index.get(cp)
        if found:
            var at = found.value()
            var count = Int(self.decomp_parts[at])
            for i in range(count):
                out.append(Int(self.decomp_parts[at + 1 + i]))
            return
        out.append(cp)

    def _compose_pair(self, a: Int, b: Int) -> Int:
        # Hangul: L + V and LV + T.
        var l = a - _L_BASE
        if l >= 0 and l < _L_COUNT:
            var v = b - _V_BASE
            if v >= 0 and v < _V_COUNT:
                return _S_BASE + (l * _V_COUNT + v) * _T_COUNT
        var s = a - _S_BASE
        if s >= 0 and s < _S_COUNT and s % _T_COUNT == 0:
            var t = b - _T_BASE
            if t > 0 and t < _T_COUNT:
                return a + t
        var found = self.compose.get((a << 21) | b)
        if found:
            return found.value()
        return -1

    def normalize_nfc(self, text: List[UInt8]) raises -> List[UInt8]:
        """`utf8proc_map(UTF8PROC_STABLE | UTF8PROC_COMPOSE)`."""
        var ascii_only = True
        for b in text:
            if b >= 0x80:
                ascii_only = False
                break
        if ascii_only:
            return text.copy()
        var cps = List[Int]()
        var offset = 0
        while offset < len(text):
            var span: CodepointSpan
            try:
                span = utf8_codepoint_at(text, offset, "")
            except:
                raise Error("failed to normalize UTF-8 text as NFC: " + INVALID_UTF8)
            self._decompose(span.value, cps)
            offset += span.length
        # Canonical ordering: stable sort of each run of non-starters by combining class.
        var i = 1
        while i < len(cps):
            var c = self.combining_class(cps[i])
            if c != 0:
                var j = i
                while j > 0:
                    var p = self.combining_class(cps[j - 1])
                    if p <= c:
                        break
                    var tmp = cps[j]
                    cps[j] = cps[j - 1]
                    cps[j - 1] = tmp
                    j -= 1
            i += 1
        # Canonical composition.
        var out = List[Int]()
        var starter = -1
        var last_cc = 0
        for k in range(len(cps)):
            var c = cps[k]
            var cc = self.combining_class(c)
            if starter >= 0 and (len(out) - 1 == starter or last_cc < cc):
                var composite = self._compose_pair(out[starter], c)
                if composite >= 0:
                    out[starter] = composite
                    continue
            if cc == 0:
                starter = len(out)
            out.append(c)
            last_cc = cc
        var result = List[UInt8]()
        for c in out:
            append_utf8(result, c)
        return result^
