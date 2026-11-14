"""Jinja strings with provenance; mirrors `third_party/llama-jinja/jinja/{string,unicode}.{h,cpp}`.

A `JString` is a list of parts. A part carries its text, the id of the input region it came
from (`origin`, 0 for none), the byte offset inside that region (`source_offset`, -1 when a
transformation lost it) and whether it is ordinary content (`literal`). Origins and literal
flags never change equality or text; they exist so the renderer can tell which bytes of the
prompt came from a request field.
"""

from ninfer.text.jinja.error import JinjaError, codepoints, jinja_error, push_utf8
from std.collections import Dict, List

from ninfer.text.unicode import Unicode, append_utf8, utf8_codepoints, _Ranges, _hex
from ninfer.text.unicode_tables import (
    CASED_RANGES,
    CASE_IGNORABLE_RANGES,
    IS_LOWER_RANGES,
    IS_UPPER_RANGES,
    SIMPLE_CASE,
    SPECIAL_CASE,
)

comptime CASE_LOWER = 0
comptime CASE_UPPER = 1
comptime CASE_TITLE = 2
comptime CASE_CAPITALIZE = 3


struct StrPart(Copyable, Movable):
    var origin: Int
    var val: String
    var source_offset: Int  # -1: not available
    var literal: Bool

    def __init__(out self, origin: Int, var val: String, source_offset: Int, literal: Bool):
        self.origin = origin
        self.val = val^
        self.source_offset = source_offset
        self.literal = literal


struct Character(Copyable, Movable):
    var value: Int
    var begin: Int
    var end: Int

    def __init__(out self, value: Int, begin: Int, end: Int):
        self.value = value
        self.begin = begin
        self.end = end


struct JU(Movable):
    """Unicode services the string methods need; built once and reused."""

    var unicode: Unicode
    var cased: _Ranges
    var ignorable: _Ranges
    var upper_ranges: _Ranges
    var lower_ranges: _Ranges
    var simple: Dict[Int, Int]  # cp -> index into simple_lo/up/ti
    var simple_lo: List[Int]
    var simple_up: List[Int]
    var simple_ti: List[Int]
    var special: Dict[Int, Int]  # cp -> index into special_*
    var special_lo: List[String]
    var special_ti: List[String]
    var special_up: List[String]

    def __init__(out self):
        self.unicode = Unicode()
        self.cased = _Ranges(CASED_RANGES)
        self.ignorable = _Ranges(CASE_IGNORABLE_RANGES)
        self.upper_ranges = _Ranges(IS_UPPER_RANGES)
        self.lower_ranges = _Ranges(IS_LOWER_RANGES)
        self.simple = Dict[Int, Int]()
        self.simple_lo = List[Int]()
        self.simple_up = List[Int]()
        self.simple_ti = List[Int]()
        for line in SIMPLE_CASE.split("\n"):
            if line.byte_length() == 0:
                continue
            var f = line.split(" ")
            self.simple[_hex(f[0])] = len(self.simple_lo)
            self.simple_lo.append(_hex(f[1]))
            self.simple_up.append(_hex(f[2]))
            self.simple_ti.append(_hex(f[3]))
        self.special = Dict[Int, Int]()
        self.special_lo = List[String]()
        self.special_ti = List[String]()
        self.special_up = List[String]()
        for line in SPECIAL_CASE.split("\n"):
            if line.byte_length() == 0:
                continue
            var f = line.split(" ")
            self.special[_hex(f[0])] = len(self.special_lo)
            self.special_lo.append(_unhex_string(f[1]))
            self.special_ti.append(_unhex_string(f[2]))
            self.special_up.append(_unhex_string(f[3]))

    def whitespace(self, cp: Int) -> Bool:
        return (cp >= 0x1C and cp <= 0x1F) or cp == 0x85 or self.unicode.is_whitespace(cp)

    def is_cased(self, cp: Int) -> Bool:
        return self.cased.lookup(cp) != 0

    def is_ignorable(self, cp: Int) -> Bool:
        return self.ignorable.lookup(cp) != 0


def _unhex_string(s: StringSpan) -> String:
    var out = List[UInt8]()
    var b = s.as_bytes()
    var i = 0
    while i + 1 < len(b):
        out.append(UInt8(_hex_digit(b[i]) * 16 + _hex_digit(b[i + 1])))
        i += 2
    return String(unsafe_from_utf8=out)


def _hex_digit(c: UInt8) -> Int:
    return Int(c) - 0x30 if c <= 0x39 else Int(c) - 0x61 + 10


def string_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def bytes_string(b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=b)


def sub_string(s: String, begin: Int, end: Int) -> String:
    var out = List[UInt8](capacity=end - begin)
    var b = s.as_bytes()
    for i in range(begin, end):
        out.append(b[i])
    return String(unsafe_from_utf8=out)


def characters(ju: JU, text: String) raises JinjaError -> List[Character]:
    var out = List[Character]()
    var spans = codepoints(string_bytes(text), "Jinja string")
    for sp in spans:
        out.append(Character(sp.value, sp.offset, sp.offset + sp.length))
    return out^


def codepoint_length(text: String) raises JinjaError -> Int:
    var n = 0
    for sp in codepoints(string_bytes(text), "Jinja string"):
        n += 1
    return n


struct Slice(Copyable, Movable):
    var start: Int
    var stop: Int
    var step: Int

    def __init__(out self, start: Int, stop: Int, step: Int):
        self.start = start
        self.stop = stop
        self.step = step


def _bound(value: Int, n: Int, lo: Int, hi: Int) -> Int:
    var v = value
    if v < 0:
        v += n
    return min(max(v, lo), hi)


def slice_indices(size: Int, start: Optional[Int], stop: Optional[Int], step: Int) raises JinjaError -> Slice:
    if step == 0:
        raise jinja_error("slice step cannot be zero")
    var n = size
    var lo = 0 if step > 0 else -1
    var hi = n if step > 0 else n - 1
    var s = 0 if step > 0 else n - 1
    if case .Some(first) = start:
        s = _bound(first, n, lo, hi)
    var e = n if step > 0 else -1
    if case .Some(last) = stop:
        e = _bound(last, n, lo, hi)
    return Slice(s, e, step)


struct JString(Copyable, Movable):
    var parts: List[StrPart]

    def __init__(out self):
        self.parts = List[StrPart]()

    def __init__(out self, var value: String, origin: Int = 0, literal: Bool = False):
        self.parts = List[StrPart]()
        self.parts.append(StrPart(origin, value^, 0, literal))

    def tag(mut self, origin: Int, exact: Bool = True):
        var offset = 0
        for i in range(len(self.parts)):
            self.parts[i].origin = origin
            self.parts[i].source_offset = offset if exact else -1
            offset += self.parts[i].val.byte_length()

    def str(self) -> String:
        var out = String()
        for p in self.parts:
            out += p.val
        return out^

    def byte_size(self) -> Int:
        var n = 0
        for p in self.parts:
            n += p.val.byte_length()
        return n

    def length(self) raises JinjaError -> Int:
        return codepoint_length(self.str())

    def append(mut self, other: JString):
        # `append(self)` aliasing is handled by copying `other` first.
        var incoming = other.parts.copy()
        for part in incoming:
            if len(self.parts) > 0:
                var last = len(self.parts) - 1
                var same_kind = self.parts[last].literal == part.literal
                var merge = same_kind and (
                    (self.parts[last].origin == 0 and part.origin == 0)
                    or (
                        self.parts[last].origin == part.origin
                        and self.parts[last].source_offset >= 0
                        and part.source_offset >= 0
                        and self.parts[last].source_offset + self.parts[last].val.byte_length() == part.source_offset
                    )
                )
                if merge:
                    self.parts[last].val += part.val
                    continue
            self.parts.append(part.copy())

    def append_str(mut self, text: String):
        self.append(JString(text))

    def cut_bytes(self, begin: Int, end: Int) raises JinjaError -> JString:
        var size = self.byte_size()
        if begin > end or end > size:
            raise jinja_error("Jinja byte slice exceeds string")
        if begin == 0 and end == size:
            return self.copy()
        var result = JString()
        var offset = 0
        for part in self.parts:
            var next = offset + part.val.byte_length()
            if begin < next and end > offset:
                var local_begin = max(begin, offset) - offset
                var local_end = min(end, next) - offset
                var out = StrPart(part.origin, sub_string(part.val, local_begin, local_end), -1, part.literal)
                if part.source_offset >= 0:
                    out.source_offset = part.source_offset + local_begin
                elif local_begin != 0 or local_end != part.val.byte_length():
                    out.origin = 0
                result.parts.append(out^)
            offset = next
        return result^

    def slice(self, ju: JU, start: Optional[Int], stop: Optional[Int], step: Int = 1) raises JinjaError -> JString:
        var text = self.str()
        var chars = characters(ju, text)
        var bounds = slice_indices(len(chars), start, stop, step)
        if step == 1:
            var begin = text.byte_length() if bounds.start == len(chars) else chars[bounds.start].begin
            var end: Int
            if bounds.stop <= bounds.start:
                end = begin
            elif bounds.stop == len(chars):
                end = text.byte_length()
            else:
                end = chars[bounds.stop].begin
            return self.cut_bytes(begin, end)
        var result = JString()
        var i = bounds.start
        while (i < bounds.stop) if step > 0 else (i > bounds.stop):
            result.append(self.cut_bytes(chars[i].begin, chars[i].end))
            if (step >= bounds.stop - i) if step > 0 else (step <= bounds.stop - i):
                break
            i += step
        return result^

    def split(
        self, ju: JU, separator: Optional[String], maxsplit_in: Int, reverse: Bool
    ) raises JinjaError -> List[JString]:
        var text = self.str()
        if case .Some(sep) = separator:
            if sep.byte_length() == 0:
                raise jinja_error("empty separator")
        var result = List[JString]()
        var begin = 0
        var end = text.byte_length()
        var chars = List[Character]()
        if not separator:
            chars = characters(ju, text)
        var first = 0
        var last = len(chars)
        var maxsplit = maxsplit_in
        var tb = text.as_bytes()
        while True:
            if not separator:
                if reverse:
                    while last > first and ju.whitespace(chars[last - 1].value):
                        last -= 1
                    end = chars[last - 1].end if last > 0 else 0
                else:
                    while first < last and ju.whitespace(chars[first].value):
                        first += 1
                    begin = chars[first].begin if first < last else text.byte_length()
                if first == last:
                    break
            if maxsplit == 0:
                result.append(self.cut_bytes(begin, end))
                break
            if case .Some(sep) = separator:
                var sb = sep.as_bytes()
                var pos = -1
                if end - begin >= len(sb):
                    if reverse:
                        # rfind starting at end - len(sep)
                        var k = end - len(sb)
                        while k >= 0:
                            var same = True
                            for q in range(len(sb)):
                                if tb[k + q] != sb[q]:
                                    same = False
                                    break
                            if same:
                                pos = k
                                break
                            k -= 1
                    else:
                        var k = begin
                        while k + len(sb) <= len(tb):
                            var same = True
                            for q in range(len(sb)):
                                if tb[k + q] != sb[q]:
                                    same = False
                                    break
                            if same:
                                pos = k
                                break
                            k += 1
                if pos < 0 or pos < begin or pos + len(sb) > end:
                    result.append(self.cut_bytes(begin, end))
                    break
                if reverse:
                    result.append(self.cut_bytes(pos + len(sb), end))
                    end = pos
                else:
                    result.append(self.cut_bytes(begin, pos))
                    begin = pos + len(sb)
            elif reverse:
                var word = last
                while word > first and not ju.whitespace(chars[word - 1].value):
                    word -= 1
                result.append(self.cut_bytes(chars[word].begin, end))
                last = word
            else:
                var word = first
                while word < last and not ju.whitespace(chars[word].value):
                    word += 1
                result.append(self.cut_bytes(begin, chars[word - 1].end))
                first = word
            if maxsplit > 0:
                maxsplit -= 1
        if reverse:
            var n = len(result)
            for i in range(n // 2):
                var t = result[i].copy()
                result[i] = result[n - 1 - i].copy()
                result[n - 1 - i] = t^
        return result^

    def strip(
        self, ju: JU, left: Bool, right: Bool, chars_to_strip: Optional[String] = None
    ) raises JinjaError -> JString:
        var text = self.str()
        var chars = characters(ju, text)
        var matching = List[Int]()
        if case .Some(strip_set) = chars_to_strip:
            for c in characters(ju, strip_set):
                matching.append(c.value)
        var selected = Bool(chars_to_strip)
        var begin = 0
        var end = len(chars)
        if left:
            while begin < end and _matches(ju, chars[begin].value, selected, matching):
                begin += 1
        if right:
            while end > begin and _matches(ju, chars[end - 1].value, selected, matching):
                end -= 1
        var first_byte = text.byte_length() if begin == len(chars) else chars[begin].begin
        var last_byte = text.byte_length() if end == len(chars) else chars[end].begin
        return self.cut_bytes(first_byte, last_byte)

    def is_uppercase(self, ju: JU) raises JinjaError -> Bool:
        var found = False
        for ch in characters(ju, self.str()):
            if not ju.is_cased(ch.value):
                continue
            if ju.upper_ranges.lookup(ch.value) == 0:
                return False
            found = True
        return found

    def is_lowercase(self, ju: JU) raises JinjaError -> Bool:
        var found = False
        for ch in characters(ju, self.str()):
            if not ju.is_cased(ch.value):
                continue
            if ju.lower_ranges.lookup(ch.value) == 0:
                return False
            found = True
        return found

    def map_case(self, ju: JU, mode: Int) raises JinjaError -> JString:
        var ends = List[Int]()
        var offset = 0
        for p in self.parts:
            offset += p.val.byte_length()
            ends.append(offset)
        var mapped = map_case_text(ju, self.str(), mode, ends)
        var result = JString()
        var off = 0
        for i in range(len(self.parts)):
            var part = self.parts[i].copy()
            var text = sub_string(mapped, off, ends[i])
            if text != part.val:
                part.source_offset = -1
            part.val = text^
            result.parts.append(part^)
            off = ends[i]
        return result^

    def uppercase(self, ju: JU) raises JinjaError -> JString:
        return self.map_case(ju, CASE_UPPER)

    def lowercase(self, ju: JU) raises JinjaError -> JString:
        return self.map_case(ju, CASE_LOWER)

    def capitalize(self, ju: JU) raises JinjaError -> JString:
        return self.map_case(ju, CASE_CAPITALIZE)

    def titlecase(self, ju: JU) raises JinjaError -> JString:
        return self.map_case(ju, CASE_TITLE)


def _matches(ju: JU, cp: Int, selected: Bool, matching: List[Int]) -> Bool:
    if selected:
        for m in matching:
            if m == cp:
                return True
        return False
    return ju.whitespace(cp)


def _append_case(ju: JU, mut out: List[UInt8], cp: Int, mode: Int) raises JinjaError:
    if case .Some(i) = ju.special.get(cp):
        var text = ju.special_lo[i] if mode == CASE_LOWER else (ju.special_up[i] if mode == CASE_UPPER else ju.special_ti[i])
        for b in text.as_bytes():
            out.append(b)
        return
    var mapped = cp
    if case .Some(i) = ju.simple.get(cp):
        mapped = ju.simple_lo[i] if mode == CASE_LOWER else (ju.simple_up[i] if mode == CASE_UPPER else ju.simple_ti[i])
    push_utf8(out, mapped)


def map_case_text(ju: JU, text: String, mode: Int, mut boundaries: List[Int]) raises JinjaError -> String:
    var chars = characters(ju, text)
    var out = List[UInt8]()
    var previous_cased = False
    var boundary = 0
    while boundary < len(boundaries) and boundaries[boundary] == 0:
        boundary += 1
    for i in range(len(chars)):
        var cp = chars[i].value
        var current = mode
        if mode == CASE_CAPITALIZE:
            current = CASE_TITLE if i == 0 else CASE_LOWER
        if mode == CASE_TITLE:
            current = CASE_LOWER if previous_cased else CASE_TITLE
        if current == CASE_LOWER and cp == 0x3A3:
            var before = False
            var after = False
            var j = i
            while j > 0:
                j -= 1
                var candidate = chars[j].value
                if ju.is_ignorable(candidate):
                    continue
                before = ju.is_cased(candidate)
                break
            j = i + 1
            while j < len(chars):
                if ju.is_ignorable(chars[j].value):
                    j += 1
                    continue
                after = ju.is_cased(chars[j].value)
                break
            push_utf8(out, 0x3C2 if (before and not after) else 0x3C3)
        else:
            _append_case(ju, out, cp, current)
        previous_cased = ju.is_cased(cp)
        while boundary < len(boundaries) and boundaries[boundary] <= chars[i].end:
            boundaries[boundary] = len(out)
            boundary += 1
    return String(unsafe_from_utf8=out)
