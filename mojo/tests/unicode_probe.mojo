"""Unicode probe: `classes`, or `iter FILE` / `nfc FILE` over lines of hex-encoded bytes."""

from std.pathlib import Path
from std.sys import argv

from ninfer.text.unicode import Unicode, append_utf8, utf8_codepoint_at, utf8_codepoints


def unhex(s: StringSpan) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    var i = 0
    while i + 1 < len(b):
        var hi = Int(b[i])
        var lo = Int(b[i + 1])
        hi = hi - 0x30 if hi <= 0x39 else hi - 0x61 + 10
        lo = lo - 0x30 if lo <= 0x39 else lo - 0x61 + 10
        out.append(UInt8(hi * 16 + lo))
        i += 2
    return out^


def hexs(data: List[UInt8]) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for b in data:
        out += digits[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += digits[byte=Int(b & 15) : Int(b & 15) + 1]
    return out


def main() raises:
    var args = argv()
    var u = Unicode()
    var mode = String(args[1])
    if mode == "classes":
        # One digit per code point: letter 1, mark 2, number 4, whitespace 8 (bitmask in 0..15 as hex).
        var out = String()
        for cp in range(0, 0x110000):
            var m = 0
            if u.is_letter(cp):
                m |= 1
            if u.is_mark(cp):
                m |= 2
            if u.is_number(cp):
                m |= 4
            if u.is_whitespace(cp):
                m |= 8
            out += "0123456789abcdef"[byte=m : m + 1]
        print(out)
        return
    var text = Path(String(args[2])).read_text()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var data = unhex(line)
        if mode == "iter":
            var s = String()
            try:
                for sp in utf8_codepoints(data, "probe"):
                    s += String(sp.value) + ":" + String(sp.length) + ","
                print("ok", s)
            except e:
                print("err", String(e))
        else:
            try:
                print("ok", hexs(u.normalize_nfc(data)))
            except e:
                print("err", String(e))
