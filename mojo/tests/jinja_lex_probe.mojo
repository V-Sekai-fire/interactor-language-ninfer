"""Token streams of Jinja templates. Each line of the case file is a hex-encoded template (`-` is
empty); every case prints `case N`, one `type pos hexvalue` line per token, then `source HEX`
or `err HEX`."""

from std.pathlib import Path
from std.sys import argv

from ninfer.text.jinja.lexer import tokenize, type_to_string


def hexs(data: String) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for b in data.as_bytes():
        out += digits[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += digits[byte=Int(b & 15) : Int(b & 15) + 1]
    return out


def unhex(s: StringSpan) -> List[UInt8]:
    var out = List[UInt8]()
    if s == "-":
        return out^
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


def main() raises:
    var text = Path(String(argv()[1])).read_text()
    var n = 0
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        print("case", n)
        n += 1
        try:
            var r = tokenize(unhex(line))
            for t in r.tokens:
                print(type_to_string(t.t), t.pos, hexs(t.value))
            print("source", hexs(String(unsafe_from_utf8=r.source)))
        except e:
            print("err", hexs(String(e)))
