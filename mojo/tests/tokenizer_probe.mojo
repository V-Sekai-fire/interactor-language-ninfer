"""Tokenizer probe: `tokenizer_probe TOKENIZER_JSON CONFIG_JSON GENERATION_JSON CASES`.

Each CASES line is `MODE HEXTEXT [BOUNDARIES [SPANS]]`: `encode` prints ids, `boundaries` prints
ids and the boundary results, `decode` takes space separated ids in place of HEXTEXT.
"""

from std.pathlib import Path
from std.sys import argv

from ninfer.text.tokenizer import (
    ByteSpan, DecodeOptions, EncodeOptions, Tokenizer,
)


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


def ints(s: StringSpan) raises -> List[Int]:
    var out = List[Int]()
    if String(s) == "-":
        return out^
    for p in s.split(","):
        out.append(atol(String(p)))
    return out^


def main() raises:
    var args = argv()
    var tok: Tokenizer
    try:
        tok = Tokenizer(
            Path(String(args[1])).read_bytes(),
            Path(String(args[2])).read_bytes(),
            Path(String(args[3])).read_bytes(),
        )
    except e:
        print("load_err", String(e))
        return
    print("vocab", tok.vocab_size(), tok.sentencepiece)
    var stops = String()
    for s in tok.default_stop_token_ids:
        stops += String(s) + ","
    print("stops", stops)
    var text = Path(String(args[4])).read_text()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var f = line.split(" ")
        var mode = String(f[0])
        try:
            if mode == "decode":
                var ids = ints(f[1])
                var opts = DecodeOptions(skip_special_tokens=String(f[2]) == "1")
                opts.stop_token_ids = ints(f[3])
                print("ok", hexs(tok.decode(ids, opts)))
            else:
                var data = unhex(f[1])
                var opts = EncodeOptions(parse_added_tokens=String(f[2]) == "1", max_tokens=atol(String(f[3])))
                var bounds = ints(f[4]) if len(f) > 4 else List[Int]()
                var spans = List[ByteSpan]()
                if len(f) > 5 and String(f[5]) != "-":
                    var pairs = ints(f[5])
                    for i in range(0, len(pairs), 2):
                        spans.append(ByteSpan(pairs[i], pairs[i + 1]))
                var r = tok.encode_with_boundaries(data, bounds, opts, spans)
                var out = String()
                for id in r.input_ids:
                    out += String(id) + ","
                var bs = String()
                for b in r.boundaries:
                    bs += (String(b.exact) if b.has_exact else String("x")) + "/" + String(b.stable) + ","
                print("ok", out, bs)
        except e:
            print("err", String(e))
