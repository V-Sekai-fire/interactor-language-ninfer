"""Splits each hex-encoded line from the file named in argv[1] with the combinator pre-tokenizer
and prints the piece end offsets, space separated."""

from std.sys import argv

from ninfer.text.qwen_split import build_qwen_split, qwen_word_end
from ninfer.text.unicode import Unicode


def _hex_value(c: UInt8) -> Int:
    return Int(c) - 0x30 if c <= 0x39 else Int(c) - 0x61 + 10


def main() raises:
    var lines = open(String(argv()[1]), "r").read().split("\n")
    var split = build_qwen_split()
    var unicode = Unicode()
    for line in lines:
        if line.byte_length() == 0:
            continue
        var text = List[UInt8]()
        var b = line.as_bytes()
        var i = 0
        while i + 1 < len(b):
            text.append(UInt8(_hex_value(b[i]) * 16 + _hex_value(b[i + 1])))
            i += 2
        var out = String()
        var begin = 0
        while begin < len(text):
            begin = qwen_word_end(split, text, begin, unicode, "probe input")
            out += String(begin) + " "
        print(out)
