"""Run NimbleParsec-style specs. Each input line is JSON: {"defs": {name: spec}, "main": spec, "inputs": [...]}.
Output per case: `case N`, then one line per input:
  ok TERMS rest=HEX line=L,O off=N   or   err REASONHEX rest=HEX line=L,O off=N"""

from std.pathlib import Path
from std.sys import argv

from ninfer.artifact.json import Json, parse_json, NULL, UINT, NEG, STRING
from ninfer.parsec.parsec import Parsec
from ninfer.text.jinja.jstring import string_bytes


def hexs(data: String) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for b in data.as_bytes():
        out += digits[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += digits[byte=Int(b & 15) : Int(b & 15) + 1]
    return out


def _int(doc: Json, n: Int) -> Int:
    var v = Int(doc.nodes[n].u)
    return -v if doc.kind(n) == NEG else v


def _ranges(doc: Json, n: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(doc.size(n)):
        var pair = doc.at(n, i)
        out.append(_int(doc, doc.at(pair, 0)))
        out.append(_int(doc, doc.at(pair, 1)))
    return out^


def _opt_int(doc: Json, n: Int, key: String, default: Int) -> Int:
    var k = doc.find(n, key)
    if k < 0 or doc.kind(k) == NULL:
        return default
    return _int(doc, k)


def build(mut p: Parsec, doc: Json, n: Int) raises -> Int:
    var op = doc.string(doc.get(n, "op"))
    if op == "empty":
        return p.empty()
    if op == "string":
        return p.string(doc.string(doc.get(n, "v")))
    if op == "ascii_char" or op == "utf8_char":
        var inc = _ranges(doc, doc.get(n, "incl"))
        var exc = _ranges(doc, doc.get(n, "excl"))
        return p.ascii_char(inc, exc) if op == "ascii_char" else p.utf8_char(inc, exc)
    if op == "bytes":
        return p.bytes(_int(doc, doc.get(n, "n")))
    if op == "eos":
        return p.eos()
    if op == "integer":
        if doc.contains(n, "n"):
            return p.integer(_int(doc, doc.get(n, "n")))
        return p.integer(_int(doc, doc.get(n, "min")), _opt_int(doc, n, "max", -1))
    if op == "ascii_string" or op == "utf8_string":
        var inc = _ranges(doc, doc.get(n, "incl"))
        var mn = _opt_int(doc, n, "min", 0)
        var mx = -2 if doc.contains(n, "n") else _opt_int(doc, n, "max", -1)
        if doc.contains(n, "n"):
            mn = _int(doc, doc.get(n, "n"))
        return p.ascii_string(inc, mn, mx) if op == "ascii_string" else p.utf8_string(inc, mn, mx)
    if op == "concat":
        var acc = p.empty()
        var cs = doc.get(n, "cs")
        for i in range(doc.size(cs)):
            acc = p.concat(acc, build(p, doc, doc.at(cs, i)))
        return acc
    if op == "choice":
        var alts = List[Int]()
        var cs = doc.get(n, "cs")
        for i in range(doc.size(cs)):
            alts.append(build(p, doc, doc.at(cs, i)))
        return p.choice(alts)
    var c = build(p, doc, doc.get(n, "c")) if doc.contains(n, "c") else -1
    if op == "optional":
        return p.optional(c)
    if op == "times":
        if doc.contains(n, "n"):
            return p.times(c, _int(doc, doc.get(n, "n")), _int(doc, doc.get(n, "n")))
        return p.times(c, _opt_int(doc, n, "min", 0), _opt_int(doc, n, "max", -1))
    if op == "duplicate":
        return p.duplicate(c, _int(doc, doc.get(n, "n")))
    if op == "repeat":
        return p.repeat(c)
    if op == "eventually":
        return p.eventually(c)
    if op == "lookahead":
        return p.lookahead(c)
    if op == "lookahead_not":
        return p.lookahead_not(c)
    if op == "ignore":
        return p.ignore(c)
    if op == "replace":
        var v = doc.get(n, "v")
        var t = p.t_bin(doc.string(v)) if doc.kind(v) == STRING else p.t_int(_int(doc, v))
        return p.replace(c, t)
    if op == "tag":
        return p.tag(c, doc.string(doc.get(n, "tag")))
    if op == "unwrap_and_tag":
        return p.unwrap_and_tag(c, doc.string(doc.get(n, "tag")))
    if op == "wrap":
        return p.wrap(c)
    if op == "label":
        return p.label(c, doc.string(doc.get(n, "label")))
    if op == "line":
        return p.line(c)
    if op == "byte_offset":
        return p.byte_offset(c)
    if op == "reduce_sum":
        return p.reduce_sum(c)
    if op == "map_inc":
        return p.map_inc(c)
    if op == "post_count":
        return p.post_count(c)
    if op == "parsec":
        return p.parsec(doc.string(doc.get(n, "name")))
    raise Error("unknown op " + op)


def main() raises:
    var text = Path(String(argv()[1])).read_text()
    var n = 0
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        print("case", n)
        n += 1
        var doc = Json()
        var root = parse_json(string_bytes(String(line)), "case", doc, False)
        var p = Parsec()
        var defs = doc.get(root, "defs")
        for i in range(doc.size(defs)):
            p.define(doc.key(defs, i), build(p, doc, doc.at(defs, i)))
        p.define("main", build(p, doc, doc.get(root, "main")))
        var inputs = doc.get(root, "inputs")
        for i in range(doc.size(inputs)):
            var r = p.parse("main", doc.string(doc.at(inputs, i)))
            var tail = " rest=" + hexs(r.rest) + " line=" + String(r.line) + "," + String(r.line_off) + " off=" + String(r.offset)
            if r.ok:
                var out = String("ok [")
                for k in range(len(r.acc)):
                    if k > 0:
                        out += ","
                    out += p.show(r.acc[k])
                print(out + "]" + tail)
            else:
                print("err " + hexs(r.reason) + tail)
