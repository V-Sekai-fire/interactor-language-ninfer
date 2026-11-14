"""Render Jinja templates. Each input line is a JSON case: template, context, regions, controls,
timestamp. Output per case: `case N`, then `text HEX`, `lit B E`, `reg TAG B E OFF`, or `err HEX`."""

from std.pathlib import Path
from std.sys import argv

from ninfer.artifact.json import Json, parse_json
from ninfer.text.jinja.jstring import string_bytes
from ninfer.text.jinja.template import JinjaTemplate, InputRegion


def hexs(data: String) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for b in data.as_bytes():
        out += digits[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += digits[byte=Int(b & 15) : Int(b & 15) + 1]
    return out


def main() raises:
    var text = Path(String(argv()[1])).read_text()
    var n = 0
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        print("case", n)
        n += 1
        try:
            var doc = Json()
            var root = parse_json(string_bytes(String(line)), "case", doc, False)
            var regions = List[InputRegion]()
            var rn = doc.find(root, "regions")
            for i in range(doc.size(rn)):
                var r = doc.at(rn, i)
                regions.append(InputRegion(doc.string(doc.get(r, "pointer")), Int(doc.nodes[doc.get(r, "tag")].u)))
            var controls = List[String]()
            var cn = doc.find(root, "controls")
            for i in range(doc.size(cn)):
                controls.append(doc.string(doc.at(cn, i)))
            var tmpl = JinjaTemplate(doc.string(doc.get(root, "template")), "t")
            var out = tmpl.render(doc, doc.get(root, "context"), regions, controls, Int(doc.nodes[doc.get(root, "timestamp")].u))
            print("text", hexs(out.text))
            for i in range(len(out.literal_begin)):
                print("lit", out.literal_begin[i], out.literal_end[i])
            for r in out.regions:
                print("reg", r.tag, r.begin, r.end, r.source_offset)
        except e:
            print("err", hexs(String(e)))
