"""Dump an artifact's directory and object bytes in a line format the Python test compares.

    mojo run -I mojo mojo/tests/dump_artifact.mojo FILE.ninfer [OBJECT_ID...]
"""

from std.pathlib import Path
from std.sys import argv

from ninfer.artifact.reader import Reader


def hex(data: List[UInt8]) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for b in data:
        out += digits[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += digits[byte=Int(b & 15) : Int(b & 15) + 1]
    return out


def main() raises:
    var args = argv()
    var reader = Reader(Path(args[1]))
    ref d = reader.directory
    print("id", hex(reader.id))
    print("entry_payload_start", reader.entry_payload_start)
    print("file_bytes", reader.file_bytes)
    print("payload_bytes", d.payload_bytes)
    for i in range(len(d.files)):
        var p = d.files[i].path.value() if d.files[i].path else String("-")
        print("file", i, p, d.files[i].payload_bytes, d.files[i].logical_begin)
    for i in range(len(d.objects)):
        ref o = d.objects[i]
        if o.is_tensor:
            var shape = String()
            for s in o.tensor.shape:
                shape += String(s) + ","
            print("tensor", o.id, shape, o.tensor.format, o.tensor.layout, o.offset, o.bytes, o.tensor.divisors)
        else:
            print("resource", o.id, o.encoding, o.offset, o.bytes)
    for name in d.component_names:
        ref c = d.components[name]
        var roles = String()
        for r in range(len(c.resource_roles)):
            roles += c.resource_roles[r] + "=" + d.objects[c.resources[r]].id + ","
        print("component", name, c.target.value() if c.target else String("-"), roles)
    for name in d.binding_names:
        ref b = d.bindings[name]
        var parts = String()
        for p in b.parts:
            parts += d.objects[p.object].id + "[" + String(p.begin) + ":" + String(p.end) + "],"
        print("binding", name, b.whole_object, b.elements, parts)
    for i in range(2, len(args)):
        var h = reader.find(String(args[i]))
        print("bytes", args[i], hex(reader.read_object(h)))
