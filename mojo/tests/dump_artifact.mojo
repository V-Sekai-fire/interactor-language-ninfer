"""Dump an artifact's directory and object bytes in a line format the Python test compares.

    mojo run -I mojo mojo/tests/dump_artifact.mojo FILE.ninfer [OBJECT_ID...]
"""

from std.pathlib import Path
from std.sys import argv

from ninfer.artifact.binder import Binder, RESIDENCY_HOST
from ninfer.artifact.formats import format_name, layout_name
from ninfer.artifact.reader import Reader
from ninfer.core.weight import BF16, CONTIGUOUS, FP32, INT32


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
    # Geometry is interpreted on request, as in C++: a bad tensor is reported, not fatal.
    var tensor_ids = List[Int]()
    for i in range(len(reader.directory.objects)):
        if reader.directory.objects[i].is_tensor:
            tensor_ids.append(i)
    for i in tensor_ids:
        var id = reader.directory.objects[i].id
        try:
            var g = reader.geometry(i)
            print("geometry", id, format_name(g.format), layout_name(g.layout), g.elements, g.bytes,
                  g.padded_columns, g.group_size, g.code_bytes_per_row, g.high_bytes_per_row,
                  g.scale_bytes_per_row, g.row_stride_bytes, g.code_bytes, g.high_offset,
                  g.high_bytes, g.scale_offset, g.scale_bytes, g.divisor_offset, g.divisor_count)
        except e:
            print("geometry_error", id, String(e))
    var binder = Binder()
    var names = reader.directory.binding_names.copy()
    for name in names:
        var b = reader.directory.bindings[name].copy()
        var shape = List[UInt64]()
        if b.whole_object:
            shape = reader.directory.objects[b.parts[0].object].tensor.shape.copy()
        else:
            shape.append(b.elements)
        try:
            var ref_ = binder.parameter(reader, name, shape, RESIDENCY_HOST)
            print("parameter", name, ref_.residency)
            var f = reader.geometry(b.parts[0].object)
            if f.layout == CONTIGUOUS and (f.format == BF16 or f.format == FP32 or f.format == INT32):
                var v = binder.values(reader, b)
                print("values", name, format_name(v.format), v.elements, hex(v.data))
        except e:
            print("parameter_error", name, String(e))
    var components = reader.directory.component_names.copy()
    for ci in range(len(components)):
        var comp = components[ci]
        var roles = reader.directory.components[comp].resource_roles.copy()
        for r in range(len(roles)):
            var h = binder.resource(reader, comp, roles[r])
            print("resource_bytes", comp, roles[r], reader.directory.objects[h].id)
    try:
        var one = List[UInt64]()
        one.append(1)
        _ = binder.parameter(reader, "nope", one)
    except e:
        print("missing_parameter", String(e))
    for i in range(2, len(args)):
        var h = reader.find(String(args[i]))
        print("bytes", args[i], hex(reader.read_object(h)))
