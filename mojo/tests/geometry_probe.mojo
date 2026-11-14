"""Print `weight_geometry` for each case line `FORMAT LAYOUT DIVISORS D0,D1,...` of a file."""

from std.pathlib import Path
from std.sys import argv

from ninfer.artifact.formats import format_name, layout_name, parse_format, parse_layout
from ninfer.core.weight import weight_geometry


def main() raises:
    var text = Path(argv()[1]).read_text()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var f = line.split(" ")
        var shape = List[UInt64]()
        if f[3].byte_length() > 0:
            for d in f[3].split(","):
                shape.append(UInt64(atol(String(d))))
        try:
            var g = weight_geometry(
                parse_format(String(f[0])),
                parse_layout(String(f[1])),
                shape,
                UInt64(atol(String(f[2]))),
            )
            print("ok", line, g.elements, g.bytes, g.padded_columns, g.group_size,
                  g.code_bytes_per_row, g.high_bytes_per_row, g.scale_bytes_per_row,
                  g.row_stride_bytes, g.code_bytes, g.high_offset, g.high_bytes,
                  g.scale_offset, g.scale_bytes, g.divisor_offset, g.divisor_count)
        except e:
            print("error", line, "::", String(e))
