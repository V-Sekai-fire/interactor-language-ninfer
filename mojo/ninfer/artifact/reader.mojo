"""Entry and continuation file reader; mirrors `src/artifact/reader.{h,cpp}`.

The reader owns the cold directory and opens continuation files lazily. It reads exact byte
ranges of the logical payload; it does not interpret tensor encodings (that is the
materializer's job, a later port).
"""

from std.collections import List, Optional
from std.pathlib import Path

from ninfer.artifact.framing import (
    HEADER_BYTES,
    PAYLOAD_ALIGNMENT,
    entry_magic,
    part_magic,
    read_u64_le,
)
from ninfer.artifact.formats import RAW_BYTES_ENCODING
from ninfer.artifact.layouts import describe_tensor
from ninfer.core.weight import WeightGeometry
from ninfer.artifact.schema import (
    Directory,
    checked_add,
    checked_mul,
    align_up,
    parse_directory,
)


struct ReadSegment(Copyable, Movable):
    var file_index: Int
    var file_offset: UInt64
    var destination_offset: UInt64
    var bytes: UInt64

    def __init__(
        out self,
        file_index: Int,
        file_offset: UInt64,
        destination_offset: UInt64,
        bytes: UInt64,
    ):
        self.file_index = file_index
        self.file_offset = file_offset
        self.destination_offset = destination_offset
        self.bytes = bytes


def _sibling(entry: Path, name: String) -> Path:
    """`name` in the directory of `entry` (continuation names carry no separators)."""
    var text = String(entry)
    var slash = text.rfind("/")
    if slash < 0:
        return Path(name)
    return Path(text[byte=0 : slash + 1] + name)


def _file_size(path: Path) raises -> UInt64:
    return UInt64(path.stat().st_size)


def read_exact(path: Path, offset: UInt64, count: Int) raises -> List[UInt8]:
    """Read exactly `count` bytes at `offset` or raise."""
    if count == 0:
        return List[UInt8]()
    var data: List[UInt8]
    with open(path, "r") as f:
        _ = f.seek(Int(offset))
        data = f.read_bytes(count)
    if len(data) != count:
        raise Error(String(path) + ": short read")
    return data^


struct Reader(Movable):
    var entry: Path
    var directory: Directory
    var id: List[UInt8]
    var entry_payload_start: UInt64
    var file_bytes: UInt64
    var sizes: List[UInt64]  # per-file byte length, 0 until the file is validated
    var validated: List[Bool]
    var geometries: List[Optional[WeightGeometry]]

    def __init__(out self, path: Path) raises:
        self.entry = path
        var size = _file_size(path)
        var header = read_exact(path, 0, HEADER_BYTES)
        var magic = entry_magic()
        var ok = True
        for i in range(8):
            if header[i] != magic[i]:
                ok = False
        if not ok:
            var v2 = True
            for i in range(7):
                if header[i] != magic[i]:
                    v2 = False
            if v2 and header[7] == 2:
                raise Error(
                    String(path)
                    + ": NInfer v2 artifact is not supported. Upgrade to v3 with: "
                    "python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer"
                )
            raise Error(String(path) + ": expected NInfer v3 entry magic")
        var json_bytes = read_u64_le(header, 8)
        if json_bytes == 0 or json_bytes > size - UInt64(HEADER_BYTES):
            raise Error(String(path) + ": invalid directory length")
        self.id = List[UInt8]()
        for i in range(16):
            self.id.append(header[16 + i])
        var text = read_exact(path, UInt64(HEADER_BYTES), Int(json_bytes))
        self.directory = parse_directory(text^, path.name())
        self.entry_payload_start = align_up(
            checked_add(UInt64(HEADER_BYTES), json_bytes, "metadata end"),
            PAYLOAD_ALIGNMENT,
            "entry payload start",
        )
        var expected = checked_add(
            self.entry_payload_start,
            self.directory.files[0].payload_bytes,
            "entry length",
        )
        if expected != size:
            raise Error(String(path) + ": entry length differs from directory")
        var total = checked_add(
            self.entry_payload_start, self.directory.payload_bytes, "file bytes"
        )
        self.file_bytes = checked_add(
            total,
            checked_mul(
                UInt64(len(self.directory.files) - 1),
                PAYLOAD_ALIGNMENT,
                "continuation headers",
            ),
            "file bytes",
        )
        self.sizes = List[UInt64]()
        self.validated = List[Bool]()
        for _ in range(len(self.directory.files)):
            self.sizes.append(0)
            self.validated.append(False)
        self.geometries = List[Optional[WeightGeometry]]()
        for _ in range(len(self.directory.objects)):
            self.geometries.append(None)
        self.sizes[0] = size
        self.validated[0] = True

    def find(self, id: String) raises -> Int:
        var found = self.directory.object_index.get(id)
        if not found:
            raise Error("missing object " + id)
        return found.value()

    def file_path(mut self, index: Int) raises -> Path:
        """Validate (once) and return the path of file `index`."""
        if index < 0 or index >= len(self.validated):
            raise Error("invalid continuation index")
        if index == 0:
            return self.entry
        ref record = self.directory.files[index]
        var name = record.path.value()
        var path = _sibling(self.entry, name)
        if not self.validated[index]:
            var header = read_exact(path, 0, HEADER_BYTES)
            var magic = part_magic()
            var ok = read_u64_le(header, 8) == UInt64(index)
            for i in range(8):
                if header[i] != magic[i]:
                    ok = False
            for i in range(16):
                if header[16 + i] != self.id[i]:
                    ok = False
            if not ok:
                raise Error(name + ": continuation header does not belong to this entry")
            var size = _file_size(path)
            if size != checked_add(PAYLOAD_ALIGNMENT, record.payload_bytes, "part length"):
                raise Error(name + ": continuation length differs from directory")
            self.sizes[index] = size
            self.validated[index] = True
        return path

    def geometry(mut self, handle: Int) raises -> WeightGeometry:
        """Encoded-parent geometry of a tensor, validated on first request and then cached."""
        self.directory.check_tensor(handle)
        if not self.geometries[handle]:
            self.geometries[handle] = describe_tensor(self.directory.objects[handle])
        return self.geometries[handle].value().copy()

    def validate_object(mut self, handle: Int) raises:
        self.directory.check_handle(handle)
        if self.directory.objects[handle].is_tensor:
            _ = self.geometry(handle)
        elif self.directory.objects[handle].encoding != RAW_BYTES_ENCODING:
            raise Error(self.directory.objects[handle].id + ": unsupported resource encoding")

    def segments(self, offset: UInt64, bytes: UInt64) raises -> List[ReadSegment]:
        ref d = self.directory
        if offset > d.payload_bytes or bytes > d.payload_bytes - offset:
            raise Error("read range exceeds logical payload")
        var result = List[ReadSegment]()
        if bytes == 0:
            return result^
        # Last file whose logical_begin <= offset (upper_bound - 1).
        var index = 0
        for i in range(len(d.files)):
            if d.files[i].logical_begin <= offset:
                index = i
            else:
                break
        var copied = UInt64(0)
        while copied < bytes:
            ref file = d.files[index]
            var local = offset + copied - file.logical_begin
            var count = min(bytes - copied, file.payload_bytes - local)
            var start = self.entry_payload_start if index == 0 else PAYLOAD_ALIGNMENT
            result.append(
                ReadSegment(index, checked_add(start, local, "file offset"), copied, count)
            )
            copied += count
            index += 1
        return result^

    def read_range(mut self, offset: UInt64, bytes: UInt64) raises -> List[UInt8]:
        var segs = self.segments(offset, bytes)
        var result = List[UInt8](capacity=Int(bytes))
        for seg in segs:
            var path = self.file_path(seg.file_index)
            var chunk = read_exact(path, seg.file_offset, Int(seg.bytes))
            result.extend(chunk^)
        return result^

    def read_object(mut self, handle: Int) raises -> List[UInt8]:
        self.validate_object(handle)
        self.directory.check_handle(handle)
        ref obj = self.directory.objects[handle]
        var offset = obj.offset
        var bytes = obj.bytes
        return self.read_range(offset, bytes)
