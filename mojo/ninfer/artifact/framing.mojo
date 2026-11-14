"""Container framing constants; mirrors `src/artifact/framing.h`."""

comptime HEADER_BYTES = 32
comptime PAYLOAD_ALIGNMENT = UInt64(4096)
comptime ENTRY_MAGIC_LEN = 8


def entry_magic() -> List[UInt8]:
    """`NINFER\\0\\3`."""
    return [UInt8(0x4E), 0x49, 0x4E, 0x46, 0x45, 0x52, 0x00, 0x03]


def part_magic() -> List[UInt8]:
    """`NINPRT\\0\\3`."""
    return [UInt8(0x4E), 0x49, 0x4E, 0x50, 0x52, 0x54, 0x00, 0x03]


def read_u64_le(bytes: List[UInt8], offset: Int) -> UInt64:
    var out = UInt64(0)
    for i in range(8):
        out |= UInt64(bytes[offset + i]) << UInt64(8 * i)
    return out


def read_u32_le(bytes: List[UInt8], offset: Int) -> UInt32:
    var out = UInt32(0)
    for i in range(4):
        out |= UInt32(bytes[offset + i]) << UInt32(8 * i)
    return out
