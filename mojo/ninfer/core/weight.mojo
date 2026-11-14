"""Weight formats, layouts and encoded-parent geometry; mirrors `src/core/weight{,_view}.{h,cpp}`.

`QType` and `QuantLayout` keep the C++ enum values so a number printed by either engine means
the same thing. Geometry functions raise the same messages as the C++ `std::invalid_argument`
and `std::overflow_error` they replace.
"""

from std.collections import List

# --- QType ---------------------------------------------------------------------------------
comptime Q4_G64_FP16 = 0
comptime Q5_G64_FP16 = 1
comptime Q6_G64_FP16 = 2
comptime Q8_G32_FP16 = 3
comptime BF16 = 4
comptime FP32 = 5
comptime INT32 = 6
comptime NVFP4 = 7
comptime FP8_E4M3FN_ROW_BF16 = 8
comptime T2_G128_FP16 = 9
comptime GGUF_Q8_0 = 10
comptime GGUF_Q2_K = 11
comptime GGUF_Q3_K = 12
comptime GGUF_Q4_K = 13
comptime GGUF_Q5_K = 14
comptime GGUF_Q6_K = 15
comptime GGUF_IQ2_XXS = 16
comptime GGUF_IQ2_XS = 17
comptime GGUF_IQ2_S = 18
comptime GGUF_IQ3_XXS = 19
comptime GGUF_IQ3_S = 20
comptime GGUF_IQ1_S = 21
comptime GGUF_IQ1_M = 22
comptime GGUF_IQ4_NL = 23
comptime GGUF_IQ4_XS = 24
comptime GGUF_Q4_0 = 25
comptime GGUF_Q5_0 = 26
comptime GGUF_Q2_0 = 27
comptime Q2_G64_FP16 = 28
comptime FP8_E4M3FN_ROW_FP16 = 29

# --- QuantLayout ---------------------------------------------------------------------------
comptime ROW_SPLIT = 0
comptime CONTIGUOUS = 1
comptime BLOCK_SCALE_K16_M128X4 = 2
comptime ROW_SCALE = 3
comptime ROW_SPLIT_PANEL = 4  # device-only; an artifact may not declare it
comptime GGUF_BLOCKS = 5
comptime ROW_INTERLEAVED = 6

comptime ROW_SPLIT_PANEL_ROWS = 4


def is_gguf(format: Int) -> Bool:
    return format >= GGUF_Q8_0 and format <= GGUF_Q2_0


def is_row_split(layout: Int) -> Bool:
    return layout == ROW_SPLIT or layout == ROW_SPLIT_PANEL


def gguf_block_elements(format: Int) -> UInt64:
    """ggml-common.h block element counts; 0 for every non-GGUF format."""
    if format == GGUF_Q8_0 or format == GGUF_IQ4_NL or format == GGUF_Q4_0 or format == GGUF_Q5_0:
        return 32
    if format == GGUF_Q2_0:
        return 64
    if format >= GGUF_Q2_K and format <= GGUF_IQ4_XS:
        return 256
    return 0


def gguf_block_bytes(format: Int) -> UInt64:
    """ggml-common.h block byte counts; 0 for every non-GGUF format."""
    if format == GGUF_Q8_0:
        return 34
    if format == GGUF_Q2_K:
        return 84
    if format == GGUF_Q3_K:
        return 110
    if format == GGUF_Q4_K:
        return 144
    if format == GGUF_Q5_K:
        return 176
    if format == GGUF_Q6_K:
        return 210
    if format == GGUF_IQ2_XXS:
        return 66
    if format == GGUF_IQ2_XS:
        return 74
    if format == GGUF_IQ2_S:
        return 82
    if format == GGUF_IQ3_XXS:
        return 98
    if format == GGUF_IQ3_S:
        return 110
    if format == GGUF_IQ1_S:
        return 50
    if format == GGUF_IQ1_M:
        return 56
    if format == GGUF_IQ4_NL:
        return 18
    if format == GGUF_IQ4_XS:
        return 136
    if format == GGUF_Q4_0:
        return 18
    if format == GGUF_Q5_0:
        return 22
    if format == GGUF_Q2_0:
        return 18
    return 0


# --- checked arithmetic (geometry flavour) -------------------------------------------------
def _add(a: UInt64, b: UInt64) raises -> UInt64:
    if b > UInt64.MAX - a:
        raise Error("weight geometry addition overflows u64")
    return a + b


def _mul(a: UInt64, b: UInt64) raises -> UInt64:
    if a != 0 and b > UInt64.MAX // a:
        raise Error("weight geometry multiplication overflows u64")
    return a * b


def _aligned(n: UInt64, alignment: UInt64) raises -> UInt64:
    return _add(n, alignment - 1) // alignment * alignment


def _word_bytes(format: Int) raises -> UInt64:
    if format == BF16:
        return 2
    if format == FP32 or format == INT32:
        return 4
    raise Error("direct weight requires BF16, FP32 or INT32")


def weight_element_count(shape: List[UInt64]) raises -> UInt64:
    if len(shape) > 16:
        raise Error("weight rank exceeds 16")
    var count = UInt64(1)
    for d in shape:
        if d == 0:
            raise Error("weight dimensions must be positive")
        count = _mul(count, d)
    return count


struct WeightGeometry(Copyable, Movable):
    """One complete encoded parent. Regions address its logical elements; padding and scale
    words do not belong to that element domain."""

    var format: Int
    var layout: Int
    var shape: List[UInt64]
    var elements: UInt64
    var bytes: UInt64
    var alignment: UInt64
    var padded_columns: UInt64
    var group_size: UInt64
    var code_bytes_per_row: UInt64
    var high_bytes_per_row: UInt64
    var scale_bytes_per_row: UInt64
    var row_stride_bytes: UInt64  # nonzero when codes and scales share each row
    var code_bytes: UInt64
    var high_offset: UInt64
    var high_bytes: UInt64
    var scale_offset: UInt64
    var scale_bytes: UInt64
    var divisor_offset: UInt64
    var divisor_count: UInt64

    def __init__(out self):
        self.format = BF16
        self.layout = CONTIGUOUS
        self.shape = List[UInt64]()
        self.elements = 0
        self.bytes = 0
        self.alignment = 256
        self.padded_columns = 0
        self.group_size = 0
        self.code_bytes_per_row = 0
        self.high_bytes_per_row = 0
        self.scale_bytes_per_row = 0
        self.row_stride_bytes = 0
        self.code_bytes = 0
        self.high_offset = 0
        self.high_bytes = 0
        self.scale_offset = 0
        self.scale_bytes = 0
        self.divisor_offset = 0
        self.divisor_count = 1


def weight_geometry(
    format: Int, layout: Int, shape: List[UInt64], divisors: UInt64 = 1
) raises -> WeightGeometry:
    if divisors == 0 or (divisors != 1 and format != NVFP4):
        raise Error("only an NVFP4 plane carries more than one divisor")
    var out = WeightGeometry()
    out.format = format
    out.divisor_count = divisors
    out.layout = layout
    out.shape = shape.copy()
    out.elements = weight_element_count(shape)
    if layout == CONTIGUOUS:
        out.code_bytes = _mul(out.elements, _word_bytes(format))
        out.bytes = out.code_bytes
        if len(shape) == 2:
            out.padded_columns = shape[1]
            out.code_bytes_per_row = _mul(shape[1], _word_bytes(format))
        return out^
    if len(shape) != 2:
        raise Error("quantized weight must be a matrix")
    var n = shape[0]
    var k = shape[1]
    out.padded_columns = k
    if is_row_split(layout):
        var code_per_group = UInt64(32)
        var high_per_group = UInt64(0)
        if format == Q2_G64_FP16:
            out.group_size = 64
            code_per_group = 16
        elif format == Q4_G64_FP16:
            out.group_size = 64
        elif format == Q5_G64_FP16:
            out.group_size = 64
            high_per_group = 8
        elif format == Q6_G64_FP16:
            out.group_size = 64
            high_per_group = 16
        elif format == Q8_G32_FP16:
            out.group_size = 32
        elif format == T2_G128_FP16:
            out.group_size = 128
        else:
            raise Error("RowSplit requires a grouped integer format")
        out.padded_columns = _aligned(k, 128)
        var groups = out.padded_columns // out.group_size
        out.code_bytes_per_row = _mul(groups, code_per_group)
        out.high_bytes_per_row = _mul(groups, high_per_group)
        out.scale_bytes_per_row = _mul(groups, 2)
        out.code_bytes = _mul(n, out.code_bytes_per_row)
        out.high_offset = _aligned(out.code_bytes, 256)
        out.high_bytes = _mul(n, out.high_bytes_per_row)
        out.scale_offset = _add(out.high_offset, _aligned(out.high_bytes, 256))
    elif layout == GGUF_BLOCKS:
        var block_elements = gguf_block_elements(format)
        if not is_gguf(format) or k % block_elements != 0:
            raise Error("GgufBlocks requires a ggml block format over whole blocks")
        out.group_size = block_elements
        out.code_bytes_per_row = _mul(k // block_elements, gguf_block_bytes(format))
        out.code_bytes = _mul(n, out.code_bytes_per_row)
        out.scale_offset = out.code_bytes
    elif layout == ROW_SCALE:
        if format != FP8_E4M3FN_ROW_BF16:
            raise Error("RowScale requires row-scaled FP8")
        out.group_size = k
        out.code_bytes_per_row = k
        out.scale_bytes_per_row = 2
        out.code_bytes = out.elements
        out.scale_offset = _aligned(out.code_bytes, 256)
    elif layout == ROW_INTERLEAVED:
        if format != FP8_E4M3FN_ROW_FP16:
            raise Error("RowInterleaved requires FP8 with FP16 row scales")
        out.group_size = k
        out.code_bytes_per_row = k
        out.scale_bytes_per_row = 2
        out.row_stride_bytes = _add(k, 2)
        out.code_bytes = out.elements
        out.scale_offset = k
        out.scale_bytes = _mul(n, 2)
        out.bytes = _mul(n, out.row_stride_bytes)
        return out^
    elif layout == BLOCK_SCALE_K16_M128X4:
        if format != NVFP4 or n % 128 != 0 or k % 64 != 0:
            raise Error("NVFP4 BlockScale requires N%128=0 and K%64=0")
        out.group_size = 16
        out.code_bytes_per_row = k // 2
        out.scale_bytes_per_row = k // 16
        out.code_bytes = out.elements // 2
        out.scale_offset = _aligned(out.code_bytes, 256)
    else:
        raise Error("unknown quantized weight layout")
    out.scale_bytes = _mul(n, out.scale_bytes_per_row)
    out.bytes = _add(out.scale_offset, out.scale_bytes)
    if format == NVFP4:
        if n % divisors != 0:
            raise Error("divisor count must divide the plane's rows")
        if divisors > 1 and (n // divisors) % 128 != 0:
            raise Error("each NVFP4 divisor must cover whole 128-row scale tiles")
        out.divisor_offset = out.bytes
        out.bytes = _add(out.bytes, _mul(divisors, 4))
    return out^


def weight_scale_offset(g: WeightGeometry, row: UInt64, group: UInt64) raises -> UInt64:
    """Byte offset of the scale word of (`row`, `group`) inside the encoded parent."""
    if (
        len(g.shape) != 2
        or g.group_size == 0
        or row >= g.shape[0]
        or group >= g.padded_columns // g.group_size
    ):
        raise Error("weight scale coordinate exceeds its parent")
    if g.layout == BLOCK_SCALE_K16_M128X4:
        var inner = row % 128
        return (
            g.scale_offset
            + (row // 128 * (g.shape[1] // 64) + group // 4) * 512
            + inner % 32 * 16
            + inner // 32 * 4
            + group % 4
        )
    var stride = g.row_stride_bytes if g.row_stride_bytes != 0 else g.scale_bytes_per_row
    return g.scale_offset + row * stride + group * 2
