"""Format and layout spellings; mirrors `src/artifact/formats.{h,cpp}`."""

from std.collections import List

from ninfer.core.weight import *

comptime RAW_BYTES_ENCODING = "raw_bytes_v1"


def _format_names() -> List[String]:
    # Index == QType value.
    return [
        "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16", "bf16", "fp32", "int32",
        "nvfp4", "fp8_e4m3fn_row_bf16", "t2_g128_fp16", "gguf_q8_0", "gguf_q2_k", "gguf_q3_k",
        "gguf_q4_k", "gguf_q5_k", "gguf_q6_k", "gguf_iq2_xxs", "gguf_iq2_xs", "gguf_iq2_s",
        "gguf_iq3_xxs", "gguf_iq3_s", "gguf_iq1_s", "gguf_iq1_m", "gguf_iq4_nl", "gguf_iq4_xs",
        "gguf_q4_0", "gguf_q5_0", "gguf_q2_0", "q2_g64_fp16", "fp8_e4m3fn_row_fp16",
    ]


def parse_format(name: String) raises -> Int:
    var names = _format_names()
    for i in range(len(names)):
        if names[i] == name:
            return i
    raise Error("unknown tensor format: " + name)


def parse_layout(name: String) raises -> Int:
    """Layouts an artifact may declare; `row_split_panel4_v1` is device-only and unknown here."""
    if name == "contiguous_le_v1":
        return CONTIGUOUS
    if name == "row_split_k128_v1":
        return ROW_SPLIT
    if name == "row_scale_v1":
        return ROW_SCALE
    if name == "row_interleaved_v1":
        return ROW_INTERLEAVED
    if name == "block_scale_k16_m128x4_v1":
        return BLOCK_SCALE_K16_M128X4
    if name == "gguf_blocks_v1":
        return GGUF_BLOCKS
    raise Error("unknown tensor layout: " + name)


def format_name(format: Int) -> String:
    var names = _format_names()
    if format >= 0 and format < len(names):
        return names[format]
    return String()


def layout_name(layout: Int) -> String:
    if layout == CONTIGUOUS:
        return "contiguous_le_v1"
    if layout == ROW_SPLIT:
        return "row_split_k128_v1"
    if layout == ROW_SCALE:
        return "row_scale_v1"
    if layout == ROW_INTERLEAVED:
        return "row_interleaved_v1"
    if layout == BLOCK_SCALE_K16_M128X4:
        return "block_scale_k16_m128x4_v1"
    if layout == GGUF_BLOCKS:
        return "gguf_blocks_v1"
    if layout == ROW_SPLIT_PANEL:
        return "row_split_panel4_v1"
    return String()
