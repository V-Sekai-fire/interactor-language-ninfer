"""The Mojo side of tool_call_oracle.cpp: same case format, same output format."""

from std.sys import argv

from ninfer.frontend.tool_call_parser import (
    GeneratedToolCall,
    ToolCallParseDiagnostics,
    ToolCallOutputDecoder,
    build_tool_call_output_contract,
    parse_qwen_tool_call_output,
)


def _hv(c: UInt8) -> Int:
    return Int(c) - 0x30 if c <= 0x39 else Int(c) - 0x61 + 10


def _unhex(h: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = h.as_bytes()
    var i = 0
    while i + 1 < len(b):
        out.append(UInt8(_hv(b[i]) * 16 + _hv(b[i + 1])))
        i += 2
    return out^


def _hex(bytes: List[UInt8]) -> String:
    var d = "0123456789abcdef"
    var out = String()
    for b in bytes:
        out += d[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += d[byte=Int(b & 15) : Int(b & 15) + 1]
    return out^


def _hex_str(s: String) -> String:
    var v = List[UInt8]()
    for b in s.as_bytes():
        v.append(b)
    return _hex(v)


def _split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    for p in s.split(sep):
        out.append(String(p))
    return out^


def _b(v: Bool) -> String:
    return "1" if v else "0"


def _calls(calls: List[GeneratedToolCall], d: ToolCallParseDiagnostics) -> String:
    var out = String(" n=") + String(len(calls))
    for c in calls:
        out += " " + _hex_str(c.name) + ":" + _hex_str(c.arguments_json)
    out += (
        " diag=" + _b(d.marker_seen) + "," + _b(d.forced_call_closed) + "," + String(d.structured_call_count)
        + "," + String(d.empty_arguments_omitted) + "," + String(d.schema_mismatch_arguments)
        + "," + String(d.duplicate_parameters_repaired) + "," + String(d.fallback_reason)
        + "," + _b(d.recovered) + "," + String(d.recovered_call_count) + "," + _b(d.malformed_call_reported)
        + "," + _b(d.trailing_content_dropped)
    )
    return out


def _run(line: String, mut out: String) raises -> String:
    var f = _split(line, "\t")
    var tools = List[String]()
    if f[4].byte_length() > 0:
        for t in _split(f[4], ","):
            tools.append(String(unsafe_from_utf8=_unhex(t)))
    var forced = String(unsafe_from_utf8=_unhex(f[2]))
    var contract = build_tool_call_output_contract(tools, f[1] == "1", forced)
    var max_len = atol(f[3])
    if f[0] == "parse":
        if not contract.enabled:
            return "disabled"
        var r = parse_qwen_tool_call_output(_unhex(f[5]), max_len, contract)
        return "parse resp=" + _b(r.is_tool_call_response) + " content=" + _hex(r.content) + _calls(r.tool_calls, r.diagnostics)
    var decoder = ToolCallOutputDecoder(contract, max_len)
    for chunk in _split(f[5], ","):
        if chunk == "R":
            out += "R=" + _hex(decoder.resolve_in_model()) + " "
        else:
            out += "v=" + _hex(decoder.feed(_unhex(chunk))) + " "
    var t = decoder.finish()
    return out + "T content=" + _hex(t.content) + _calls(t.tool_calls, t.diagnostics)


def main() raises:
    var text = open(String(argv()[1]), "r").read()
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var partial = String()
        try:
            print(_run(String(line), partial))
        except:
            # the C++ oracle has already printed what the decoder returned before it threw
            print(partial + "EXC")
