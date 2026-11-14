// Runs the engine's Qwen tool-call parser and decoder on the cases in argv[1]. One case per line,
// tab separated: kind (parse|stream), enabled (0|1), forced tool name (hex), max name length,
// tool definitions (hex, comma separated), input chunks (hex, comma separated; R = resolve_in_model).
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace ninfer;
using namespace ninfer::models::qwen3_5::frontend;

static std::string unhex(const std::string& h) {
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
    return out;
}
static std::string hex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out.push_back(d[c >> 4]); out.push_back(d[c & 15]); }
    return out;
}
static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) { if (c == sep) { out.push_back(cur); cur.clear(); } else cur.push_back(c); }
    out.push_back(cur);
    return out;
}
static void print_calls(const std::vector<GeneratedToolCall>& calls, const ToolCallParseDiagnostics& d) {
    std::printf(" n=%zu", calls.size());
    for (const auto& c : calls) std::printf(" %s:%s", hex(c.name).c_str(), hex(c.arguments_json).c_str());
    std::printf(" diag=%d,%d,%u,%u,%u,%u,%d,%d,%u,%d,%d\n", d.marker_seen, d.forced_call_closed, d.structured_call_count,
                d.empty_arguments_omitted, d.schema_mismatch_arguments, d.duplicate_parameters_repaired,
                static_cast<int>(d.fallback_reason), d.recovered, d.recovered_call_count, d.malformed_call_reported,
                d.trailing_content_dropped);
}

int main(int argc, char** argv) {
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split(line, '\t');
        std::vector<std::string> tools;
        if (!f[4].empty()) for (const auto& t : split(f[4], ',')) tools.push_back(unhex(t));
        try {
            auto contract = build_tool_call_output_contract(tools, f[1] == "1", unhex(f[2]));
            const std::size_t max_len = std::stoul(f[3]);
            if (f[0] == "format") {
                if (!contract) { std::printf("disabled\n"); continue; }
                std::printf("format %s\n", hex(structured_tool_call_format(*contract)).c_str());
            } else if (f[0] == "parse") {
                const std::string text = unhex(f[5]);
                if (!contract) { std::printf("disabled\n"); continue; }
                auto out = parse_qwen_tool_call_output(text, max_len, *contract);
                std::printf("parse resp=%d content=%s", out.is_tool_call_response, hex(out.content).c_str());
                print_calls(out.tool_calls, out.diagnostics);
            } else {
                ToolCallOutputDecoder decoder(contract, max_len);
                for (const auto& chunk : split(f[5], ',')) {
                    if (chunk == "R") std::printf("R=%s ", hex(decoder.resolve_in_model()).c_str());
                    else std::printf("v=%s ", hex(decoder.feed(unhex(chunk))).c_str());
                }
                auto t = decoder.finish();
                std::printf("T content=%s", hex(t.content).c_str());
                print_calls(t.tool_calls, t.diagnostics);
            }
        } catch (const std::exception&) { std::printf("EXC\n"); }
    }
}
