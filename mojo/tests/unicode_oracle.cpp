// Runs the engine's own src/text/unicode.cpp (utf8proc) in the same line format as
// unicode_probe.mojo, so the Python test can diff the two byte for byte.
#include "text/unicode.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace u = ninfer::text::unicode_internal;

static std::string unhex(const std::string& s) {
    std::string out;
    for (std::size_t i = 0; i + 1 < s.size(); i += 2) {
        out.push_back(static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)));
    }
    return out;
}
static std::string hex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out.push_back(d[c >> 4]); out.push_back(d[c & 15]); }
    return out;
}

int main(int argc, char** argv) {
    const std::string mode = argv[1];
    if (mode == "classes") {
        std::string out;
        for (int cp = 0; cp < 0x110000; ++cp) {
            int m = (u::is_letter(cp) ? 1 : 0) | (u::is_mark(cp) ? 2 : 0) |
                    (u::is_number(cp) ? 4 : 0) | (u::is_whitespace(cp) ? 8 : 0);
            out.push_back("0123456789abcdef"[m]);
        }
        std::puts(out.c_str());
        return 0;
    }
    std::ifstream in(argv[2]);
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) { continue; }
        const std::string data = unhex(line);
        try {
            if (mode == "iter") {
                std::string s;
                for (const auto& sp : u::utf8_codepoints(data, "probe")) {
                    s += std::to_string(sp.value) + ":" + std::to_string(sp.length) + ",";
                }
                std::cout << "ok " << s << "\n";
            } else {
                const std::string normalized = hex(u::normalize_nfc(data));
                std::cout << "ok " << normalized << "\n";
            }
        } catch (const std::exception& e) { std::cout << "err " << e.what() << "\n"; }
    }
}
