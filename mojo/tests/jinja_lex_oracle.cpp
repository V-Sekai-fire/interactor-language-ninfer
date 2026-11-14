// Token streams of third_party/llama-jinja's lexer in the format of jinja_lex_probe.mojo.
#include <jinja/lexer.h>

#include <fstream>
#include <iostream>
#include <sstream>

static std::string hex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out.push_back(d[c >> 4]); out.push_back(d[c & 15]); }
    return out;
}
static std::string unhex(const std::string& s) {
    std::string out;
    if (s == "-") { return out; }
    for (std::size_t i = 0; i + 1 < s.size(); i += 2) {
        out.push_back(static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)));
    }
    return out;
}

int main(int, char** argv) {
    std::ifstream in(argv[1]);
    int n = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) { continue; }
        std::cout << "case " << n++ << "\n";
        try {
            const auto r = jinja::lexer().tokenize(unhex(line));
            for (const auto& t : r.tokens) {
                std::cout << jinja::type_to_string(t.t) << " " << t.pos << " " << hex(t.value) << "\n";
            }
            std::cout << "source " << hex(r.source) << "\n";
        } catch (const std::exception& e) { std::cout << "err " << hex(e.what()) << "\n"; }
    }
}
