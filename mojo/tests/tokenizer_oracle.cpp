// The engine's own Tokenizer (src/models/qwen3_5/frontend/tokenizer.cpp) driven with the same
// case file and output format as tokenizer_probe.mojo.
#include "models/qwen3_5/frontend/tokenizer.h"

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

using namespace ninfer::models::qwen3_5::frontend;

static std::string slurp(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
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
static std::vector<long> ints(const std::string& s) {
    std::vector<long> out;
    if (s == "-") { return out; }
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) { out.push_back(std::stol(item)); }
    return out;
}

int main(int, char** argv) {
    const std::string tj = slurp(argv[1]), tc = slurp(argv[2]), gc = slurp(argv[3]);
    std::optional<Tokenizer> loaded;
    try { loaded.emplace(TokenizerResources{tj, tc, gc}); }
    catch (const std::exception& e) { std::cout << "load_err " << e.what() << "\n"; return 0; }
    Tokenizer& tok = *loaded;
    std::cout << "vocab " << tok.vocab_size() << " " << (tok.sentencepiece() ? "True" : "False") << "\n";
    std::cout << "stops ";
    for (int s : tok.default_stop_token_ids()) { std::cout << s << ","; }
    std::cout << "\n";
    std::ifstream cases(argv[4]);
    for (std::string line; std::getline(cases, line);) {
        if (line.empty()) { continue; }
        std::stringstream ss(line);
        std::vector<std::string> f;
        for (std::string item; std::getline(ss, item, ' ');) { f.push_back(item); }
        try {
            if (f[0] == "decode") {
                DecodeOptions opts;
                opts.skip_special_tokens = f[2] == "1";
                for (long v : ints(f[3])) { opts.stop_token_ids.push_back(static_cast<int>(v)); }
                std::vector<int> ids;
                for (long v : ints(f[1])) { ids.push_back(static_cast<int>(v)); }
                const std::string decoded = hex(tok.decode(ids, opts));
                std::cout << "ok " << decoded << "\n";
            } else {
                EncodeOptions opts;
                opts.parse_added_tokens = f[2] == "1";
                opts.max_tokens = static_cast<std::size_t>(std::stoull(f[3]));
                std::vector<std::size_t> bounds;
                if (f.size() > 4) { for (long v : ints(f[4])) { bounds.push_back(static_cast<std::size_t>(v)); } }
                std::vector<ninfer::text::ByteSpan> spans;
                if (f.size() > 5 && f[5] != "-") {
                    auto pairs = ints(f[5]);
                    for (std::size_t i = 0; i + 1 < pairs.size(); i += 2) {
                        spans.push_back({static_cast<std::size_t>(pairs[i]), static_cast<std::size_t>(pairs[i + 1])});
                    }
                }
                const std::string text = unhex(f[1]);
                auto r = tok.encode_with_boundaries(text, bounds, opts, spans);
                std::string out, bs;
                for (int id : r.input_ids) { out += std::to_string(id) + ","; }
                for (const auto& b : r.boundaries) {
                    bs += (b.exact_frontier ? std::to_string(*b.exact_frontier) : "x") + "/" + std::to_string(b.stable_frontier) + ",";
                }
                std::cout << "ok " << out << " " << bs << "\n";
            }
        } catch (const std::exception& e) { std::cout << "err " << e.what() << "\n"; }
    }
}
