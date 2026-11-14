// Renders with src/text/jinja.cpp; output format of jinja_render_probe.mojo.
#include "text/jinja.h"

#include <fstream>
#include <iostream>

static std::string hex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out.push_back(d[c >> 4]); out.push_back(d[c & 15]); }
    return out;
}

int main(int, char** argv) {
    std::ifstream in(argv[1]);
    int n = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) { continue; }
        std::cout << "case " << n++ << "\n";
        try {
            const auto doc = nlohmann::ordered_json::parse(line);
            std::vector<ninfer::text::TemplateInputRegion> regions;
            for (const auto& r : doc["regions"]) { regions.push_back({r["pointer"].get<std::string>(), r["tag"].get<std::uint32_t>()}); }
            std::vector<std::string> controls;
            for (const auto& c : doc["controls"]) { controls.push_back(c.get<std::string>()); }
            ninfer::text::JinjaTemplate tmpl(doc["template"].get<std::string>(), "t");
            ninfer::text::TemplateRenderOptions options;
            options.timestamp = doc["timestamp"].get<std::time_t>();
            options.regions = regions;
            options.control_variables = controls;
            const auto out = tmpl.render(doc["context"], options);
            std::cout << "text " << hex(out.text) << "\n";
            for (const auto& s : out.literal_spans) { std::cout << "lit " << s.begin << " " << s.end << "\n"; }
            for (const auto& r : out.regions) {
                std::cout << "reg " << r.tag << " " << r.begin << " " << r.end << " "
                          << (r.source_offset ? static_cast<long>(*r.source_offset) : -1L) << "\n";
            }
        } catch (const std::exception& e) { std::cout << "err " << hex(e.what()) << "\n"; }
    }
}
