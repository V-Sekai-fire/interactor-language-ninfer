// Renders with the engine's CompiledChatTemplate; output format of chat_template_probe.mojo.
#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/media_cache.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace ninfer;
using namespace ninfer::models::qwen3_5::frontend;
using Json = nlohmann::ordered_json;
using FChatMessage = ninfer::models::qwen3_5::frontend::ChatMessage;

// The frontend headers reach check_preparation_control, which needs the media cache's whole
// translation unit; the renderer only calls it to honour a cancellation, which this harness never sets.
namespace ninfer::models::qwen3_5::frontend {
void check_preparation_control(const PreparationControl&, std::string_view) {}
}

static std::string hex(const std::string& s) {
    static const char* d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) { out.push_back(d[c >> 4]); out.push_back(d[c & 15]); }
    return out;
}

static std::optional<bool> opt(const Json& o, const char* key) {
    if (!o.contains(key)) return std::nullopt;
    return o[key].get<bool>();
}

int main(int, char** argv) {
    std::ifstream in(argv[1]);
    std::string cached_template, cached_special;
    std::optional<CompiledChatTemplate> compiled;
    int n = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        std::cout << "case " << n++ << "\n";
        try {
            const auto doc = Json::parse(line);
            const std::string name = doc["template"].get<std::string>();
            const std::string special = doc.contains("special_json") ? doc["special_json"].get<std::string>() : "{}";
            if (!compiled || name != cached_template || special != cached_special) {
                std::ifstream f(name);
                std::stringstream ss;
                ss << f.rdbuf();
                compiled.emplace(CompiledChatTemplate::resolve(ss.str(), "chat_template.jinja", Json::parse(special)));
                cached_template = name;
                cached_special  = special;
            }
            std::vector<FChatMessage> messages;
            for (const auto& m : doc["messages"]) {
                FChatMessage msg;
                msg.role = static_cast<ChatRole>(m["role"].get<int>());
                for (const auto& p : m["parts"]) {
                    if (p.is_string()) msg.parts.push_back(ChatPart::text_part(p.get<std::string>()));
                    else if (p["kind"].get<int>() == 1) msg.parts.push_back(ChatPart::image(MediaData{}));
                    else msg.parts.push_back(ChatPart::video(MediaData{}));
                }
                if (m.contains("reasoning")) msg.reasoning_content = m["reasoning"].get<std::string>();
                if (m.contains("tool_call_id")) msg.tool_call_id = m["tool_call_id"].get<std::string>();
                if (m.contains("tool_calls")) {
                    for (const auto& tc : m["tool_calls"]) {
                        msg.tool_calls.push_back({tc.value("id", ""), tc.value("name", ""), tc.value("arguments", "")});
                    }
                }
                messages.push_back(std::move(msg));
            }
            ChatRenderOptions options;
            const auto& o = doc["options"];
            if (opt(o, "continue").value_or(false)) options.continuation = PromptContinuationMode::ContinueFinalAssistant;
            options.add_generation_prompt = opt(o, "add_generation_prompt").value_or(true);
            options.enable_thinking       = opt(o, "enable_thinking");
            options.preserve_thinking     = opt(o, "preserve_thinking");
            if (o.contains("reasoning_effort")) options.reasoning_effort = static_cast<ReasoningEffort>(o["reasoning_effort"].get<int>());
            options.chat_template_kwargs_json = o.value("kwargs", "");
            options.add_vision_id             = opt(o, "add_vision_id").value_or(false);
            options.forced_tool_name          = o.value("forced", "");
            if (o.contains("syntax")) options.tool_call_syntax = static_cast<ToolCallSyntax>(o["syntax"].get<int>());
            if (o.contains("markers")) {
                for (const auto& mk : o["markers"]) {
                    PromptCacheMarker marker;
                    marker.location = static_cast<PromptCacheMarkerLocation>(mk["loc"].get<int>());
                    marker.after_message_count = mk["messages"].get<std::uint32_t>();
                    marker.leading_instruction_bytes = mk["bytes"].get<std::uint32_t>();
                    marker.after_tool_count = mk["tools"].get<std::uint32_t>();
                    marker.after_message_part_count = mk["parts"].get<std::uint32_t>();
                    options.cache_markers.push_back(marker);
                }
            }
            if (o.contains("tools")) for (const auto& t : o["tools"]) options.tool_jsons.push_back(t.get<std::string>());
            const auto r = compiled->render(messages, options);
            std::cout << "text " << hex(r.text) << "\n";
            for (const auto& s : r.literal_spans) std::cout << "lit " << s.begin << " " << s.end << "\n";
            std::cout << "reasoning " << (r.starts_in_reasoning ? 1 : 0) << "\n";
            std::cout << "exec ";
            for (std::size_t i = 0; i < r.rewrite_execution_boundaries.size(); ++i)
                std::cout << (i ? "," : "") << r.rewrite_execution_boundaries[i];
            std::cout << "\n";
            for (const auto& pl : r.media_placeholders)
                std::cout << "media " << pl.bytes.begin << " " << pl.bytes.end << " " << static_cast<int>(pl.modality) << " " << pl.item_index << "\n";
            std::cout << "msgb ";
            for (const auto& b : r.message_boundaries) std::cout << (b ? std::to_string(*b) : "-") << ",";
            std::cout << "\ncacheb ";
            for (const auto& b : r.cache_boundaries) std::cout << (b ? std::to_string(*b) : "-") << ",";
            std::cout << "\nrewrite " << (r.rewrite_checkpoint ? static_cast<int>(r.rewrite_checkpoint->kind) : -1) << " "
                      << (r.rewrite_checkpoint ? r.rewrite_checkpoint->offset : 0) << "\n";
        } catch (const nlohmann::json::exception&) { std::cout << "errjson\n";
        } catch (const std::exception& e) { std::cout << "err " << hex(e.what()) << "\n"; }
    }
}
