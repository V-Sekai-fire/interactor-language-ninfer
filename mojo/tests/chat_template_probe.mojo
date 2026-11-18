"""Render chat conversations. Each line of argv[1] is a JSON case {template, special, messages,
options}; the output per case is `text HEX`, `lit B E`..., `reasoning 0|1`, `exec N,...` or
`err HEX` / `errjson`, the format of chat_template_oracle.cpp."""

from std.pathlib import Path
from std.sys import argv

from ninfer.artifact.json import Json, parse_json, BOOL
from ninfer.frontend.chat_template import (
    ChatMessage,
    ChatRenderOptions,
    CompiledChatTemplate,
    ChatPart,
    PromptCacheMarker,
    ToolCall,
    NONE,
    UNSET,
)
from ninfer.text.jinja.jstring import string_bytes


def hexs(data: String) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for b in data.as_bytes():
        out += digits[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += digits[byte=Int(b & 15) : Int(b & 15) + 1]
    return out


def _opt(doc: Json, o: Int, key: String) -> Int:
    var n = doc.find(o, key)
    if n < 0:
        return UNSET
    return 1 if doc.nodes[n].u != 0 else 0


def _str(doc: Json, o: Int, key: String) -> String:
    var n = doc.find(o, key)
    return doc.string(n) if n >= 0 else String()


def main() raises:
    var text = Path(String(argv()[1])).read_text()
    var cache_name = String()
    var cache_special = String()
    var compiled: Optional[CompiledChatTemplate] = None
    var n = 0
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        print("case", n)
        n += 1
        try:
            var doc = Json()
            var root = parse_json(string_bytes(String(line)), "case", doc, False)
            var name = _str(doc, root, "template")
            var special = doc.find(root, "special")
            var special_text = String("{}")
            if special >= 0:
                special_text = _str(doc, root, "special_json")
            if not compiled or name != cache_name or special_text != cache_special:
                var source = Path(name).read_text()
                compiled = CompiledChatTemplate(source, "chat_template.jinja", special_text)
                cache_name = name
                cache_special = special_text
            var messages = List[ChatMessage]()
            var ms = doc.find(root, "messages")
            for i in range(doc.size(ms)):
                var m = doc.at(ms, i)
                var msg = ChatMessage(Int(doc.nodes[doc.find(m, "role")].u))
                msg.parts = List[ChatPart]()
                var ps = doc.find(m, "parts")
                for k in range(doc.size(ps)):
                    var part = doc.at(ps, k)
                    if doc.is_string(part):
                        msg.parts.append(ChatPart(0, doc.string(part)))
                    else:
                        msg.parts.append(ChatPart(Int(doc.nodes[doc.find(part, "kind")].u)))
                msg.reasoning_content = _str(doc, m, "reasoning")
                msg.tool_call_id = _str(doc, m, "tool_call_id")
                var tcs = doc.find(m, "tool_calls")
                if tcs >= 0:
                    for k in range(doc.size(tcs)):
                        var tc = doc.at(tcs, k)
                        msg.tool_calls.append(ToolCall(_str(doc, tc, "id"), _str(doc, tc, "name"), _str(doc, tc, "arguments")))
                messages.append(msg^)
            var options = ChatRenderOptions()
            var o = doc.find(root, "options")
            options.continue_final_assistant = _opt(doc, o, "continue") == 1
            var agp = _opt(doc, o, "add_generation_prompt")
            options.add_generation_prompt = agp != 0
            options.enable_thinking = _opt(doc, o, "enable_thinking")
            options.preserve_thinking = _opt(doc, o, "preserve_thinking")
            var re = doc.find(o, "reasoning_effort")
            if re >= 0:
                options.reasoning_effort = Int(doc.nodes[re].u)
            options.chat_template_kwargs_json = _str(doc, o, "kwargs")
            options.add_vision_id = _opt(doc, o, "add_vision_id") == 1
            options.forced_tool_name = _str(doc, o, "forced")
            var syntax = doc.find(o, "syntax")
            if syntax >= 0:
                options.tool_call_syntax = Int(doc.nodes[syntax].u)
            var markers = doc.find(o, "markers")
            if markers >= 0:
                for k in range(doc.size(markers)):
                    var mk = doc.at(markers, k)
                    var marker = PromptCacheMarker(Int(doc.nodes[doc.find(mk, "loc")].u))
                    marker.after_message_count = Int(doc.nodes[doc.find(mk, "messages")].u)
                    marker.leading_instruction_bytes = Int(doc.nodes[doc.find(mk, "bytes")].u)
                    marker.after_tool_count = Int(doc.nodes[doc.find(mk, "tools")].u)
                    marker.after_message_part_count = Int(doc.nodes[doc.find(mk, "parts")].u)
                    options.cache_markers.append(marker^)
            var tools = doc.find(o, "tools")
            if tools >= 0:
                for k in range(doc.size(tools)):
                    options.tool_jsons.append(doc.string(doc.at(tools, k)))
            var r = compiled.value().render(messages, options)
            print("text", hexs(r.text))
            for k in range(len(r.literal_begin)):
                print("lit", r.literal_begin[k], r.literal_end[k])
            print("reasoning", 1 if r.starts_in_reasoning else 0)
            var ex = String()
            for k in range(len(r.execution_boundaries)):
                if k > 0:
                    ex += ","
                ex += String(r.execution_boundaries[k])
            print("exec", ex)
            for k in range(len(r.media_placeholders)):
                ref pl = r.media_placeholders[k]
                print("media", pl.begin, pl.end, pl.modality, pl.item_index)
            var mb = String()
            for k in range(len(r.message_boundaries)):
                mb += ("-" if r.message_boundaries[k] == NONE else String(r.message_boundaries[k])) + ","
            print("msgb", mb)
            var cb = String()
            for k in range(len(r.cache_boundaries)):
                cb += ("-" if r.cache_boundaries[k] == NONE else String(r.cache_boundaries[k])) + ","
            print("cacheb", cb)
            print("rewrite", r.rewrite_checkpoint_kind, r.rewrite_checkpoint_offset)
        except e:
            var msg = String(e)
            if msg.startswith("JSON parse error"):
                print("errjson")
            else:
                print("err", hexs(msg))
