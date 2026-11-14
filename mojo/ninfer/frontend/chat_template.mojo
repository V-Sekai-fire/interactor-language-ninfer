"""Chat template rendering for the Qwen3.5 family and Gemma 4; mirrors
`src/models/qwen3_5/frontend/chat_template.{h,cpp}` and `prompt_layout.{h,cpp}`.

A conversation becomes the template's `messages`/`tools` context, the Jinja template renders it
(`ninfer.text.jinja`), and the layout pass reads the rendered text back: where each turn starts and
ends, whether the generation prompt leaves the model inside its reasoning block, and the
byte offsets at which the engine executes the assistant turn. The bytes that came from the
request (message text, tool definitions) are reported as literal spans, so the tokenizer never
reads a control token out of user text.

Not ported, because nothing here consumes them yet: media parts (a message with an image or video
is not representable in `ChatMessage`), and the prefix-cache bookkeeping of the C++ renderer
(`message_boundaries`, `cache_boundaries`, `rewrite_checkpoint`).
"""

from std.collections import List
from std.ffi import external_call

from ninfer.artifact.json import (
    Json,
    parse_json,
    ARRAY,
    BOOL,
    FLOAT,
    NEG,
    NULL,
    OBJECT,
    STRING,
    UINT,
)
from ninfer.text.jinja.grisu import dump_double
from ninfer.text.jinja.template import InputRegion, JinjaTemplate, OutputRegion, TemplateOutput
from ninfer.text.jinja.value import json_escape

comptime ROLE_SYSTEM = 0
comptime ROLE_DEVELOPER = 1
comptime ROLE_USER = 2
comptime ROLE_ASSISTANT = 3
comptime ROLE_TOOL = 4

comptime SYNTAX_QWEN = 0
comptime SYNTAX_GEMMA = 1

comptime UNSET = -1  # an `std::optional` that holds no value

comptime CANONICAL_REASONING_CLOSE = "\n</think>\n\n"
comptime GEMMA_CALL_OPEN = "<|tool_call>"
comptime GEMMA_CALL_HEAD = "call:"
comptime GEMMA_ARGUMENTS_OPEN = "{"

comptime _JSON_ERROR = "JSON parse error: "


def role_name(role: Int) raises -> String:
    if role == ROLE_SYSTEM:
        return "system"
    if role == ROLE_DEVELOPER:
        return "developer"
    if role == ROLE_USER:
        return "user"
    if role == ROLE_ASSISTANT:
        return "assistant"
    if role == ROLE_TOOL:
        return "tool"
    raise Error("invalid chat role")


def _instruction(role: Int) -> Bool:
    return role == ROLE_SYSTEM or role == ROLE_DEVELOPER


# --- request types -----------------------------------------------------------------------------


struct ToolCall(Copyable, Movable):
    var id: String
    var name: String
    var arguments_json: String

    def __init__(out self, id: String, name: String, arguments_json: String):
        self.id = id
        self.name = name
        self.arguments_json = arguments_json


struct ChatMessage(Copyable, Movable):
    var role: Int
    var parts: List[String]  # text parts; media parts are not supported
    var reasoning_content: String
    var tool_calls: List[ToolCall]
    var tool_call_id: String

    def __init__(out self, role: Int, text: String = String()):
        self.role = role
        self.parts = List[String]()
        self.parts.append(text)
        self.reasoning_content = String()
        self.tool_calls = List[ToolCall]()
        self.tool_call_id = String()


struct ChatRenderOptions(Copyable, Movable):
    var continue_final_assistant: Bool  # PromptContinuationMode::ContinueFinalAssistant
    var add_generation_prompt: Bool
    var enable_thinking: Int  # UNSET, 0 or 1
    var reasoning_effort: Int  # UNSET, or ReasoningEffort 0 (none) .. 6 (max)
    var preserve_thinking: Int  # UNSET, 0 or 1
    var chat_template_kwargs_json: String
    var add_vision_id: Bool
    var tool_jsons: List[String]
    var forced_tool_name: String
    var tool_call_syntax: Int

    def __init__(out self):
        self.continue_final_assistant = False
        self.add_generation_prompt = True
        self.enable_thinking = UNSET
        self.reasoning_effort = UNSET
        self.preserve_thinking = UNSET
        self.chat_template_kwargs_json = String()
        self.add_vision_id = False
        self.tool_jsons = List[String]()
        self.forced_tool_name = String()
        self.tool_call_syntax = SYNTAX_QWEN


struct RenderedChat(Movable):
    var text: String
    var literal_begin: List[Int]
    var literal_end: List[Int]
    var starts_in_reasoning: Bool
    var execution_boundaries: List[Int]

    def __init__(out self):
        self.text = String()
        self.literal_begin = List[Int]()
        self.literal_end = List[Int]()
        self.starts_in_reasoning = False
        self.execution_boundaries = List[Int]()


# --- JSON context ------------------------------------------------------------------------------


struct JObj(Copyable, Movable):
    """An insertion-ordered JSON object whose values are JSON text, as `nlohmann::ordered_json`
    holds the template context."""

    var keys: List[String]
    var vals: List[String]

    def __init__(out self):
        self.keys = List[String]()
        self.vals = List[String]()

    def find(self, key: String) -> Int:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return i
        return -1

    def has(self, key: String) -> Bool:
        return self.find(key) >= 0

    def get(self, key: String) -> String:
        return self.vals[self.find(key)]

    def set(mut self, key: String, value: String):
        var i = self.find(key)
        if i >= 0:
            self.vals[i] = value
        else:
            self.keys.append(key)
            self.vals.append(value)

    def erase(mut self, key: String):
        var i = self.find(key)
        if i < 0:
            return
        var keys = List[String]()
        var vals = List[String]()
        for k in range(len(self.keys)):
            if k != i:
                keys.append(self.keys[k])
                vals.append(self.vals[k])
        self.keys = keys^
        self.vals = vals^

    def dump(self) raises -> String:
        var out = String("{")
        for i in range(len(self.keys)):
            if i > 0:
                out += ","
            out += json_escape(self.keys[i], False) + ":" + self.vals[i]
        out += "}"
        return out^


def _bytes_of(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


def _parse(text: String, label: String, mut doc: Json) raises -> Int:
    """`Json::parse`: the last of a repeated member wins; failure is a JSON error."""
    try:
        return parse_json(_bytes_of(text), label, doc, False)
    except e:
        raise Error(_JSON_ERROR + String(e))


def _dump(doc: Json, n: Int) raises -> String:
    """`nlohmann::json::dump()` of a parsed value."""
    var kind = doc.kind(n)
    if kind == NULL:
        return "null"
    if kind == BOOL:
        return "true" if doc.nodes[n].u != 0 else "false"
    if kind == UINT:
        return String(doc.nodes[n].u)
    if kind == NEG:
        return "-" + String(doc.nodes[n].u)
    if kind == FLOAT:
        return dump_double(doc.nodes[n].f)
    if kind == STRING:
        return json_escape(doc.string(n), False)
    var out = String()
    if kind == ARRAY:
        out += "["
        for i in range(doc.size(n)):
            if i > 0:
                out += ","
            out += _dump(doc, doc.at(n, i))
        return out + "]"
    out += "{"
    for i in range(doc.size(n)):
        if i > 0:
            out += ","
        out += json_escape(doc.key(n, i), False) + ":" + _dump(doc, doc.at(n, i))
    return out + "}"


def _object_of(text: String, label: String) raises -> JObj:
    var doc = Json()
    var root = _parse(text, label, doc)
    var out = JObj()
    if not doc.is_object(root):
        return out^
    for i in range(doc.size(root)):
        out.set(doc.key(root, i), _dump(doc, doc.at(root, i)))
    return out^


# --- thinking efforts --------------------------------------------------------------------------


def _effort_name(index: Int) -> String:
    """Thinking efforts in ascending order; bit i of an accepted-effort mask stands for entry i."""
    if index == 0:
        return "minimal"
    if index == 1:
        return "low"
    if index == 2:
        return "medium"
    if index == 3:
        return "high"
    if index == 4:
        return "xhigh"
    return "max"


comptime _EFFORT_COUNT = 6
comptime _ALL_EFFORTS = 0x3F


def reasoning_effort_name(effort: Int) -> String:
    if effort == 0:
        return "none"
    if effort >= 1 and effort <= 6:
        return _effort_name(effort - 1)
    return String()


def _accepted_effort(requested: String, accepted: Int) -> String:
    """Templates accept different effort sets (the official Qwen3.8 template only low, medium and
    xhigh), while clients send the standard values. A rejected effort becomes the nearest accepted
    one, and a tie rounds up so the request thinks at least as much as it asked: high and max
    become xhigh there. A template that accepts none keeps the request's value and its error."""
    var index = -1
    for i in range(_EFFORT_COUNT):
        if _effort_name(i) == requested:
            index = i
    if index < 0 or accepted == 0:
        return requested
    if ((accepted >> index) & 1) != 0:
        return requested
    for distance in range(1, _EFFORT_COUNT):
        if index + distance < _EFFORT_COUNT and ((accepted >> (index + distance)) & 1) != 0:
            return _effort_name(index + distance)
        if index >= distance and ((accepted >> (index - distance)) & 1) != 0:
            return _effort_name(index - distance)
    return requested


def _now() -> Int:
    return external_call["time", Int](Int(0))


def _control_variables(special: JObj) -> List[String]:
    return special.keys.copy()


def _renders(compiled: JinjaTemplate, special: JObj, controls: List[String], effort: Int) -> Bool:
    try:
        var context = JObj()
        context.set("enable_thinking", "true")
        if effort >= 0:
            context.set("reasoning_effort", '"' + _effort_name(effort) + '"')
        context.set("continue_final_message", "false")
        context.set("messages", '[{"role":"user","content":"probe"}]')
        context.set("add_generation_prompt", "true")
        for i in range(len(special.keys)):
            context.set(special.keys[i], special.vals[i])
        var doc = Json()
        var root = _parse(context.dump(), "probe", doc)
        _ = compiled.render(doc, root, List[InputRegion](), controls, _now())
        return True
    except:
        return False


def _probe_accepted_efforts(compiled: JinjaTemplate, special: JObj) -> Int:
    """Renders a one-message thinking chat once per effort. A template that cannot render the
    probe without an effort says nothing about its efforts, so every effort stays accepted."""
    var controls = _control_variables(special)
    if not _renders(compiled, special, controls, -1):
        return _ALL_EFFORTS
    var accepted = 0
    for i in range(_EFFORT_COUNT):
        if _renders(compiled, special, controls, i):
            accepted |= 1 << i
    return accepted


# --- template parameters -----------------------------------------------------------------------


def _merge_option(mut context: JObj, key: String, value: String) raises:
    if context.has(key) and context.get(key) != "null" and context.get(key) != value:
        raise Error("conflicting chat template parameter: " + key)
    context.set(key, value)


def _is_reserved(key: String) -> Bool:
    return (
        key == "messages"
        or key == "tools"
        or key == "add_generation_prompt"
        or key == "continue_final_message"
        or key == "bos_token"
        or key == "eos_token"
        or key == "pad_token"
        or key == "unk_token"
        or key == "sep_token"
        or key == "cls_token"
        or key == "mask_token"
        or key == "additional_special_tokens"
    )


def _bool_text(v: Bool) -> String:
    return "true" if v else "false"


def _template_parameters(options: ChatRenderOptions, special: JObj, accepted_efforts: Int) raises -> JObj:
    var context = JObj()
    if options.chat_template_kwargs_json.byte_length() > 0:
        var doc = Json()
        var root = _parse(options.chat_template_kwargs_json, "chat_template_kwargs", doc)
        if not doc.is_object(root):
            raise Error("chat_template_kwargs must be an object")
        for i in range(doc.size(root)):
            context.set(doc.key(root, i), _dump(doc, doc.at(root, i)))
    for key in context.keys:
        if _is_reserved(key) or special.has(key):
            raise Error("chat_template_kwargs cannot override " + key)
    for name in ["enable_thinking", "preserve_thinking", "reasoning_effort", "add_vision_id"]:
        if context.has(String(name)) and context.get(String(name)) == "null":
            context.erase(String(name))
    if options.enable_thinking != UNSET:
        _merge_option(context, "enable_thinking", _bool_text(options.enable_thinking == 1))
    if options.preserve_thinking != UNSET:
        _merge_option(context, "preserve_thinking", _bool_text(options.preserve_thinking == 1))
    if options.reasoning_effort != UNSET:
        var name = reasoning_effort_name(options.reasoning_effort)
        if name.byte_length() == 0:
            raise Error("invalid reasoning effort")
        _merge_option(context, "reasoning_effort", json_escape(name, False))
    if options.add_vision_id:
        _merge_option(context, "add_vision_id", "true")
    for name in ["enable_thinking", "preserve_thinking", "add_vision_id"]:
        if context.has(String(name)):
            var v = context.get(String(name))
            if v != "true" and v != "false":
                raise Error(String(name) + " must be a boolean")
    if context.has("reasoning_effort"):
        var effort = context.get("reasoning_effort")
        var valid = False
        for name in ["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
            if effort == '"' + String(name) + '"':
                valid = True
        if not valid:
            raise Error("invalid reasoning_effort template parameter")
        var thinking = effort != '"none"'
        if context.has("enable_thinking") and context.get("enable_thinking") != _bool_text(thinking):
            raise Error("reasoning_effort conflicts with enable_thinking")
        # The protocol's 'none' is an explicit disable. Other efforts retain template defaults and
        # render as the nearest effort the template accepts.
        if not thinking:
            context.set("enable_thinking", "false")
        else:
            var requested = String(effort[byte=1 : effort.byte_length() - 1])
            context.set("reasoning_effort", '"' + _accepted_effort(requested, accepted_efforts) + '"')
    for i in range(len(special.keys)):
        context.set(special.keys[i], special.vals[i])
    return context^


# --- prompt layout -----------------------------------------------------------------------------

comptime _QWEN_START = "<|im_start|>"
comptime _QWEN_END = "<|im_end|>"
# Gemma 4's turns: <|turn>role\n ... <turn|>, the assistant's role named "model".
comptime _GEMMA_START = "<|turn>"
comptime _GEMMA_END = "<turn|>"
comptime _VISION_START = "<|vision_start|>"
comptime _VISION_END = "<|vision_end|>"
comptime NPOS = -1


struct MessageLayout(Copyable, Movable):
    var role: Int
    var begin: Int
    var content_begin: Int
    var content_end: Int
    var end: Int
    var closed: Bool

    def __init__(out self, role: Int, begin: Int, content_begin: Int, content_end: Int, end: Int, closed: Bool):
        self.role = role
        self.begin = begin
        self.content_begin = content_begin
        self.content_end = content_end
        self.end = end
        self.closed = closed


struct PromptLayout(Movable):
    var messages: List[MessageLayout]
    var execution_boundaries: List[Int]
    var starts_in_reasoning: Bool

    def __init__(out self):
        self.messages = List[MessageLayout]()
        self.execution_boundaries = List[Int]()
        self.starts_in_reasoning = False


def _overlaps(output: TemplateOutput, begin: Int, end: Int) -> Bool:
    """`text::overlaps`: spans are nonempty, ordered and disjoint; a partial overlap counts."""
    if begin >= end:
        return False
    for i in range(len(output.literal_begin)):
        if output.literal_end[i] > begin:
            return output.literal_begin[i] < end
    return False


def _template_bytes(output: TemplateOutput, begin: Int, end: Int) -> Bool:
    return not _overlaps(output, begin, end)


def _find_from(source: Span[UInt8, _], marker: String, begin: Int) -> Int:
    var m = marker.as_bytes()
    var i = begin
    while i + len(m) <= len(source):
        var hit = True
        for k in range(len(m)):
            if source[i + k] != m[k]:
                hit = False
                break
        if hit:
            return i
        i += 1
    return NPOS


def _find_marker(output: TemplateOutput, marker: String, begin: Int) -> Int:
    var source = output.text.as_bytes()
    var pos = _find_from(source, marker, begin)
    while pos != NPOS:
        if _template_bytes(output, pos, pos + marker.byte_length()):
            return pos
        pos = _find_from(source, marker, pos + marker.byte_length())
    return NPOS


def _starts_with(source: Span[UInt8, _], begin: Int, end: Int, prefix: String) -> Bool:
    var p = prefix.as_bytes()
    if end - begin < len(p):
        return False
    for i in range(len(p)):
        if source[begin + i] != p[i]:
            return False
    return True


def _equals(source: Span[UInt8, _], begin: Int, end: Int, text: String) -> Bool:
    return end - begin == text.byte_length() and _starts_with(source, begin, end, text)


def _min_pos(a: Int, b: Int) -> Int:
    """`std::min` over size_t, where npos is the largest value."""
    if a == NPOS:
        return b
    if b == NPOS:
        return a
    return a if a < b else b


def inspect_prompt_layout(output: TemplateOutput, media_count: Int) raises -> PromptLayout:
    var result = PromptLayout()
    var source = output.text.as_bytes()
    var size = len(source)
    var gemma = _find_marker(output, _QWEN_START, 0) == NPOS and _find_marker(output, _GEMMA_START, 0) != NPOS
    var start = String(_GEMMA_START) if gemma else String(_QWEN_START)
    var end_marker = String(_GEMMA_END) if gemma else String(_QWEN_END)
    var pos = _find_marker(output, start, 0)
    while pos != NPOS:
        var header_end = NPOS
        for k in range(pos + start.byte_length(), size):
            if source[k] == 0x0A:
                header_end = k
                break
        if header_end == NPOS:
            break
        var rb = pos + start.byte_length()
        var role = -1
        if _equals(source, rb, header_end, "assistant") or (gemma and _equals(source, rb, header_end, "model")):
            role = ROLE_ASSISTANT
        elif _equals(source, rb, header_end, "user"):
            role = ROLE_USER
        elif _equals(source, rb, header_end, "system"):
            role = ROLE_SYSTEM
        elif _equals(source, rb, header_end, "developer"):
            role = ROLE_DEVELOPER
        elif _equals(source, rb, header_end, "tool"):
            role = ROLE_TOOL
        else:
            pos = _find_marker(output, start, header_end + 1)
            continue
        var next = _find_marker(output, start, header_end + 1)
        var close = _find_marker(output, end_marker, header_end + 1)
        var closed = close != NPOS and (next == NPOS or close < next)
        var content_end = close if closed else (size if next == NPOS else next)
        var end = close + end_marker.byte_length() if closed else content_end
        if closed and end < size and source[end] == 0x0A:
            end += 1
        result.messages.append(MessageLayout(role, pos, header_end + 1, content_end, end, closed))
        if role == ROLE_ASSISTANT:
            result.execution_boundaries.append(header_end + 1)
            var open = "<think>\n"
            var body_begin = header_end + 1
            var body_end = content_end
            if _starts_with(source, body_begin, body_end, open) and _template_bytes(
                output, header_end + 1, header_end + 1 + open.byte_length()
            ):
                result.execution_boundaries.append(header_end + 1 + open.byte_length())
                var reasoning_close = _find_marker(
                    output, CANONICAL_REASONING_CLOSE, header_end + 1 + open.byte_length()
                )
                if reasoning_close != NPOS and reasoning_close < content_end:
                    result.execution_boundaries.append(reasoning_close + String(CANONICAL_REASONING_CLOSE).byte_length())
            if not closed and next == NPOS:
                # Only the final assistant prefix controls the initial output channel.
                var open_end = NPOS
                if _starts_with(source, body_begin, body_end, "<think>") and _template_bytes(
                    output, header_end + 1, header_end + 1 + 7
                ):
                    open_end = header_end + 1 + 7
                var close_think = NPOS if open_end == NPOS else _find_marker(output, "</think>", open_end)
                result.starts_in_reasoning = open_end != NPOS and close_think == NPOS
        pos = next

    # Gemma 4's template writes one <|image|> or <|video|> per media input, in input order; the
    # processor adds the begin and end markers around the soft tokens it expands it into.
    var placeholders = 0
    if gemma:
        var p = 0
        while p < size:
            var image = _find_marker(output, "<|image|>", p)
            var video = _find_marker(output, "<|video|>", p)
            var pad = _min_pos(image, video)
            if pad == NPOS:
                break
            if placeholders >= media_count:
                raise Error("chat template media placeholders must match input count, type and order")
            placeholders += 1
            p = pad + 9  # <|image|> and <|video|>
        if placeholders != media_count:
            raise Error("chat template omitted an input media placeholder")
        return result^
    # Each media input contributes one complete Qwen placeholder in input order.
    var p = 0
    while p < size:
        var image = _find_marker(output, "<|image_pad|>", p)
        var video = _find_marker(output, "<|video_pad|>", p)
        var pad = _min_pos(image, video)
        if pad == NPOS:
            break
        var token = String("<|image_pad|>") if pad == image else String("<|video_pad|>")
        var vs = String(_VISION_START)
        var ve = String(_VISION_END)
        if (
            pad < vs.byte_length()
            or not _equals(source, pad - vs.byte_length(), pad, vs)
            or not _starts_with(source, pad + token.byte_length(), min(size, pad + token.byte_length() + ve.byte_length()), ve)
            or not _template_bytes(output, pad - vs.byte_length(), pad + token.byte_length() + ve.byte_length())
        ):
            raise Error("chat template media pad requires a complete Qwen vision wrapper")
        if placeholders >= media_count:
            raise Error("chat template media placeholders must match input count, type and order")
        placeholders += 1
        p = pad + token.byte_length()
    if placeholders != media_count:
        raise Error("chat template omitted an input media placeholder")
    return result^


def _unique_output_region(output: TemplateOutput, tag: Int) -> Span2:
    """The one region carrying `tag`; begin == -1 when there is none or more than one."""
    var found = Span2(-1, -1)
    var count = 0
    for r in output.regions:
        if r.tag != tag:
            continue
        count += 1
        found = Span2(r.begin, r.end)
    if count != 1:
        return Span2(-1, -1)
    return found^


struct Span2(Copyable, Movable):
    var begin: Int
    var end: Int

    def __init__(out self, begin: Int, end: Int):
        self.begin = begin
        self.end = end


# --- the compiled template ---------------------------------------------------------------------


struct CompiledChatTemplate(Movable):
    var compiled: JinjaTemplate
    var special_tokens: JObj
    # One bit per thinking effort, minimal through max, that the template renders without raising,
    # found by rendering each once at resolve. A rejected effort renders as the nearest accepted
    # one. Every bit is set until probing ends and whenever the probe cannot render.
    var accepted_efforts: Int

    def __init__(out self, source: String, source_name: String = "chat_template.jinja", special_tokens_json: String = "{}") raises:
        self.compiled = JinjaTemplate(source, source_name)
        self.special_tokens = _object_of(special_tokens_json, "special tokens")
        self.accepted_efforts = _ALL_EFFORTS
        self.accepted_efforts = _probe_accepted_efforts(self.compiled, self.special_tokens)

    def render(self, messages: List[ChatMessage], options: ChatRenderOptions) raises -> RenderedChat:
        try:
            return self._render_as(messages, options, False)
        except e:
            # The upstream Qwen templates raise on the developer role. A developer message is a
            # system-level instruction, so a template that refuses it gets it as one; a template
            # that refuses the system role as well still fails.
            var any_developer = False
            for m in messages:
                if m.role == ROLE_DEVELOPER:
                    any_developer = True
            if not any_developer:
                raise e^
            return self._render_as(messages, options, True)

    def _render_as(
        self, messages: List[ChatMessage], options: ChatRenderOptions, developer_as_system: Bool
    ) raises -> RenderedChat:
        if len(messages) == 0:
            raise Error("chat requires at least one message")
        var continuation = options.continue_final_assistant
        var context = _template_parameters(options, self.special_tokens, self.accepted_efforts)
        if continuation:
            ref final = messages[len(messages) - 1]
            if (
                final.role != ROLE_ASSISTANT
                or final.reasoning_content.byte_length() > 0
                or len(final.tool_calls) > 0
                or (context.has("enable_thinking") and context.get("enable_thinking") == "true")
            ):
                raise Error("assistant continuation requires a final text-only assistant message and disabled thinking")
        context.set("continue_final_message", _bool_text(continuation))
        context.set("add_generation_prompt", _bool_text(not continuation and options.add_generation_prompt))
        var regions = List[InputRegion]()
        var next_tag = 1
        var content_tags = List[Int]()
        var message_texts = List[String]()
        for i in range(len(messages)):
            ref message = messages[i]
            if _instruction(message.role) and (
                message.reasoning_content.byte_length() > 0
                or len(message.tool_calls) > 0
                or message.tool_call_id.byte_length() > 0
            ):
                raise Error("system and developer messages may contain only text")
            var role = String("system") if (message.role == ROLE_DEVELOPER and developer_as_system) else role_name(message.role)
            var pointer = "/messages/" + String(i)
            var content = String()
            for part in message.parts:
                content += part
            var value = '{"role":' + json_escape(role, False) + ',"content":' + json_escape(content, False)
            content_tags.append(next_tag)
            regions.append(InputRegion(pointer + "/content", next_tag))
            next_tag += 1
            if message.reasoning_content.byte_length() > 0:
                value += ',"reasoning_content":' + json_escape(message.reasoning_content, False)
                regions.append(InputRegion(pointer + "/reasoning_content", next_tag))
                next_tag += 1
            if len(message.tool_calls) > 0:
                value += ',"tool_calls":['
                for c in range(len(message.tool_calls)):
                    ref call = message.tool_calls[c]
                    var arguments = String("{}")
                    if call.arguments_json.byte_length() > 0:
                        var check = Json()
                        _ = _parse(call.arguments_json, "tool call arguments", check)
                        arguments = call.arguments_json
                    if c > 0:
                        value += ","
                    value += (
                        '{"id":' + json_escape(call.id, False)
                        + ',"type":"function","function":{"name":' + json_escape(call.name, False)
                        + ',"arguments":' + arguments + "}}"
                    )
                value += "]"
            if message.tool_call_id.byte_length() > 0:
                value += ',"tool_call_id":' + json_escape(message.tool_call_id, False)
            value += "}"
            message_texts.append(value^)
        var joined = String("[")
        for i in range(len(message_texts)):
            if i > 0:
                joined += ","
            joined += message_texts[i]
        context.set("messages", joined + "]")
        if len(options.tool_jsons) > 0:
            var tools = String("[")
            for i in range(len(options.tool_jsons)):
                var check = Json()
                _ = _parse(options.tool_jsons[i], "tool definition", check)
                if i > 0:
                    tools += ","
                tools += options.tool_jsons[i]
                regions.append(InputRegion("/tools/" + String(i), next_tag))
                next_tag += 1
            context.set("tools", tools + "]")
        var controls = _control_variables(self.special_tokens)
        if options.forced_tool_name.byte_length() > 0:
            # The opener of the forced call is appended to the generation prompt, so the turn has
            # to open fresh, with the reasoning block already closed (a thinking prompt ends
            # inside it) and a declared tool to name.
            if continuation or not options.add_generation_prompt:
                raise Error("a forced tool call requires a new assistant turn to open")
            if options.enable_thinking != 0:
                raise Error("a forced tool call cannot start in thinking mode")
            if len(options.tool_jsons) == 0:
                raise Error("a forced tool call requires declared tools")
        var doc = Json()
        var root = _parse(context.dump(), "chat context", doc)
        var output = self.compiled.render(doc, root, regions, controls, _now())
        var layout = inspect_prompt_layout(output, 0)
        if continuation:
            var content = _unique_output_region(output, content_tags[len(content_tags) - 1])
            if content.begin < 0 or len(layout.messages) == 0:
                raise Error("chat template does not expose the final assistant content for continuation")
            ref last = layout.messages[len(layout.messages) - 1]
            if last.role != ROLE_ASSISTANT or content.begin < last.content_begin or content.end > last.content_end:
                raise Error("chat template cannot continue the final assistant content unambiguously")
            var kept = List[UInt8]()
            var all_bytes = output.text.as_bytes()
            for k in range(content.end):
                kept.append(all_bytes[k])
            output.text = String(unsafe_from_utf8=kept)
            var limit = content.end
            var new_regions = List[OutputRegion]()
            for r in output.regions:
                if r.begin > limit:
                    continue
                var copy = r.copy()
                copy.end = min(copy.end, limit)
                new_regions.append(copy^)
            output.regions = new_regions^
            var lb = List[Int]()
            var le = List[Int]()
            for k in range(len(output.literal_begin)):
                if output.literal_begin[k] >= limit:
                    continue
                lb.append(output.literal_begin[k])
                le.append(output.literal_end[k])
            if len(le) > 0:
                le[len(le) - 1] = min(le[len(le) - 1], limit)
            output.literal_begin = lb^
            output.literal_end = le^
            layout = inspect_prompt_layout(output, 0)
        var result = RenderedChat()
        result.starts_in_reasoning = layout.starts_in_reasoning
        result.execution_boundaries = layout.execution_boundaries.copy()
        var text = _bytes_of(output.text)
        var lit_begin = output.literal_begin.copy()
        var lit_end = output.literal_end.copy()
        if options.forced_tool_name.byte_length() > 0:
            # The syntax's call opens with the function name; everything the model produces after
            # this can only continue that call. The name is caller text, so it is a literal span
            # like any other user-sourced bytes.
            var head = String()
            var tail = String()
            if options.tool_call_syntax == SYNTAX_QWEN:
                head = "<tool_call>\n<function="
                tail = ">\n"
            elif options.tool_call_syntax == SYNTAX_GEMMA:
                head = String(GEMMA_CALL_OPEN) + String(GEMMA_CALL_HEAD)
                tail = GEMMA_ARGUMENTS_OPEN
            else:
                raise Error("invalid tool call syntax")
            text.extend(_bytes_of(head))
            var name_begin = len(text)
            text.extend(_bytes_of(options.forced_tool_name))
            lit_begin.append(name_begin)
            lit_end.append(len(text))
            text.extend(_bytes_of(tail))
        result.text = String(unsafe_from_utf8=text)
        result.literal_begin = lit_begin^
        result.literal_end = lit_end^
        return result^
