"""`JinjaTemplate`: parse once, render with provenance. Mirrors `src/text/jinja.{h,cpp}`."""

from std.collections import List

from ninfer.artifact.json import (
    Json, NULL, BOOL, UINT, NEG, FLOAT, STRING, ARRAY, OBJECT,
)
from ninfer.text.jinja.jstring import JString, string_bytes
from ninfer.text.jinja.lexer import tokenize
from ninfer.text.jinja.parser import Ast, parse_from_tokens
from ninfer.text.jinja.runtime import Runtime, clean_error
from ninfer.text.jinja.value import Heap, V_STRING, obj_insert, obj_insert_str
from ninfer.text.unicode import utf8_codepoints


struct InputRegion(Copyable, Movable):
    var pointer: String
    var tag: Int

    def __init__(out self, pointer: String, tag: Int):
        self.pointer = pointer
        self.tag = tag


struct OutputRegion(Copyable, Movable):
    var tag: Int
    var begin: Int
    var end: Int
    var source_offset: Int  # -1: absent

    def __init__(out self, tag: Int, begin: Int, end: Int, source_offset: Int):
        self.tag = tag
        self.begin = begin
        self.end = end
        self.source_offset = source_offset


struct TemplateOutput(Movable):
    var text: String
    var literal_begin: List[Int]
    var literal_end: List[Int]
    var regions: List[OutputRegion]

    def __init__(out self):
        self.text = String()
        self.literal_begin = List[Int]()
        self.literal_end = List[Int]()
        self.regions = List[OutputRegion]()


def _pointer_component(value: String) -> String:
    return value.replace("~", "~0").replace("/", "~1")


struct JinjaTemplate(Movable):
    var source_name: String
    var source: List[UInt8]  # normalized by the lexer, as in C++
    var original: List[UInt8]
    var ast: Ast

    def __init__(out self, source: String, source_name: String) raises:
        self.source_name = source_name
        self.original = string_bytes(source)
        self.source = List[UInt8]()
        self.ast = Ast()
        try:
            _ = utf8_codepoints(self.original, "chat template")
            var lexed = tokenize(self.original)
            self.source = lexed.source.copy()
            self.ast = parse_from_tokens(lexed)
        except e:
            raise Error(source_name + ": " + clean_error(String(e)))

    def render(
        self,
        doc: Json,
        root: Int,
        regions: List[InputRegion],
        control_variables: List[String],
        timestamp: Int,
    ) raises -> TemplateOutput:
        if not doc.is_object(root):
            raise Error("Template context must be an object")
        try:
            return self._render(doc, root, regions, control_variables, timestamp)
        except e:
            raise Error(self.source_name + ": " + clean_error(String(e)))

    def _render(
        self,
        doc: Json,
        root: Int,
        regions: List[InputRegion],
        control_variables: List[String],
        timestamp: Int,
    ) raises -> TemplateOutput:
        # `context(impl_->source)`: the normalized source drives error excerpts.
        var rt = Runtime(self.ast.copy(), self.source.copy())
        var env = rt.new_env()
        for i in range(doc.size(root)):
            var key = doc.key(root, i)
            var literal = True
            for c in control_variables:
                if c == key:
                    literal = False
                    break
            var v = _convert(rt.h, doc, doc.at(root, i), "/" + _pointer_component(key), regions, literal)
            rt.set_val(env, key, v)
        var rendered = rt.execute_program(env, timestamp)
        var out = TemplateOutput()
        for part in rt.h.nodes[rendered].s.parts:
            var begin = out.text.byte_length()
            out.text += part.val
            if part.literal and part.val.byte_length() > 0:
                if len(out.literal_end) > 0 and out.literal_end[len(out.literal_end) - 1] == begin:
                    out.literal_end[len(out.literal_end) - 1] = out.text.byte_length()
                else:
                    out.literal_begin.append(begin)
                    out.literal_end.append(out.text.byte_length())
            if part.origin != 0:
                out.regions.append(OutputRegion(part.origin, begin, out.text.byte_length(), part.source_offset))
        return out^


def _convert(
    mut h: Heap, doc: Json, n: Int, pointer: String, regions: List[InputRegion], literal: Bool
) raises -> Int:
    var k = doc.kind(n)
    var result = -1
    if k == NULL:
        result = h.none()
    elif k == BOOL:
        result = h.boolean(doc.nodes[n].u != 0)
    elif k == FLOAT:
        result = h.float(doc.nodes[n].f)
    elif k == UINT:
        var number = doc.nodes[n].u
        if number > 9223372036854775807:
            raise Error("Template integer exceeds int64")
        result = h.integer(Int(number))
    elif k == NEG:
        result = h.integer(-Int(doc.nodes[n].u))
    elif k == STRING:
        result = h.string(JString(doc.string(n), 0, literal))
    elif k == ARRAY:
        result = h.array()
        for i in range(doc.size(n)):
            var item = _convert(h, doc, doc.at(n, i), pointer + "/" + String(i), regions, literal)
            h.nodes[result].arr.append(item)
    elif k == OBJECT:
        result = h.object()
        for i in range(doc.size(n)):
            var key = doc.key(n, i)
            var kv = h.string(JString(key, 0, literal))
            var vv = _convert(h, doc, doc.at(n, i), pointer + "/" + _pointer_component(key), regions, literal)
            obj_insert(h, result, kv, vv)
    else:
        raise Error("Unsupported template JSON value")
    for r in regions:
        if r.pointer == pointer:
            h.nodes[result].origin = r.tag
            if h.nodes[result].kind == V_STRING:
                h.nodes[result].s.tag(r.tag)
            break
    return result
