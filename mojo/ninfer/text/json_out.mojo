"""A small language for building JSON, so no JSON text is ever assembled by hand.

    var j = JsonOut()
    var call = j.typed("tag")                       # {"type":"tag"}
    j.put(call, "begin", j.s("<parameter=" + name + ">"))
    j.put(call, "content", j.raw(already_valid_json))
    var items = j.arr()
    j.add(items, call)
    print(j.text(items))

Values are indices into one arena (`Json`), as in the reader, so a value can be used in several
places and is written wherever it is referenced. `text` writes what `nlohmann::json::dump()` writes:
compact, members in insertion order, strings escaped by `json_escape`, doubles by Grisu2.
"""

from std.collections import List

from ninfer.artifact.json import (
    Json,
    JsonNode,
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
from ninfer.text.jinja.value import json_escape


def dump_json(doc: Json, n: Int) raises -> String:
    """`nlohmann::json::dump()` of value `n` of `doc`. A STRING node whose `u` is 1 is raw JSON text
    (`JsonOut.raw`) and is written as it is."""
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
        if doc.nodes[n].u == 1:
            return doc.string(n)
        return json_escape(doc.string(n), False)
    var out = String()
    if kind == ARRAY:
        out += "["
        for i in range(doc.size(n)):
            if i > 0:
                out += ","
            out += dump_json(doc, doc.at(n, i))
        return out + "]"
    out += "{"
    for i in range(doc.size(n)):
        if i > 0:
            out += ","
        out += json_escape(doc.key(n, i), False) + ":" + dump_json(doc, doc.at(n, i))
    return out + "}"


struct JsonOut(Movable):
    var doc: Json

    def __init__(out self):
        self.doc = Json()

    # --- values ---------------------------------------------------------------------------

    def s(mut self, text: String) -> Int:
        var node = JsonNode(STRING)
        node.s = text
        return self.doc.add(node^)

    def raw(mut self, json_text: String) -> Int:
        """JSON text the caller has already checked; written verbatim."""
        var node = JsonNode(STRING)
        node.s = json_text
        node.u = 1
        return self.doc.add(node^)

    def u(mut self, value: Int) -> Int:
        var node = JsonNode(UINT if value >= 0 else NEG)
        node.u = UInt64(value if value >= 0 else -value)
        return self.doc.add(node^)

    def b(mut self, value: Bool) -> Int:
        var node = JsonNode(BOOL)
        node.u = 1 if value else 0
        return self.doc.add(node^)

    def null(mut self) -> Int:
        return self.doc.add(JsonNode(NULL))

    def f(mut self, value: Float64) -> Int:
        var node = JsonNode(FLOAT)
        node.f = value
        return self.doc.add(node^)

    # --- containers -----------------------------------------------------------------------

    def obj(mut self) -> Int:
        return self.doc.add(JsonNode(OBJECT))

    def typed(mut self, kind: String) -> Int:
        """`{"type": kind}`, the opening of every node of the grammar the constrained decoder reads."""
        var o = self.obj()
        self.put(o, "type", self.s(kind))
        return o

    def arr(mut self) -> Int:
        return self.doc.add(JsonNode(ARRAY))

    def put(mut self, object: Int, key: String, value: Int):
        """Sets a member; an existing key keeps its place and takes the new value."""
        ref node = self.doc.nodes[object]
        for i in range(len(node.keys)):
            if node.keys[i] == key:
                node.items[i] = value
                return
        node.keys.append(key)
        node.items.append(value)

    def add(mut self, array: Int, value: Int):
        self.doc.nodes[array].items.append(value)

    def strings(mut self, values: List[String]) -> Int:
        var a = self.arr()
        for v in values:
            self.add(a, self.s(v))
        return a

    # --- output ---------------------------------------------------------------------------

    def text(self, value: Int) raises -> String:
        return dump_json(self.doc, value)
