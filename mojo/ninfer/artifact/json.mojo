"""Minimal strict JSON reader for the artifact directory.

Mojo's standard library has no JSON module, so this is the Mojo counterpart of the
`nlohmann::json` use in `src/artifact/schema.cpp`. Values live in one arena (`Json.nodes`) and
are addressed by index, which avoids a recursive value type. Duplicate object members are an
error, as in `parse_json` of the C++ reader. Integers that fit `u64` (or negated `i64`) stay
exact; anything else becomes a float, as nlohmann does.
"""

from std.collections import Dict, List

comptime NULL = 0
comptime BOOL = 1
comptime UINT = 2
comptime NEG = 3
comptime FLOAT = 4
comptime STRING = 5
comptime ARRAY = 6
comptime OBJECT = 7


struct JsonNode(Copyable, Movable):
    var kind: Int
    var u: UInt64  # UINT value, NEG magnitude, BOOL (0/1)
    var f: Float64
    var s: String  # STRING value
    var items: List[Int]  # ARRAY elements or OBJECT member values
    var keys: List[String]  # OBJECT member names, parallel to `items`

    def __init__(out self, kind: Int):
        self.kind = kind
        self.u = 0
        self.f = 0.0
        self.s = String()
        self.items = List[Int]()
        self.keys = List[String]()


struct Json(Movable):
    var nodes: List[JsonNode]

    def __init__(out self):
        self.nodes = List[JsonNode]()

    def add(mut self, var node: JsonNode) -> Int:
        self.nodes.append(node^)
        return len(self.nodes) - 1

    def kind(self, n: Int) -> Int:
        return self.nodes[n].kind

    def is_object(self, n: Int) -> Bool:
        return self.nodes[n].kind == OBJECT

    def is_array(self, n: Int) -> Bool:
        return self.nodes[n].kind == ARRAY

    def is_string(self, n: Int) -> Bool:
        return self.nodes[n].kind == STRING

    def is_null(self, n: Int) -> Bool:
        return self.nodes[n].kind == NULL

    def is_integer(self, n: Int) -> Bool:
        var k = self.nodes[n].kind
        return k == UINT or k == NEG

    def is_unsigned(self, n: Int) -> Bool:
        return self.nodes[n].kind == UINT

    def size(self, n: Int) -> Int:
        return len(self.nodes[n].items)

    def at(self, n: Int, i: Int) -> Int:
        """Array element or object member value `i`."""
        return self.nodes[n].items[i]

    def find(self, n: Int, key: String) -> Int:
        """Member value index of an object, or -1."""
        ref node = self.nodes[n]
        for i in range(len(node.keys)):
            if node.keys[i] == key:
                return node.items[i]
        return -1

    def contains(self, n: Int, key: String) -> Bool:
        return self.find(n, key) >= 0

    def get(self, n: Int, key: String) raises -> Int:
        var r = self.find(n, key)
        if r < 0:
            raise Error("missing member " + key)
        return r

    def string(self, n: Int) -> String:
        return self.nodes[n].s

    def key(self, n: Int, i: Int) -> String:
        return self.nodes[n].keys[i]


struct _Parser(Movable):
    var text: List[UInt8]
    var pos: Int
    var label: String
    var strict_keys: Bool  # False: a repeated member replaces the earlier one (nlohmann)

    def __init__(out self, var text: List[UInt8], label: String, strict_keys: Bool):
        self.text = text^
        self.pos = 0
        self.label = label
        self.strict_keys = strict_keys

    def fail(self, what: String) -> Error:
        return Error(self.label + ": " + what + " at byte " + String(self.pos))

    def skip_ws(mut self):
        while self.pos < len(self.text):
            var c = self.text[self.pos]
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                break

    def expect_word(mut self, word: String) raises:
        var b = word.as_bytes()
        if self.pos + len(b) > len(self.text):
            raise self.fail("invalid literal")
        for i in range(len(b)):
            if self.text[self.pos + i] != b[i]:
                raise self.fail("invalid literal")
        self.pos += len(b)

    def hex4(mut self) raises -> Int:
        if self.pos + 4 > len(self.text):
            raise self.fail("truncated \\u escape")
        var v = 0
        for _ in range(4):
            var c = Int(self.text[self.pos])
            var d: Int
            if c >= 0x30 and c <= 0x39:
                d = c - 0x30
            elif c >= 0x61 and c <= 0x66:
                d = c - 0x61 + 10
            elif c >= 0x41 and c <= 0x46:
                d = c - 0x41 + 10
            else:
                raise self.fail("invalid \\u escape")
            v = v * 16 + d
            self.pos += 1
        return v

    def parse_string(mut self) raises -> String:
        # Caller has verified the opening quote.
        self.pos += 1
        var out = List[UInt8]()
        while True:
            if self.pos >= len(self.text):
                raise self.fail("unterminated string")
            var c = self.text[self.pos]
            self.pos += 1
            if c == 0x22:
                break
            if c < 0x20:
                raise self.fail("control character in string")
            if c != 0x5C:
                out.append(c)
                continue
            if self.pos >= len(self.text):
                raise self.fail("unterminated string")
            var e = self.text[self.pos]
            self.pos += 1
            if e == 0x22 or e == 0x5C or e == 0x2F:
                out.append(e)
            elif e == 0x62:
                out.append(0x08)
            elif e == 0x66:
                out.append(0x0C)
            elif e == 0x6E:
                out.append(0x0A)
            elif e == 0x72:
                out.append(0x0D)
            elif e == 0x74:
                out.append(0x09)
            elif e == 0x75:
                var cp = self.hex4()
                if cp >= 0xD800 and cp < 0xDC00:
                    if (
                        self.pos + 2 <= len(self.text)
                        and self.text[self.pos] == 0x5C
                        and self.text[self.pos + 1] == 0x75
                    ):
                        self.pos += 2
                        var lo = self.hex4()
                        if lo < 0xDC00 or lo > 0xDFFF:
                            raise self.fail("invalid surrogate pair")
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    else:
                        raise self.fail("lone surrogate")
                elif cp >= 0xDC00 and cp < 0xE000:
                    raise self.fail("lone surrogate")
                _append_utf8(out, cp)
            else:
                raise self.fail("invalid escape")
        if not _valid_utf8(out):
            raise self.fail("invalid UTF-8 in string")
        return String(unsafe_from_utf8=out)

    def parse_number(mut self, mut doc: Json) raises -> Int:
        var start = self.pos
        var negative = False
        if self.text[self.pos] == 0x2D:
            negative = True
            self.pos += 1
        var digits_start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] >= 0x30 and self.text[self.pos] <= 0x39:
            self.pos += 1
        var ndigits = self.pos - digits_start
        if ndigits == 0:
            raise self.fail("invalid number")
        if ndigits > 1 and self.text[digits_start] == 0x30:
            raise self.fail("leading zero")
        var is_float = False
        if self.pos < len(self.text) and self.text[self.pos] == 0x2E:
            is_float = True
            self.pos += 1
            var f0 = self.pos
            while self.pos < len(self.text) and self.text[self.pos] >= 0x30 and self.text[self.pos] <= 0x39:
                self.pos += 1
            if self.pos == f0:
                raise self.fail("invalid number")
        if self.pos < len(self.text) and (self.text[self.pos] == 0x65 or self.text[self.pos] == 0x45):
            is_float = True
            self.pos += 1
            if self.pos < len(self.text) and (self.text[self.pos] == 0x2B or self.text[self.pos] == 0x2D):
                self.pos += 1
            var e0 = self.pos
            while self.pos < len(self.text) and self.text[self.pos] >= 0x30 and self.text[self.pos] <= 0x39:
                self.pos += 1
            if self.pos == e0:
                raise self.fail("invalid number")
        if not is_float:
            # Exact u64 accumulation with overflow detection; overflow falls through to float.
            var acc = UInt64(0)
            var overflow = False
            for i in range(digits_start, digits_start + ndigits):
                var d = UInt64(self.text[i] - 0x30)
                if acc > (UInt64.MAX - d) // 10:
                    overflow = True
                    break
                acc = acc * 10 + d
            # A negative value must fit i64 (magnitude <= 2^63).
            if not overflow and (not negative or acc <= UInt64(1) << 63):
                var node = JsonNode(NEG if negative else UINT)
                node.u = acc
                return doc.add(node^)
        var span = List[UInt8]()
        for i in range(start, self.pos):
            span.append(self.text[i])
        var node = JsonNode(FLOAT)
        node.f = atof(String(unsafe_from_utf8=span))
        return doc.add(node^)

    def parse_value(mut self, mut doc: Json, depth: Int) raises -> Int:
        if depth > 512:
            raise self.fail("nesting too deep")
        self.skip_ws()
        if self.pos >= len(self.text):
            raise self.fail("unexpected end of input")
        var c = self.text[self.pos]
        if c == 0x7B:  # {
            self.pos += 1
            var node = JsonNode(OBJECT)
            var seen = Dict[String, Int]()
            self.skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == 0x7D:
                self.pos += 1
                return doc.add(node^)
            while True:
                self.skip_ws()
                if self.pos >= len(self.text) or self.text[self.pos] != 0x22:
                    raise self.fail("expected object key")
                var key = self.parse_string()
                var replace_at = -1
                if key in seen:
                    if self.strict_keys:
                        raise Error(self.label + ": duplicate JSON member " + key)
                    replace_at = seen[key]
                else:
                    seen[key] = len(node.keys)
                self.skip_ws()
                if self.pos >= len(self.text) or self.text[self.pos] != 0x3A:
                    raise self.fail("expected ':'")
                self.pos += 1
                var child = self.parse_value(doc, depth + 1)
                if replace_at >= 0:
                    node.items[replace_at] = child
                else:
                    node.keys.append(key^)
                    node.items.append(child)
                self.skip_ws()
                if self.pos >= len(self.text):
                    raise self.fail("unterminated object")
                if self.text[self.pos] == 0x2C:
                    self.pos += 1
                    continue
                if self.text[self.pos] == 0x7D:
                    self.pos += 1
                    break
                raise self.fail("expected ',' or '}'")
            return doc.add(node^)
        if c == 0x5B:  # [
            self.pos += 1
            var node = JsonNode(ARRAY)
            self.skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == 0x5D:
                self.pos += 1
                return doc.add(node^)
            while True:
                var child = self.parse_value(doc, depth + 1)
                node.items.append(child)
                self.skip_ws()
                if self.pos >= len(self.text):
                    raise self.fail("unterminated array")
                if self.text[self.pos] == 0x2C:
                    self.pos += 1
                    continue
                if self.text[self.pos] == 0x5D:
                    self.pos += 1
                    break
                raise self.fail("expected ',' or ']'")
            return doc.add(node^)
        if c == 0x22:
            var node = JsonNode(STRING)
            node.s = self.parse_string()
            return doc.add(node^)
        if c == 0x74:
            self.expect_word("true")
            var node = JsonNode(BOOL)
            node.u = 1
            return doc.add(node^)
        if c == 0x66:
            self.expect_word("false")
            return doc.add(JsonNode(BOOL))
        if c == 0x6E:
            self.expect_word("null")
            return doc.add(JsonNode(NULL))
        if c == 0x2D or (c >= 0x30 and c <= 0x39):
            return self.parse_number(doc)
        raise self.fail("unexpected character")


def _valid_utf8(b: List[UInt8]) -> Bool:
    """The standard library's validation, which agrees with `utf8proc_iterate` (nlohmann applies
    those rules to every string)."""
    try:
        _ = String(from_utf8=b)
        return True
    except:
        return False


def _append_utf8(mut out: List[UInt8], cp: Int):
    for b in chr(cp).as_bytes():
        out.append(b)


def parse_json(
    var text: List[UInt8], label: String, mut doc: Json, strict_keys: Bool = True
) raises -> Int:
    """Parse `text` into `doc` and return the root node index.

    `strict_keys=False` keeps the last of a repeated member, as nlohmann does for
    `Json::parse` (the artifact directory is the strict case)."""
    var p = _Parser(text^, label, strict_keys)
    var root = p.parse_value(doc, 0)
    p.skip_ws()
    if p.pos != len(p.text):
        raise p.fail("trailing characters")
    return root
