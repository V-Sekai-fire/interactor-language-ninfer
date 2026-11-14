"""Jinja values, builtins and runtime; mirrors `third_party/llama-jinja/jinja/{value,runtime,json}.*`.

Values live in a `Heap` and are addressed by index, which gives the shared, mutable objects
Jinja needs (`namespace()` attributes, `list.append`) without reference counting. A render
owns one heap; nothing is freed until it ends, the same lifetime a C++ render gives them.

Exceptions: every function here raises `JinjaError` (error.mojo), whose `kind` stands for the
C++ exception class. `K_BREAK` and `K_CONTINUE` are loop control, `K_RETHROWN` is a message that
already carries its source context, `K_NOTIMPL` is a NotImplemented error that `execute` must
not decorate. A `K_ERROR` is an ordinary `std::exception` and gets decorated by the nearest
`execute`, exactly as in C++.
"""

from ninfer.text.jinja.error import JinjaError, codepoints, jinja_error, parse_int, K_NOTIMPL
from std.collections import Dict, List, Optional

from ninfer.text.jinja.jstring import (
    JU,
    JString,
    Character,
    CASE_TITLE,
    characters,
    codepoint_length,
    slice_indices,
    string_bytes,
    sub_string,
)
from ninfer.text.jinja.lexer import peak_source
from ninfer.text.jinja.parser import (
    Ast,
    Node,
    type_name,
    N_OUTPUT, N_IF, N_FOR, N_BREAK, N_CONTINUE, N_NOOP, N_SET, N_MACRO, N_COMMENT, N_BLANK,
    N_MEMBER, N_CALL, N_IDENT, N_INT, N_FLOAT, N_STRING, N_ARRAY, N_TUPLE, N_OBJECT, N_BINARY,
    N_FILTER, N_FILTER_STMT, N_SELECT, N_TEST, N_UNARY, N_SLICE, N_KWARG, N_SPREAD, N_CALL_STMT,
    N_TERNARY,
)


comptime V_NONE = 0
comptime V_UNDEF = 1
comptime V_BOOL = 2
comptime V_INT = 3
comptime V_FLOAT = 4
comptime V_STRING = 5
comptime V_ARRAY = 6
comptime V_TUPLE = 7
comptime V_OBJECT = 8
comptime V_FUNC = 9
comptime V_KWARG = 10

# function kinds
comptime F_GLOBAL = 0  # global builtin (no receiver)
comptime F_METHOD = 1  # builtin bound to a receiver
comptime F_MACRO = 2
comptime F_CALLER = 3


def is_numeric_kind(k: Int) -> Bool:
    return k == V_INT or k == V_FLOAT or k == V_BOOL


def is_array_kind(k: Int) -> Bool:
    return k == V_ARRAY or k == V_TUPLE


struct VNode(Copyable, Movable):
    var kind: Int
    var i: Int
    var f: Float64
    var s: JString
    var arr: List[Int]
    var keys: List[Int]
    var vals: List[Int]
    var origin: Int
    var hint: String  # V_UNDEF
    var name: String  # V_FUNC name, V_KWARG key
    var fkind: Int
    var bound: Int  # receiver / arg0 or -1
    var node: Int  # macro statement or call statement node
    var scope: Int  # defining scope (object id)
    var has_builtins: Bool  # V_OBJECT
    var kwval: Int  # V_KWARG value

    def __init__(out self, kind: Int):
        self.kind = kind
        self.i = 0
        self.f = 0.0
        self.s = JString()
        self.arr = List[Int]()
        self.keys = List[Int]()
        self.vals = List[Int]()
        self.origin = 0
        self.hint = String()
        self.name = String()
        self.fkind = 0
        self.bound = -1
        self.node = -1
        self.scope = -1
        self.has_builtins = True
        self.kwval = -1


struct Heap(Movable):
    var nodes: List[VNode]
    var ju: JU
    var time: Int

    def __init__(out self, var ju: JU):
        self.nodes = List[VNode]()
        self.ju = ju^
        self.time = 0

    def new(mut self, kind: Int) -> Int:
        self.nodes.append(VNode(kind))
        return len(self.nodes) - 1

    def none(mut self) -> Int:
        return self.new(V_NONE)

    def undef(mut self, hint: String = "") -> Int:
        var v = self.new(V_UNDEF)
        self.nodes[v].hint = hint
        return v

    def boolean(mut self, b: Bool) -> Int:
        var v = self.new(V_BOOL)
        self.nodes[v].i = 1 if b else 0
        self.nodes[v].f = 1.0 if b else 0.0
        return v

    def integer(mut self, i: Int) -> Int:
        var v = self.new(V_INT)
        self.nodes[v].i = i
        var f = Float64(i)
        if i == 9223372036854775807:
            f = Float64.MAX * 2.0  # +inf: the round trip through double does not return i
        self.nodes[v].f = f
        return v

    def float(mut self, f: Float64) -> Int:
        var v = self.new(V_FLOAT)
        self.nodes[v].f = f
        self.nodes[v].i = _f2i(f)
        return v

    def string(mut self, var s: JString) -> Int:
        var v = self.new(V_STRING)
        self.nodes[v].s = s^
        return v

    def text(mut self, var s: String) -> Int:
        return self.string(JString(s^))

    def array(mut self) -> Int:
        return self.new(V_ARRAY)

    def tuple(mut self, items: List[Int]) -> Int:
        var v = self.new(V_TUPLE)
        self.nodes[v].arr = items.copy()
        return v

    def object(mut self) -> Int:
        return self.new(V_OBJECT)

    def kwarg(mut self, key: String, val: Int) -> Int:
        var v = self.new(V_KWARG)
        self.nodes[v].name = key
        self.nodes[v].kwval = val
        return v

    def func(mut self, name: String, fkind: Int, bound: Int = -1) -> Int:
        var v = self.new(V_FUNC)
        self.nodes[v].name = name
        self.nodes[v].fkind = fkind
        self.nodes[v].bound = bound
        return v


def _f2i(f: Float64) -> Int:
    if not _is_finite(f):
        return 0
    if f >= 9.2233720368547758e18:
        return 9223372036854775807
    if f <= -9.2233720368547758e18:
        return -9223372036854775807 - 1
    return Int(f)


def _is_finite(f: Float64) -> Bool:
    return f == f and f - f == 0.0


def _is_nan(f: Float64) -> Bool:
    return f != f


# --- type names and basic predicates ---------------------------------------------------------


def vtype(h: Heap, v: Int) -> String:
    var k = h.nodes[v].kind
    if k == V_NONE:
        return "None"
    if k == V_UNDEF:
        var hint = h.nodes[v].hint
        return "Undefined" if hint.byte_length() == 0 else "Undefined (hint: '" + hint + "')"
    if k == V_BOOL:
        return "Boolean"
    if k == V_INT:
        return "Integer"
    if k == V_FLOAT:
        return "Float"
    if k == V_STRING:
        return "String"
    if k == V_ARRAY:
        return "Array"
    if k == V_TUPLE:
        return "Tuple"
    if k == V_OBJECT:
        return "Object"
    if k == V_FUNC:
        return "Function"
    return "KwArg"


def type_error(h: Heap, v: Int, what: String) -> JinjaError:
    return jinja_error(vtype(h, v) + " " + what)


def is_immutable(h: Heap, v: Int) -> Bool:
    var k = h.nodes[v].kind
    return not (k == V_ARRAY or k == V_OBJECT)


def is_hashable(h: Heap, v: Int) -> Bool:
    var k = h.nodes[v].kind
    if k == V_NONE or k == V_BOOL or k == V_INT or k == V_FLOAT or k == V_STRING or k == V_KWARG:
        return True
    if k == V_ARRAY or k == V_TUPLE:
        for item in h.nodes[v].arr:
            if not (is_immutable(h, item) and is_hashable(h, item)):
                return False
        return True
    if k == V_OBJECT:
        for item in h.nodes[v].vals:
            if not (is_immutable(h, item) and is_hashable(h, item)):
                return False
        return True
    return False


def equivalent(h: Heap, a: Int, b: Int) -> Bool:
    """`*a == *b` (value_t::operator==)."""
    var ka = h.nodes[a].kind
    var kb = h.nodes[b].kind
    if ka == V_UNDEF:
        return (kb == V_UNDEF)  # is_undefined() == other.is_undefined() with this undefined
    if ka == V_NONE:
        return kb == V_NONE
    if ka == V_BOOL or ka == V_INT or ka == V_FLOAT:
        return is_numeric_kind(kb) and h.nodes[a].i == h.nodes[b].i and h.nodes[a].f == h.nodes[b].f
    if ka == V_STRING:
        return kb == V_STRING and h.nodes[a].s.str() == h.nodes[b].s.str()
    if ka == V_ARRAY or ka == V_TUPLE:
        if kb != ka or not is_hashable(h, a) or not is_hashable(h, b):
            return False
        if len(h.nodes[a].arr) != len(h.nodes[b].arr):
            return False
        for i in range(len(h.nodes[a].arr)):
            if not equivalent(h, h.nodes[a].arr[i], h.nodes[b].arr[i]):
                return False
        return True
    if ka == V_OBJECT:
        if kb != V_OBJECT or not is_hashable(h, a) or not is_hashable(h, b):
            return False
        if len(h.nodes[a].keys) != len(h.nodes[b].keys):
            return False
        for i in range(len(h.nodes[a].keys)):
            if not equivalent(h, h.nodes[a].keys[i], h.nodes[b].keys[i]):
                return False
            if not equivalent(h, h.nodes[a].vals[i], h.nodes[b].vals[i]):
                return False
        return True
    if ka == V_KWARG:
        return kb == V_KWARG and h.nodes[a].name == h.nodes[b].name and h.nodes[a].kwval == h.nodes[b].kwval
    # functions: no support as keys; compare by identity of the callable description
    return ka == kb and h.nodes[a].name == h.nodes[b].name and h.nodes[a].fkind == h.nodes[b].fkind and h.nodes[a].node == h.nodes[b].node


# --- numbers to text -------------------------------------------------------------------------


def format_float(f: Float64) raises JinjaError -> String:
    """nlohmann::json(double).dump(): shortest round-trip digits, its own layout rules."""
    if f == 0.0:
        # sign of zero
        var negative = String(f).startswith("-")
        return "-0.0" if negative else "0.0"
    var text = String(f)
    var negative = text.startswith("-")
    if negative:
        var t2 = String(text[byte=1:])
        text = t2^
    # split mantissa / exponent
    var exp10 = 0
    var mantissa = text
    var epos = text.find("e")
    if epos >= 0:
        mantissa = String(text[byte=0:epos])
        exp10 = parse_int(String(text[byte = epos + 1 :]))
    var digits = String()
    var point = mantissa.find(".")
    var int_part = mantissa if point < 0 else String(mantissa[byte=0:point])
    var frac_part = String() if point < 0 else String(mantissa[byte = point + 1 :])
    digits = int_part + frac_part
    var n = int_part.byte_length() + exp10  # decimal point position relative to the digits
    var lead = 0
    var end = digits.byte_length()
    while lead < end - 1 and digits.as_bytes()[lead] == 0x30:
        lead += 1
        n -= 1
    while end - lead > 1 and digits.as_bytes()[end - 1] == 0x30:
        end -= 1
    var d2 = String(digits[byte=lead:end])
    digits = d2^
    var k = digits.byte_length()
    var out = String()
    if k <= n and n <= 15:
        out = digits + "0" * (n - k) + ".0"
    elif 0 < n and n <= 15:
        out = String(digits[byte=0:n]) + "." + String(digits[byte=n:])
    elif -4 < n and n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        var e = n - 1
        out = String(digits[byte=0:1])
        if k > 1:
            out += "." + String(digits[byte=1:])
        out += "e"
        out += "-" if e < 0 else "+"
        var ae = -e if e < 0 else e
        out += ("0" + String(ae)) if ae < 10 else String(ae)
    return ("-" + out) if negative else out


def float_to_string(f: Float64) raises JinjaError -> String:
    if _is_nan(f):
        return "nan"
    if not _is_finite(f):
        return "-inf" if f < 0.0 else "inf"
    return format_float(f)


# --- JSON string escaping (nlohmann::json(string).dump(-1, ' ', ensure_ascii)) ---------------


def json_escape(s: String, ensure_ascii: Bool) raises JinjaError -> String:
    var out = String()
    var bytes = string_bytes(s)
    var spans = utf8_spans(bytes)
    out += '"'
    for sp in spans:
        var cp = sp.value
        if cp == 0x22:
            out += '\\"'
        elif cp == 0x5C:
            out += "\\\\"
        elif cp == 0x08:
            out += "\\b"
        elif cp == 0x0C:
            out += "\\f"
        elif cp == 0x0A:
            out += "\\n"
        elif cp == 0x0D:
            out += "\\r"
        elif cp == 0x09:
            out += "\\t"
        elif cp < 0x20:
            out += "\\u" + _hex4(cp)
        elif cp < 0x80 or not ensure_ascii:
            out += String(chr(cp))
        elif cp < 0x10000:
            out += "\\u" + _hex4(cp)
        else:
            var v = cp - 0x10000
            out += "\\u" + _hex4(0xD800 + (v >> 10)) + "\\u" + _hex4(0xDC00 + (v & 0x3FF))
    out += '"'
    return out^


def _hex4(v: Int) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for shift in [12, 8, 4, 0]:
        var d = (v >> shift) & 15
        out += digits[byte=d : d + 1]
    return out^


from ninfer.text.unicode import utf8_codepoints, CodepointSpan


def utf8_spans(bytes: List[UInt8]) raises JinjaError -> List[CodepointSpan]:
    return codepoints(bytes, "Jinja string")


# --- as_string / as_bool / repr --------------------------------------------------------------


def as_string(mut h: Heap, v: Int) raises JinjaError -> JString:
    var k = h.nodes[v].kind
    if k == V_STRING:
        return h.nodes[v].s.copy()
    if k == V_INT:
        return JString(String(h.nodes[v].i))
    if k == V_FLOAT:
        return JString(float_to_string(h.nodes[v].f))
    if k == V_BOOL:
        return JString("True" if h.nodes[v].i != 0 else "False")
    if k == V_NONE:
        return JString("None")
    if k == V_UNDEF:
        return JString()
    if k == V_ARRAY or k == V_TUPLE:
        var immutable = k == V_TUPLE
        var out = "(" if immutable else "["
        var items = h.nodes[v].arr.copy()
        for i in range(len(items)):
            if i > 0:
                out += ", "
            out += to_string_repr(h, items[i])
        if immutable and len(items) == 1:
            out += ","
        out += ")" if immutable else "]"
        return JString(out^, 0, True)
    if k == V_OBJECT:
        var out = String("{")
        var keys = h.nodes[v].keys.copy()
        var vals = h.nodes[v].vals.copy()
        for i in range(len(keys)):
            if i > 0:
                out += ", "
            out += to_string_repr(h, keys[i]) + ": " + to_string_repr(h, vals[i])
        out += "}"
        return JString(out^, 0, True)
    raise type_error(h, v, "is not a string value")


def as_repr(mut h: Heap, v: Int) raises JinjaError -> String:
    var k = h.nodes[v].kind
    if k == V_NONE:
        return "None"
    if k == V_UNDEF:
        return vtype(h, v)
    if k == V_FUNC:
        var bound = h.nodes[v].bound
        return "Function<" + h.nodes[v].name + ">(" + (as_repr(h, bound) if bound >= 0 else String()) + ")"
    if k == V_KWARG:
        return "KwArg"
    return as_string(h, v).str()


def to_string_repr(mut h: Heap, v: Int) raises JinjaError -> String:
    if h.nodes[v].kind == V_STRING:
        var s = h.nodes[v].s.str()
        if s.find("'") >= 0:
            return value_to_json(h, v)
        return "'" + s + "'"
    return as_repr(h, v)


def as_bool(mut h: Heap, v: Int) raises JinjaError -> Bool:
    var k = h.nodes[v].kind
    if k == V_INT or k == V_BOOL:
        return h.nodes[v].i != 0
    if k == V_FLOAT:
        return h.nodes[v].f != 0.0
    if k == V_STRING:
        return h.nodes[v].s.byte_size() > 0
    if k == V_ARRAY or k == V_TUPLE:
        return len(h.nodes[v].arr) > 0
    if k == V_OBJECT:
        return len(h.nodes[v].keys) > 0
    if k == V_NONE or k == V_UNDEF:
        return False
    raise type_error(h, v, "is not a bool value")


def as_int(h: Heap, v: Int) raises JinjaError -> Int:
    var k = h.nodes[v].kind
    if k == V_INT or k == V_FLOAT or k == V_BOOL:
        return h.nodes[v].i
    raise type_error(h, v, "is not an int value")


def as_float(h: Heap, v: Int) raises JinjaError -> Float64:
    var k = h.nodes[v].kind
    if k == V_INT or k == V_FLOAT or k == V_BOOL:
        return h.nodes[v].f
    raise type_error(h, v, "is not a float value")


def as_array(h: Heap, v: Int) raises JinjaError -> List[Int]:
    var k = h.nodes[v].kind
    if k == V_ARRAY or k == V_TUPLE:
        return h.nodes[v].arr.copy()
    raise type_error(h, v, "is not an array value")


def as_ordered_object_check(h: Heap, v: Int) raises JinjaError:
    if h.nodes[v].kind != V_OBJECT:
        raise type_error(h, v, "is not an object value")


# --- objects ----------------------------------------------------------------------------------


def obj_find(h: Heap, obj: Int, key: Int) raises JinjaError -> Int:
    """Index of `key` in the object, or -1; raises for an unhashable key like has_key."""
    if not is_immutable(h, key) or not is_hashable(h, key):
        raise jinja_error("Object key of unhashable type: " + vtype(h, key))
    var keys = h.nodes[obj].keys.copy()
    for i in range(len(keys)):
        if equivalent(h, keys[i], key):
            return i
    return -1


def obj_insert(mut h: Heap, obj: Int, key: Int, val: Int) raises JinjaError:
    if h.nodes[obj].kind != V_OBJECT:
        raise type_error(h, obj, "is not an object value")
    var at = obj_find(h, obj, key)
    if at >= 0:
        h.nodes[obj].vals[at] = val
        return
    h.nodes[obj].keys.append(key)
    h.nodes[obj].vals.append(val)


def obj_insert_str(mut h: Heap, obj: Int, key: String, val: Int) raises JinjaError:
    var k = h.text(key)
    obj_insert(h, obj, k, val)


def obj_at(mut h: Heap, obj: Int, key: Int, default: Int) raises JinjaError -> Int:
    var at = obj_find(h, obj, key)
    if at < 0:
        return default
    return h.nodes[obj].vals[at]


def obj_at_strict(mut h: Heap, obj: Int, key: Int) raises JinjaError -> Int:
    var at = obj_find(h, obj, key)
    if at < 0:
        raise jinja_error("Key '" + as_string(h, key).str() + "' not found in value of type " + vtype(h, obj))
    return h.nodes[obj].vals[at]


# --- comparison --------------------------------------------------------------------------------

comptime OP_EQ = 0
comptime OP_GE = 1
comptime OP_GT = 2
comptime OP_LT = 3
comptime OP_NE = 4


def raised(msg: String) -> JinjaError:
    return jinja_error(msg)


def value_compare(mut h: Heap, a: Int, b: Int, op: Int) raises JinjaError -> Bool:
    if op == OP_EQ:
        return equivalent(h, a, b)
    if op == OP_NE:
        return not equivalent(h, a, b)
    var ka = h.nodes[a].kind
    var kb = h.nodes[b].kind
    if is_numeric_kind(ka) and is_numeric_kind(kb):
        var x = h.nodes[a].f
        var y = h.nodes[b].f
        if op == OP_GE:
            return x >= y
        if op == OP_GT:
            return x > y
        return x < y
    if ka == V_STRING and kb == V_STRING:
        var x = h.nodes[a].s.str()
        var y = h.nodes[b].s.str()
        var c = _compare_bytes(x, y)
        if op == OP_GE:
            return c >= 0
        if op == OP_GT:
            return c > 0
        return c < 0
    raise jinja_error("Cannot order " + vtype(h, a) + " and " + vtype(h, b))


def _compare_bytes(x: String, y: String) -> Int:
    var xb = x.as_bytes()
    var yb = y.as_bytes()
    var n = min(len(xb), len(yb))
    for i in range(n):
        if xb[i] != yb[i]:
            return -1 if xb[i] < yb[i] else 1
    if len(xb) == len(yb):
        return 0
    return -1 if len(xb) < len(yb) else 1


# --- JSON serialization -----------------------------------------------------------------------


def value_to_json(
    mut h: Heap,
    v: Int,
    indent: Int = -1,
    item_sep: String = ", ",
    key_sep: String = ": ",
    ensure_ascii: Bool = False,
    sort_keys: Bool = False,
) raises JinjaError -> String:
    var out = String()
    var ancestors = List[Int]()
    _write_json(h, out, v, 0, indent, item_sep, key_sep, ensure_ascii, sort_keys, ancestors)
    return out^


def _write_json(
    mut h: Heap,
    mut out: String,
    v: Int,
    level: Int,
    indent: Int,
    item_sep: String,
    key_sep: String,
    ensure_ascii: Bool,
    sort_keys: Bool,
    mut ancestors: List[Int],
) raises JinjaError:
    var k = h.nodes[v].kind
    if k == V_NONE:
        out += "null"
    elif k == V_BOOL:
        out += "true" if h.nodes[v].i != 0 else "false"
    elif k == V_INT:
        out += String(h.nodes[v].i)
    elif k == V_FLOAT:
        var number = h.nodes[v].f
        if _is_nan(number):
            out += "NaN"
        elif not _is_finite(number):
            out += "-Infinity" if number < 0.0 else "Infinity"
        else:
            out += format_float(number)
    elif k == V_STRING:
        out += json_escape(h.nodes[v].s.str(), ensure_ascii)
    elif k == V_ARRAY or k == V_TUPLE or k == V_OBJECT:
        for a in ancestors:
            if a == v:
                raise jinja_error("Circular value cannot be serialized to JSON")
        ancestors.append(v)
        var is_obj = k == V_OBJECT
        var keys = h.nodes[v].keys.copy()
        var vals = h.nodes[v].vals.copy()
        var items = h.nodes[v].arr.copy()
        if is_obj and sort_keys:
            # stable insertion sort by key
            for i in range(1, len(keys)):
                var j = i
                while j > 0 and value_compare(h, keys[j], keys[j - 1], OP_LT):
                    var tk = keys[j]
                    keys[j] = keys[j - 1]
                    keys[j - 1] = tk
                    var tv = vals[j]
                    vals[j] = vals[j - 1]
                    vals[j - 1] = tv
                    j -= 1
        var size = len(keys) if is_obj else len(items)
        out += "{" if is_obj else "["
        for i in range(size):
            if i > 0:
                out += item_sep
            if indent >= 0:
                out += "\n" + " " * ((level + 1) * indent)
            if is_obj:
                var key = keys[i]
                var name = String()
                var kk = h.nodes[key].kind
                if kk == V_STRING:
                    name = h.nodes[key].s.str()
                elif kk == V_BOOL:
                    name = "true" if h.nodes[key].i != 0 else "false"
                elif kk == V_NONE:
                    name = "null"
                elif is_numeric_kind(kk):
                    name = as_string(h, key).str()
                else:
                    raise jinja_error("Invalid JSON object key: " + vtype(h, key))
                out += json_escape(name, ensure_ascii) + key_sep
            _write_json(h, out, vals[i] if is_obj else items[i], level + 1, indent, item_sep, key_sep, ensure_ascii, sort_keys, ancestors)
        if size > 0 and indent >= 0:
            out += "\n" + " " * (level * indent)
        out += "}" if is_obj else "]"
        _ = ancestors.pop()
    else:
        raise jinja_error("Cannot serialize " + vtype(h, v) + " to JSON")


# --- function arguments -----------------------------------------------------------------------


struct Ctx(Copyable, Movable):
    """A render scope: the environment object and the clock."""

    var env: Int
    var time: Int

    def __init__(out self, env: Int, time: Int):
        self.env = env
        self.time = time


struct FuncArgs(Copyable, Movable):
    var func_name: String
    var ctx: Ctx
    var args: List[Int]

    def __init__(out self, ctx: Ctx):
        self.func_name = String()
        self.ctx = ctx.copy()
        self.args = List[Int]()

    def count(self) -> Int:
        return len(self.args)

    def push_back(mut self, v: Int):
        self.args.append(v)

    def push_front(mut self, v: Int):
        self.args.insert(0, v)

    def get_pos(self, pos: Int) raises JinjaError -> Int:
        if len(self.args) > pos:
            return self.args[pos]
        raise jinja_error(
            "Function '" + self.func_name + "' expected at least " + String(pos + 1)
            + " arguments, got " + String(len(self.args))
        )

    def get_pos(self, pos: Int, default: Int) -> Int:
        if len(self.args) > pos:
            return self.args[pos]
        return default

    def get_kwarg(self, h: Heap, key: String, default: Int) -> Int:
        for a in self.args:
            if h.nodes[a].kind == V_KWARG and h.nodes[a].name == key:
                return h.nodes[a].kwval
        return default

    def get_kwarg_or_pos(self, mut h: Heap, key: String, pos: Int) -> Int:
        var val = self.get_kwarg(h, key, h.undef())
        if h.nodes[val].kind == V_UNDEF and pos < len(self.args) and h.nodes[self.args[pos]].kind != V_KWARG:
            return self.args[pos]
        return val

    def ensure_count(self, min_count: Int, max_count: Int = 999) raises JinjaError:
        var n = len(self.args)
        if n < min_count or n > max_count:
            raise jinja_error(
                "Function '" + self.func_name + "' expected between " + String(min_count)
                + " and " + String(max_count) + " arguments, got " + String(n)
            )

    def ensure_kind(self, h: Heap, v: Int, kinds: List[Int], type_label: String) raises JinjaError:
        for k in kinds:
            if h.nodes[v].kind == k:
                return
        raise jinja_error(
            "Function '" + self.func_name + "' expected value of type " + type_label
            + ", got " + vtype(h, v)
        )

    def ensure_vals(
        self, h: Heap, labels: List[String], kinds: List[List[Int]], required: List[Bool]
    ) raises JinjaError:
        var need = 0
        for r in required:
            if r:
                need += 1
        self.ensure_count(need)
        for i in range(len(required)):
            if required[i] and len(self.args) > i:
                self.ensure_kind(h, self.args[i], kinds[i], labels[i])


def _kinds(*ks: Int) -> List[Int]:
    var out = List[Int]()
    for k in ks:
        out.append(k)
    return out^


def require_string(h: Heap, a: FuncArgs) raises JinjaError:
    a.ensure_count(1)
    if len(a.args) > 0:
        a.ensure_kind(h, a.args[0], _kinds(V_STRING), "value_string_t")


def require_array(h: Heap, a: FuncArgs) raises JinjaError:
    a.ensure_count(1)
    if len(a.args) > 0:
        a.ensure_kind(h, a.args[0], _kinds(V_ARRAY, V_TUPLE), "value_array_t")


def require_int(h: Heap, a: FuncArgs) raises JinjaError:
    a.ensure_count(1)
    if len(a.args) > 0:
        a.ensure_kind(h, a.args[0], _kinds(V_INT), "value_int_t")


def require_float(h: Heap, a: FuncArgs) raises JinjaError:
    a.ensure_count(1)
    if len(a.args) > 0:
        a.ensure_kind(h, a.args[0], _kinds(V_FLOAT), "value_float_t")


def require_bool(h: Heap, a: FuncArgs) raises JinjaError:
    a.ensure_count(1)
    if len(a.args) > 0:
        a.ensure_kind(h, a.args[0], _kinds(V_BOOL), "value_bool_t")


def require_object(h: Heap, a: FuncArgs) raises JinjaError:
    a.ensure_count(1)
    if len(a.args) > 0:
        a.ensure_kind(h, a.args[0], _kinds(V_OBJECT), "value_object_t")


def not_implemented(msg: String) -> JinjaError:
    return JinjaError(K_NOTIMPL, "NotImplemented: " + msg)
