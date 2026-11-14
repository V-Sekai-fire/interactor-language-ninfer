"""Jinja builtins: global functions, tests, and the per-type filters/methods.

Mirrors `third_party/llama-jinja/jinja/value.cpp`. C++ keeps `std::function` tables per value
type; here a builtin is identified by `(receiver kind, name)` and dispatched by `call_method`
(receiver bound as argument 0) or `call_global`.
"""

from std.collections import List

from ninfer.text.jinja.jstring import (
    JString,
    characters,
    codepoint_length,
    string_bytes,
    sub_string,
)
from ninfer.text.jinja.value import (
    Heap,
    FuncArgs,
    Ctx,
    V_NONE, V_UNDEF, V_BOOL, V_INT, V_FLOAT, V_STRING, V_ARRAY, V_TUPLE, V_OBJECT, V_FUNC, V_KWARG,
    OP_EQ, OP_GE, OP_GT, OP_LT, OP_NE,
    vtype,
    as_string,
    as_bool,
    as_int,
    as_float,
    as_array,
    equivalent,
    obj_find,
    obj_insert,
    obj_insert_str,
    obj_at,
    obj_at_strict,
    value_compare,
    value_to_json,
    is_numeric_kind,
    is_array_kind,
)

# classes for `is_val<T>` checks
comptime C_INT = 0
comptime C_FLOAT = 1
comptime C_STRING = 2
comptime C_BOOL = 3
comptime C_ARRAY = 4  # includes tuples, as dynamic_cast does
comptime C_OBJECT = 5
comptime C_NONE = 6
comptime C_UNDEF = 7
comptime C_FUNC = 8


def is_class(h: Heap, v: Int, c: Int) -> Bool:
    var k = h.nodes[v].kind
    if c == C_INT:
        return k == V_INT
    if c == C_FLOAT:
        return k == V_FLOAT
    if c == C_STRING:
        return k == V_STRING
    if c == C_BOOL:
        return k == V_BOOL
    if c == C_ARRAY:
        return k == V_ARRAY or k == V_TUPLE
    if c == C_OBJECT:
        return k == V_OBJECT
    if c == C_NONE:
        return k == V_NONE
    if c == C_UNDEF:
        return k == V_UNDEF
    return k == V_FUNC


def _class_label(c: Int) -> String:
    var name = String("value_func_t")
    if c == C_INT:
        name = "value_int_t"
    elif c == C_FLOAT:
        name = "value_float_t"
    elif c == C_STRING:
        name = "value_string_t"
    elif c == C_BOOL:
        name = "value_bool_t"
    elif c == C_ARRAY:
        name = "value_array_t"
    elif c == C_OBJECT:
        name = "value_object_t"
    elif c == C_NONE:
        name = "value_none_t"
    elif c == C_UNDEF:
        name = "value_undefined_t"
    # typeid(std::shared_ptr<T>).name() under the Itanium ABI with libstdc++ (Linux)
    return "St10shared_ptrIN5jinja" + String(name.byte_length()) + name + "EE"


def ensure_vals(h: Heap, a: FuncArgs, classes: List[Int], required: List[Bool]) raises:
    var need = 0
    for r in required:
        if r:
            need += 1
    a.ensure_count(need)
    for i in range(len(classes)):
        if required[i] and len(a.args) > i:
            if not is_class(h, a.args[i], classes[i]):
                raise Error(
                    "Function '" + a.func_name + "' expected value of type " + _class_label(classes[i])
                    + ", got " + vtype(h, a.args[i])
                )


def ev1(h: Heap, a: FuncArgs, c0: Int) raises:
    var cs = List[Int]()
    cs.append(c0)
    var rs = List[Bool]()
    rs.append(True)
    ensure_vals(h, a, cs, rs)


def ev2(h: Heap, a: FuncArgs, c0: Int, c1: Int) raises:
    var cs = List[Int]()
    cs.append(c0)
    cs.append(c1)
    var rs = List[Bool]()
    rs.append(True)
    rs.append(True)
    ensure_vals(h, a, cs, rs)


def _bounds_msg(msg: String) -> Error:
    return Error(msg)


def _bool(mut h: Heap, b: Bool) -> Int:
    return h.boolean(b)


def kwarg_or_pos(mut h: Heap, a: FuncArgs, key: String, pos: Int) -> Int:
    var undef = h.undef()
    var val = a.get_kwarg(h, key, undef)
    if h.nodes[val].kind == V_UNDEF and pos < len(a.args) and h.nodes[a.args[pos]].kind != V_KWARG:
        return a.args[pos]
    return val


def pos_or(mut h: Heap, a: FuncArgs, pos: Int, default: Int) -> Int:
    return a.args[pos] if len(a.args) > pos else default


def get_pos(a: FuncArgs, pos: Int) raises -> Int:
    return a.get_pos(pos)


# --- global functions and tests ---------------------------------------------------------------


def has_global(name: String) -> Bool:
    if name == "raise_exception" or name == "namespace" or name == "strftime_now" or name == "range" or name == "tojson":
        return True
    return name.startswith("test_is_") and _is_test_name(String(name[byte=8:]))


def _is_test_name(t: String) -> Bool:
    var names = String(
        "boolean callable odd even false true divisibleby string integer float number iterable sequence mapping"
        " lower upper none defined undefined eq equalto ge gt greaterthan lt lessthan ne in test sameas escaped filter"
    ).split(" ")
    for n in names:
        if n == t:
            return True
    return False


def call_global(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    a.func_name = name
    if name == "raise_exception":
        ev1(h, a, C_STRING)
        raise Error("Jinja Exception: " + as_string(h, a.get_pos(0)).str())
    if name == "namespace":
        var out = h.object()
        h.nodes[out].has_builtins = True
        for arg in a.args:
            if h.nodes[arg].kind != V_KWARG:
                raise Error("namespace() arguments must be kwargs")
            var kname = h.nodes[arg].name
            obj_insert_str(h, out, kname, h.nodes[arg].kwval)
        return out
    if name == "strftime_now":
        ev1(h, a, C_STRING)
        var fmt = as_string(h, a.get_pos(0)).str()
        if fmt.byte_length() == 0:
            return h.text("")
        if fmt.find("%") < 0:
            return h.string(JString(fmt, 0, True))
        raise Error("NotImplemented: strftime_now is not supported by the Mojo port")
    if name == "range":
        a.ensure_count(1, 3)
        # ensure_vals<int,int,int>(true,false,false): one required argument, type-checked
        a.ensure_count(1)
        if not is_class(h, a.args[0], C_INT):
            raise Error(
                "Function '" + a.func_name + "' expected value of type " + _class_label(C_INT)
                + ", got " + vtype(h, a.args[0])
            )
        var start = 0
        var stop = 0
        var step = 1
        if len(a.args) == 1:
            stop = as_int(h, a.args[0])
        elif len(a.args) == 2:
            start = as_int(h, a.args[0])
            stop = as_int(h, a.args[1])
        else:
            start = as_int(h, a.args[0])
            stop = as_int(h, a.args[1])
            step = as_int(h, a.args[2])
        if step == 0:
            raise Error("range() step argument must not be zero")
        var out = h.array()
        var i = start
        while (i < stop) if step > 0 else (i > stop):
            out_push(h, out, h.integer(i))
            i += step
        return out
    if name == "tojson":
        return tojson(h, a)
    if name.startswith("test_is_"):
        return call_test(h, String(name[byte=8:]), a)
    raise Error("Unknown function '" + name + "'")


def out_push(mut h: Heap, arr: Int, v: Int) raises:
    if h.nodes[arr].kind != V_ARRAY:
        raise Error("Attempting to modify immutable type")
    h.nodes[arr].arr.append(v)


def _type_test(mut h: Heap, a: FuncArgs, c0: Int, c1: Int = -1, c2: Int = -1) raises -> Int:
    a.ensure_count(1)
    var v = a.get_pos(0)
    var r = is_class(h, v, c0) or (c1 >= 0 and is_class(h, v, c1)) or (c2 >= 0 and is_class(h, v, c2))
    return h.boolean(r)


def _cmp_test(mut h: Heap, a: FuncArgs, op: Int) raises -> Int:
    a.ensure_count(2, 2)
    return h.boolean(value_compare(h, a.get_pos(0), a.get_pos(1), op))


def call_test(mut h: Heap, t: String, mut a: FuncArgs) raises -> Int:
    if t == "boolean":
        return _type_test(h, a, C_BOOL)
    if t == "callable":
        return _type_test(h, a, C_FUNC)
    if t == "odd" or t == "even":
        ev1(h, a, C_INT)
        var val = as_int(h, a.get_pos(0))
        var odd = (val % 2) != 0
        return h.boolean(odd if t == "odd" else not odd)
    if t == "false":
        a.ensure_count(1)
        var v = a.get_pos(0)
        return h.boolean(is_class(h, v, C_BOOL) and not as_bool(h, v))
    if t == "true":
        a.ensure_count(1)
        var v = a.get_pos(0)
        return h.boolean(is_class(h, v, C_BOOL) and as_bool(h, v))
    if t == "divisibleby":
        ev2(h, a, C_INT, C_INT)
        var d = h.nodes[a.get_pos(1)].i
        if d == 0:
            raise Error("integer division by zero")
        return h.boolean(h.nodes[a.get_pos(0)].i % d == 0)
    if t == "string":
        return _type_test(h, a, C_STRING)
    if t == "integer":
        return _type_test(h, a, C_INT)
    if t == "float":
        return _type_test(h, a, C_FLOAT)
    if t == "number":
        return _type_test(h, a, C_INT, C_FLOAT)
    if t == "iterable" or t == "sequence":
        return _type_test(h, a, C_ARRAY, C_STRING, C_UNDEF)
    if t == "mapping":
        return _type_test(h, a, C_OBJECT)
    if t == "lower" or t == "upper":
        ev1(h, a, C_STRING)
        var s = h.nodes[a.get_pos(0)].s.copy()
        return h.boolean(s.is_lowercase(h.ju) if t == "lower" else s.is_uppercase(h.ju))
    if t == "none":
        return _type_test(h, a, C_NONE)
    if t == "defined":
        a.ensure_count(1)
        return h.boolean(h.nodes[a.get_pos(0)].kind != V_UNDEF)
    if t == "undefined":
        return _type_test(h, a, C_UNDEF)
    if t == "eq" or t == "equalto":
        return _cmp_test(h, a, OP_EQ)
    if t == "ge":
        return _cmp_test(h, a, OP_GE)
    if t == "gt" or t == "greaterthan":
        return _cmp_test(h, a, OP_GT)
    if t == "lt" or t == "lessthan":
        return _cmp_test(h, a, OP_LT)
    if t == "ne":
        return _cmp_test(h, a, OP_NE)
    if t == "in":
        a.ensure_count(2)
        var needle = a.get_pos(0)
        var hay = a.get_pos(1)
        var hk = h.nodes[hay].kind
        if hk == V_UNDEF:
            return h.boolean(False)
        if hk == V_ARRAY or hk == V_TUPLE:
            for item in h.nodes[hay].arr.copy():
                if equivalent(h, needle, item):
                    return h.boolean(True)
            return h.boolean(False)
        if hk == V_STRING:
            if h.nodes[needle].kind != V_STRING:
                raise Error(
                    "'in' test expects args[1] as string when args[0] is string, got args[1] as "
                    + vtype(h, needle)
                )
            return h.boolean(h.nodes[hay].s.str().find(h.nodes[needle].s.str()) >= 0)
        if hk == V_OBJECT:
            return h.boolean(obj_find(h, hay, needle) >= 0)
        raise Error("'in' test expects iterable as first argument, got " + vtype(h, hay))
    if t == "test":
        ev1(h, a, C_STRING)
        return h.boolean(_is_test_name(h.nodes[a.get_pos(0)].s.str()))
    if t == "sameas":
        raise Error("\x03NotImplemented: sameas test not implemented")
    if t == "escaped":
        raise Error("\x03NotImplemented: escaped test not implemented")
    if t == "filter":
        raise Error("\x03NotImplemented: filter test not implemented")
    raise Error("Unknown test '" + t + "'")


# --- tojson -----------------------------------------------------------------------------------


def tojson(mut h: Heap, mut a: FuncArgs) raises -> Int:
    a.ensure_count(1, 5)
    var val_ascii = kwarg_or_pos(h, a, "ensure_ascii", 1)
    var val_indent = kwarg_or_pos(h, a, "indent", 2)
    var val_sep = kwarg_or_pos(h, a, "separators", 3)
    var val_sort = kwarg_or_pos(h, a, "sort_keys", 4)
    var indent = -1
    if h.nodes[val_indent].kind == V_INT:
        indent = h.nodes[val_indent].i
    var ensure_ascii = as_bool(h, val_ascii)
    var separators = List[Int]()
    if is_array_kind(h.nodes[val_sep].kind):
        separators = h.nodes[val_sep].arr.copy()
    var item_sep = String(", " if indent < 0 else ",")
    if len(separators) > 0:
        item_sep = as_string(h, separators[0]).str()
    var key_sep = String(": ")
    if len(separators) > 1:
        key_sep = as_string(h, separators[1]).str()
    var first = a.get_pos(0)
    var json_str = value_to_json(h, first, indent, item_sep, key_sep, ensure_ascii, as_bool(h, val_sort))
    var result = JString(json_str, 0, True)
    if h.nodes[first].origin != 0:
        result.tag(h.nodes[first].origin, False)
    elif h.nodes[first].kind == V_STRING:
        if len(h.nodes[first].s.parts) == 1:
            result.tag(h.nodes[first].s.parts[0].origin, False)
    return h.string(result^)


# --- shared filters -----------------------------------------------------------------------------


def default_value(mut h: Heap, a: FuncArgs) raises -> Int:
    a.ensure_count(1, 3)
    var input = a.get_pos(0)
    var use_default = h.nodes[input].kind == V_UNDEF
    if not use_default:
        var b = kwarg_or_pos(h, a, "boolean", 2)
        use_default = as_bool(h, b) and not as_bool(h, input)
    if use_default:
        return a.args[1] if len(a.args) > 1 else h.text("")
    return input


def _slice_bound(h: Heap, v: Int) raises -> Optional[Int]:
    var k = h.nodes[v].kind
    if k == V_UNDEF or k == V_NONE:
        return None
    return as_int(h, v)


def slice_value(mut h: Heap, a: FuncArgs) raises -> Int:
    a.ensure_count(4, 4)
    var input = a.get_pos(0)
    var start = _slice_bound(h, a.get_pos(1))
    var stop = _slice_bound(h, a.get_pos(2))
    var st = _slice_bound(h, a.get_pos(3))
    var step = st.value() if st else 1
    if h.nodes[input].kind == V_STRING:
        var s = h.nodes[input].s.copy()
        return h.string(s.slice(h.ju, start, stop, step))
    var array = as_array(h, input)
    from ninfer.text.jinja.jstring import slice_indices

    var bounds = slice_indices(len(array), start, stop, step)
    var result = h.array()
    var i = bounds.start
    while (i < bounds.stop) if step > 0 else (i > bounds.stop):
        h.nodes[result].arr.append(array[i])
        i += step
    return result


def _split_string(mut h: Heap, a: FuncArgs, reverse: Bool) raises -> Int:
    a.ensure_count(1, 3)
    var separator = kwarg_or_pos(h, a, "sep", 1)
    var count = kwarg_or_pos(h, a, "maxsplit", 2)
    var source = as_string(h, a.get_pos(0))
    var delimiter = Optional[String]()
    if h.nodes[separator].kind != V_UNDEF and h.nodes[separator].kind != V_NONE:
        delimiter = as_string(h, separator).str()
    var maxsplit = -1 if h.nodes[count].kind == V_UNDEF else as_int(h, count)
    var result = h.array()
    for part in source.split(h.ju, delimiter, maxsplit, reverse):
        var s = h.string(part.copy())
        h.nodes[result].arr.append(s)
    return result


def _stoi(text: String, base_in: Int) raises -> Int:
    """`std::stoi(str, nullptr, base)`: raises when nothing converts or the value leaves int range."""
    var b = text.as_bytes()
    var i = 0
    while i < len(b) and (b[i] == 0x20 or (b[i] >= 0x09 and b[i] <= 0x0D)):
        i += 1
    var neg = False
    if i < len(b) and (b[i] == 0x2B or b[i] == 0x2D):
        neg = b[i] == 0x2D
        i += 1
    var base = base_in
    if (base == 0 or base == 16) and i + 1 < len(b) and b[i] == 0x30 and (b[i + 1] == 0x78 or b[i + 1] == 0x58):
        if i + 2 < len(b) and _digit(b[i + 2]) >= 0 and _digit(b[i + 2]) < 16:
            i += 2
            base = 16
    if base == 0:
        base = 8 if (i < len(b) and b[i] == 0x30) else 10
    var any = False
    var value = 0
    while i < len(b):
        var d = _digit(b[i])
        if d < 0 or d >= base:
            break
        any = True
        value = value * base + d
        if value > 4294967296:
            raise Error("out of range")
        i += 1
    if not any:
        raise Error("no conversion")
    if neg:
        value = -value
    if value > 2147483647 or value < -2147483648:
        raise Error("out of range")
    return value


def _digit(c: UInt8) -> Int:
    if c >= 0x30 and c <= 0x39:
        return Int(c) - 0x30
    if c >= 0x61 and c <= 0x7A:
        return Int(c) - 0x61 + 10
    if c >= 0x41 and c <= 0x5A:
        return Int(c) - 0x41 + 10
    return -1


def _stod(text: String) raises -> Float64:
    var b = text.as_bytes()
    var i = 0
    while i < len(b) and (b[i] == 0x20 or (b[i] >= 0x09 and b[i] <= 0x0D)):
        i += 1
    var start = i
    if i < len(b) and (b[i] == 0x2B or b[i] == 0x2D):
        i += 1
    var rest = String(text[byte=i:]).lower()
    if rest.startswith("infinity") or rest.startswith("inf"):
        return -Float64.MAX * 2.0 if b[start] == 0x2D else Float64.MAX * 2.0
    if rest.startswith("nan"):
        return Float64.MAX * 2.0 - Float64.MAX * 2.0
    var digits = 0
    while i < len(b) and b[i] >= 0x30 and b[i] <= 0x39:
        i += 1
        digits += 1
    if i < len(b) and b[i] == 0x2E:
        i += 1
        while i < len(b) and b[i] >= 0x30 and b[i] <= 0x39:
            i += 1
            digits += 1
    if digits == 0:
        raise Error("no conversion")
    if i < len(b) and (b[i] == 0x65 or b[i] == 0x45):
        var j = i + 1
        if j < len(b) and (b[j] == 0x2B or b[j] == 0x2D):
            j += 1
        var e = 0
        while j < len(b) and b[j] >= 0x30 and b[j] <= 0x39:
            j += 1
            e += 1
        if e > 0:
            i = j
    var prefix = String(text[byte=start:i])
    var value = atof(prefix)
    if value > 1.7976931348623157e308 or value < -1.7976931348623157e308:
        raise Error("out of range")
    return value


def _selectattr(mut h: Heap, mut a: FuncArgs, is_reject: Bool) raises -> Int:
    a.ensure_count(2, 4)
    var cs = List[Int]()
    cs.append(C_ARRAY)
    cs.append(C_STRING)
    cs.append(C_STRING)
    cs.append(C_STRING)
    var rs = List[Bool]()
    rs.append(True)
    rs.append(True)
    rs.append(False)
    rs.append(False)
    ensure_vals(h, a, cs, rs)
    var arr = as_array(h, a.get_pos(0))
    var attribute = a.get_pos(1)
    var out = h.array()
    var val_default = h.undef()
    var n = len(a.args)
    if n == 2:
        for item in arr:
            if h.nodes[item].kind != V_OBJECT:
                raise Error("selectattr: item is not an object")
            var attr_val = obj_at(h, item, attribute, val_default)
            var sel = as_bool(h, attr_val)
            if is_reject:
                sel = not sel
            if sel:
                h.nodes[out].arr.append(item)
        return out
    if n == 3:
        var test_name = as_string(h, a.get_pos(1)).str()
        var test_val = a.get_pos(2)
        if not _is_test_name(test_name):
            raise Error("selectattr: unknown test '" + test_name + "'")
        for item in arr:
            var ta = FuncArgs(a.ctx)
            ta.push_back(item)
            ta.push_back(test_val)
            var res = call_test(h, test_name, ta)
            var sel = as_bool(h, res)
            if is_reject:
                sel = not sel
            if sel:
                h.nodes[out].arr.append(item)
        return out
    var test_name = as_string(h, a.get_pos(2)).str()
    var extra = a.get_pos(3)
    if not _is_test_name(test_name):
        raise Error("selectattr: unknown test '" + test_name + "'")
    for item in arr:
        if h.nodes[item].kind != V_OBJECT:
            raise Error("selectattr: item is not an object")
        var attr_val = obj_at(h, item, attribute, val_default)
        var ta = FuncArgs(a.ctx)
        ta.push_back(attr_val)
        ta.push_back(extra)
        var res = call_test(h, test_name, ta)
        var sel = as_bool(h, res)
        if is_reject:
            sel = not sel
        if sel:
            h.nodes[out].arr.append(item)
    return out


# --- per-type builtin tables ---------------------------------------------------------------------


def _names(s: String) -> List[String]:
    var out = List[String]()
    for p in s.split(" "):
        out.append(String(p))
    return out^


def _has(names: String, name: String) -> Bool:
    for n in names.split(" "):
        if n == name:
            return True
    return False


comptime _INT_B = "default abs int float safe string tojson"
comptime _STR_B = (
    "default upper lower strip rstrip lstrip title capitalize length startswith endswith split rsplit replace"
    " format int float string slice safe tojson indent join"
)
comptime _BOOL_B = "default int float safe string tojson"
comptime _ARR_B = (
    "default list first last length slice selectattr select rejectattr reject join string tojson map append pop"
    " sort reverse min max unique"
)
comptime _OBJ_B = "get keys values items tojson string length dictsort join"
comptime _NONE_B = "default tojson string safe items map reject rejectattr select selectattr unique"
comptime _UNDEF_B = (
    "default capitalize first items join last length list lower map max min reject rejectattr replace reverse"
    " safe select selectattr sort string strip sum title truncate unique upper wordcount"
)


def has_method(h: Heap, recv: Int, name: String) raises -> Bool:
    """`input->get_builtins().find(name)`; raises for kinds without builtins."""
    var k = h.nodes[recv].kind
    if k == V_INT or k == V_FLOAT:
        return _has(_INT_B, name)
    if k == V_STRING:
        return _has(_STR_B, name)
    if k == V_BOOL:
        return _has(_BOOL_B, name)
    if k == V_ARRAY or k == V_TUPLE:
        return _has(_ARR_B, name)
    if k == V_OBJECT:
        return h.nodes[recv].has_builtins and _has(_OBJ_B, name)
    if k == V_NONE:
        return _has(_NONE_B, name)
    if k == V_UNDEF:
        return _has(_UNDEF_B, name)
    raise Error(vtype(h, recv) + " has no builtins")


def call_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    """Invoke builtin `name` of the type of argument 0."""
    a.func_name = name
    var recv = a.get_pos(0) if len(a.args) > 0 else -1
    var k = h.nodes[recv].kind if recv >= 0 else V_UNDEF
    if recv < 0:
        raise Error("Function '" + name + "' expected at least 1 arguments, got 0")
    if k == V_UNDEF:
        return _undef_method(h, name, a)
    if k == V_NONE:
        return _none_method(h, name, a)
    if name == "default":
        return default_value(h, a)
    if name == "tojson" and k != V_OBJECT:
        return tojson(h, a)
    if k == V_INT:
        return _int_method(h, name, a)
    if k == V_FLOAT:
        return _float_method(h, name, a)
    if k == V_BOOL:
        return _bool_method(h, name, a)
    if k == V_STRING:
        return _string_method(h, name, a)
    if k == V_OBJECT:
        return _object_method(h, name, a)
    return _array_method(h, name, a)


def _undef_method(mut h: Heap, name: String, a: FuncArgs) raises -> Int:
    if name == "default":
        return default_value(h, a)
    var strings = " capitalize join lower replace safe string strip title truncate upper "
    var arrays = " items list map reject rejectattr reverse select selectattr sort unique "
    var undefs = " first last max min "
    var ints = " length sum wordcount "
    var key = " " + name + " "
    if strings.find(key) >= 0:
        return h.text("")
    if arrays.find(key) >= 0:
        return h.array()
    if undefs.find(key) >= 0:
        return h.undef()
    if ints.find(key) >= 0:
        return h.integer(0)
    raise Error("Unknown builtin '" + name + "'")


def _none_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    if name == "default":
        return default_value(h, a)
    if name == "tojson":
        return tojson(h, a)
    if name == "string" or name == "safe":
        return h.text("None")
    return h.array()


def _int_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    if name == "abs":
        ev1(h, a, C_INT)
        var v = as_int(h, a.get_pos(0))
        return h.integer(-v if v < 0 else v)
    if name == "int":
        ev1(h, a, C_INT)
        return h.integer(as_int(h, a.get_pos(0)))
    if name == "float":
        ev1(h, a, C_INT)
        return h.float(Float64(as_int(h, a.get_pos(0))))
    # safe, string
    return tojson(h, a)


def _float_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    if name == "abs":
        ev1(h, a, C_FLOAT)
        var v = as_float(h, a.get_pos(0))
        return h.float(-v if v < 0.0 else v)
    if name == "int":
        ev1(h, a, C_FLOAT)
        return h.integer(as_int(h, a.get_pos(0)))
    if name == "float":
        ev1(h, a, C_FLOAT)
        return h.float(as_float(h, a.get_pos(0)))
    return tojson(h, a)


def _bool_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    if name == "int":
        ev1(h, a, C_BOOL)
        return h.integer(1 if as_bool(h, a.get_pos(0)) else 0)
    if name == "float":
        ev1(h, a, C_BOOL)
        return h.float(1.0 if as_bool(h, a.get_pos(0)) else 0.0)
    ev1(h, a, C_BOOL)
    return h.text("True" if as_bool(h, a.get_pos(0)) else "False")


def _strip(mut h: Heap, a: FuncArgs, left: Bool, right: Bool, check_string: Bool) raises -> Int:
    var input = a.get_pos(0)
    if check_string:
        if h.nodes[input].kind != V_STRING:
            raise Error("strip() first argument must be a string")
    else:
        ev1(h, a, C_STRING)
    var chars = kwarg_or_pos(h, a, "chars", 1)
    var s = as_string(h, a.get_pos(0))
    if h.nodes[chars].kind == V_UNDEF:
        return h.string(s.strip(h.ju, left, right))
    return h.string(s.strip(h.ju, left, right, as_string(h, chars).str()))


def _string_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    if name == "upper":
        ev1(h, a, C_STRING)
        return h.string(as_string(h, a.get_pos(0)).uppercase(h.ju))
    if name == "lower":
        ev1(h, a, C_STRING)
        return h.string(as_string(h, a.get_pos(0)).lowercase(h.ju))
    if name == "strip":
        return _strip(h, a, True, True, True)
    if name == "rstrip":
        return _strip(h, a, False, True, False)
    if name == "lstrip":
        return _strip(h, a, True, False, False)
    if name == "title":
        ev1(h, a, C_STRING)
        return h.string(as_string(h, a.get_pos(0)).titlecase(h.ju))
    if name == "capitalize":
        ev1(h, a, C_STRING)
        return h.string(as_string(h, a.get_pos(0)).capitalize(h.ju))
    if name == "length":
        ev1(h, a, C_STRING)
        return h.integer(as_string(h, a.get_pos(0)).length())
    if name == "startswith" or name == "endswith":
        ev2(h, a, C_STRING, C_STRING)
        var s = as_string(h, a.get_pos(0)).str()
        var p = as_string(h, a.get_pos(1)).str()
        return h.boolean(s.startswith(p) if name == "startswith" else s.endswith(p))
    if name == "split":
        return _split_string(h, a, False)
    if name == "rsplit":
        return _split_string(h, a, True)
    if name == "replace":
        return _replace(h, a)
    if name == "format":
        return _format(h, a)
    if name == "int":
        var val_input = a.get_pos(0)
        var val_default = kwarg_or_pos(h, a, "default", 1)
        var val_base = kwarg_or_pos(h, a, "base", 2)
        var base = 10 if h.nodes[val_base].kind == V_UNDEF else as_int(h, val_base)
        if base != 0 and (base < 2 or base > 36):
            raise Error("int() base must be 0 or between 2 and 36")
        if h.nodes[val_input].kind != V_STRING:
            raise Error("int() first argument must be a string")
        try:
            return h.integer(_stoi(h.nodes[val_input].s.str(), base))
        except:
            var d = 0 if h.nodes[val_default].kind == V_UNDEF else as_int(h, val_default)
            return h.integer(d)
    if name == "float":
        ev1(h, a, C_STRING)
        var val_default = kwarg_or_pos(h, a, "default", 1)
        try:
            return h.float(_stod(as_string(h, a.get_pos(0)).str()))
        except:
            return h.float(0.0 if h.nodes[val_default].kind == V_UNDEF else as_float(h, val_default))
    if name == "string":
        ev1(h, a, C_STRING)
        return h.string(as_string(h, a.get_pos(0)))
    if name == "slice":
        return slice_value(h, a)
    if name == "safe":
        ev1(h, a, C_STRING)
        return a.get_pos(0)
    if name == "indent":
        return _indent(h, a)
    if name == "join":
        raise Error("\x03NotImplemented: String join builtin not implemented")
    raise Error("Unknown builtin '" + name + "'")


def _replace(mut h: Heap, a: FuncArgs) raises -> Int:
    a.ensure_count(3, 4)
    var source = as_string(h, a.get_pos(0))
    var old_str = as_string(h, a.get_pos(1)).str()
    var new_str = as_string(h, a.get_pos(2))
    var limit = kwarg_or_pos(h, a, "count", 3)
    var remaining = -1
    if h.nodes[limit].kind != V_UNDEF and h.nodes[limit].kind != V_NONE:
        remaining = as_int(h, limit)
    if remaining == 0:
        return h.string(source^)
    var text = source.str()
    var result = JString()
    var cursor = 0
    if old_str.byte_length() == 0:
        result.append(new_str)
        if remaining > 0:
            remaining -= 1
        for ch in characters(h.ju, text):
            result.append(source.cut_bytes(ch.begin, ch.end))
            cursor = ch.end
            if remaining == 0:
                break
            result.append(new_str)
            if remaining > 0:
                remaining -= 1
    else:
        var tb = text.as_bytes()
        var ob = old_str.as_bytes()
        while remaining != 0:
            var pos = -1
            var k = cursor
            while k + len(ob) <= len(tb):
                var same = True
                for q in range(len(ob)):
                    if tb[k + q] != ob[q]:
                        same = False
                        break
                if same:
                    pos = k
                    break
                k += 1
            if pos < 0:
                break
            result.append(source.cut_bytes(cursor, pos))
            result.append(new_str)
            cursor = pos + len(ob)
            if remaining > 0:
                remaining -= 1
    result.append(source.cut_bytes(cursor, text.byte_length()))
    return h.string(result^)


def _format(mut h: Heap, a: FuncArgs) raises -> Int:
    var val_input = a.get_pos(0)
    if h.nodes[val_input].kind != V_STRING:
        raise Error("format() first argument must be a string")
    var fmt = as_string(h, val_input)
    var s = fmt.str()
    var result = JString()
    var begin = 0
    var arg_idx = 1
    var sb = s.as_bytes()
    var i = 0
    while i < len(sb):
        if sb[i] != 0x7B:
            i += 1
            continue
        if i + 1 >= len(sb) or sb[i + 1] != 0x7D:
            raise Error("\x03NotImplemented: format() only supports simple '{}' placeholders")
        result.append(fmt.cut_bytes(begin, i))
        result.append(as_string(h, a.get_pos(arg_idx)))
        arg_idx += 1
        i += 1
        begin = i + 1
        i += 1
    result.append(fmt.cut_bytes(begin, len(sb)))
    return h.string(result^)


def _indent(mut h: Heap, a: FuncArgs) raises -> Int:
    a.ensure_count(1, 4)
    var val_input = a.get_pos(0)
    var val_width = kwarg_or_pos(h, a, "width", 1)
    var first = as_bool(h, kwarg_or_pos(h, a, "first", 2))
    var blank = as_bool(h, kwarg_or_pos(h, a, "blank", 3))
    if h.nodes[val_input].kind != V_STRING:
        raise Error("indent() first argument must be a string")
    var indent = JString("    ")
    if h.nodes[val_width].kind == V_INT:
        indent = JString(" " * h.nodes[val_width].i)
    elif h.nodes[val_width].kind == V_STRING:
        indent = h.nodes[val_width].s.copy()
    var result = JString()
    var source = h.nodes[val_input].s.copy()
    var input = source.str()
    var ib = input.as_bytes()
    var begin = 0
    while begin < len(ib):
        var newline = -1
        var k = begin
        while k < len(ib):
            if ib[k] == 0x0A:
                newline = k
                break
            k += 1
        var end = len(ib) if newline < 0 else newline
        var should = first if begin == 0 else (begin != end or blank)
        if should:
            result.append(indent)
        var nxt = end if newline < 0 else end + 1
        result.append(source.cut_bytes(begin, nxt))
        begin = nxt
    if len(ib) > 0 and ib[len(ib) - 1] == 0x0A and blank:
        result.append(indent)
    return h.string(result^)


# --- arrays ------------------------------------------------------------------------------------


def _sorted(mut h: Heap, items: List[Int], reverse: Bool, attribute: Int, attr_is_int: Bool, attr_int: Int) raises -> List[Int]:
    var arr = items.copy()
    var op = OP_GT if reverse else OP_LT
    for i in range(1, len(arr)):
        var j = i
        while j > 0:
            var a = arr[j]
            var b = arr[j - 1]
            var va = a
            var vb = b
            if h.nodes[attribute].kind != V_UNDEF:
                if attr_is_int and is_array_kind(h.nodes[a].kind) and is_array_kind(h.nodes[b].kind):
                    va = _array_at(h, a, attr_int)
                    vb = _array_at(h, b, attr_int)
                elif (not attr_is_int) and h.nodes[a].kind == V_OBJECT and h.nodes[b].kind == V_OBJECT:
                    va = obj_at_strict(h, a, attribute)
                    vb = obj_at_strict(h, b, attribute)
                else:
                    raise Error(
                        "sort: unsupported object attribute comparison between " + vtype(h, a) + " and " + vtype(h, b)
                    )
            if value_compare(h, va, vb, op):
                arr[j] = b
                arr[j - 1] = a
                j -= 1
            else:
                break
    return arr^


def _array_at(h: Heap, arr: Int, index_in: Int) raises -> Int:
    var index = index_in
    var n = len(h.nodes[arr].arr)
    if index < 0:
        index += n
    if index < 0 or index >= n:
        raise Error("Index " + String(index) + " out of bounds for array of size " + String(n))
    return h.nodes[arr].arr[index]


def _array_at_default(h: Heap, arr: Int, index_in: Int, default: Int) -> Int:
    var index = index_in
    var n = len(h.nodes[arr].arr)
    if index < 0:
        index += n
    if index < 0 or index >= n:
        return default
    return h.nodes[arr].arr[index]


def _copy_kind(mut h: Heap, like: Int, items: List[Int]) -> Int:
    if h.nodes[like].kind == V_TUPLE:
        return h.tuple(items)
    var r = h.array()
    h.nodes[r].arr = items.copy()
    return r


def _array_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    if name == "list":
        ev1(h, a, C_ARRAY)
        var r = h.array()
        h.nodes[r].arr = as_array(h, a.get_pos(0))
        return r
    if name == "first" or name == "last":
        ev1(h, a, C_ARRAY)
        var arr = as_array(h, a.get_pos(0))
        if len(arr) == 0:
            return h.undef()
        return arr[0] if name == "first" else arr[len(arr) - 1]
    if name == "length":
        ev1(h, a, C_ARRAY)
        return h.integer(len(as_array(h, a.get_pos(0))))
    if name == "slice":
        return slice_value(h, a)
    if name == "selectattr" or name == "select":
        return _selectattr(h, a, False)
    if name == "rejectattr" or name == "reject":
        return _selectattr(h, a, True)
    if name == "join":
        return _join(h, a)
    if name == "string":
        ev1(h, a, C_ARRAY)
        return h.string(as_string(h, a.get_pos(0)))
    if name == "map":
        return _map(h, a)
    if name == "append":
        a.ensure_count(2)
        if not is_class(h, a.get_pos(0), C_ARRAY):
            raise Error("append: first argument must be an array")
        out_push(h, a.get_pos(0), a.get_pos(1))
        return a.get_pos(0)
    if name == "pop":
        a.ensure_count(1, 2)
        var cs = List[Int]()
        cs.append(C_ARRAY)
        cs.append(C_INT)
        var rs = List[Bool]()
        rs.append(True)
        rs.append(False)
        ensure_vals(h, a, cs, rs)
        var index = as_int(h, a.get_pos(1)) if len(a.args) == 2 else -1
        var arr = a.get_pos(0)
        if h.nodes[arr].kind != V_ARRAY:
            raise Error("Attempting to modify immutable type")
        var n = len(h.nodes[arr].arr)
        if index < 0:
            index = n + index
        if index < 0 or index >= n:
            raise Error("Index " + String(index) + " out of bounds for array of size " + String(n))
        var val = h.nodes[arr].arr[index]
        _ = h.nodes[arr].arr.pop(index)
        return val
    if name == "sort":
        a.ensure_count(1, 4)
        if not is_class(h, a.get_pos(0), C_ARRAY):
            raise Error("sort: first argument must be an array")
        var val = a.get_pos(0)
        var reverse = as_bool(h, kwarg_or_pos(h, a, "reverse", 1))
        var attribute = kwarg_or_pos(h, a, "attribute", 3)
        _ = kwarg_or_pos(h, a, "case_sensitive", 2)
        var attr_is_int = h.nodes[attribute].kind == V_INT
        var attr_int = h.nodes[attribute].i if attr_is_int else 0
        var items_copy = h.nodes[val].arr.copy()
        var sorted = _sorted(h, items_copy, reverse, attribute, attr_is_int, attr_int)
        return _copy_kind(h, val, sorted)
    if name == "reverse":
        ev1(h, a, C_ARRAY)
        var val = a.get_pos(0)
        var arr = as_array(h, val)
        var out = List[Int]()
        for i in range(len(arr)):
            out.append(arr[len(arr) - 1 - i])
        return _copy_kind(h, val, out)
    if name == "min" or name == "max":
        a.ensure_count(1, 4)
        ev1(h, a, C_ARRAY)
        _ = kwarg_or_pos(h, a, "case_sensitive", 1)
        var attribute = kwarg_or_pos(h, a, "attribute", 2)
        if h.nodes[attribute].kind != V_UNDEF:
            raise Error("\x03NotImplemented: " + name + ": attribute not implemented")
        var arr = as_array(h, a.get_pos(0))
        if len(arr) == 0:
            return h.undef()
        var result = arr[0]
        for i in range(1, len(arr)):
            if value_compare(h, arr[i], result, OP_LT if name == "min" else OP_GT):
                result = arr[i]
        return result
    if name == "unique":
        raise Error("\x03NotImplemented: Array unique builtin not implemented")
    raise Error("Unknown builtin '" + name + "'")


def _join(mut h: Heap, a: FuncArgs) raises -> Int:
    a.ensure_count(1, 3)
    if not is_class(h, a.get_pos(0), C_ARRAY):
        raise Error("join() first argument must be an array")
    var val_delim = kwarg_or_pos(h, a, "d", 1)
    var attribute = kwarg_or_pos(h, a, "attribute", 2)
    var arr = as_array(h, a.get_pos(0))
    var attr_is_int = h.nodes[attribute].kind == V_INT
    if h.nodes[attribute].kind != V_UNDEF and h.nodes[attribute].kind != V_STRING and not attr_is_int:
        raise Error("join() attribute must be string or integer")
    var attr_int = h.nodes[attribute].i if attr_is_int else 0
    var delim = JString() if h.nodes[val_delim].kind == V_UNDEF else as_string(h, val_delim)
    var result = JString()
    for i in range(len(arr)):
        var val_arr = arr[i]
        if h.nodes[attribute].kind != V_UNDEF:
            if attr_is_int and is_array_kind(h.nodes[val_arr].kind):
                val_arr = _array_at(h, val_arr, attr_int)
            elif (not attr_is_int) and h.nodes[val_arr].kind == V_OBJECT:
                val_arr = obj_at_strict(h, val_arr, attribute)
        var k = h.nodes[val_arr].kind
        if k != V_STRING and k != V_INT and k != V_FLOAT:
            raise Error("join() can only join arrays of strings or numerics")
        result.append(as_string(h, val_arr))
        if i < len(arr) - 1:
            result.append(delim)
    return h.string(result^)


def _map(mut h: Heap, a: FuncArgs) raises -> Int:
    a.ensure_count(2)
    if not is_class(h, a.get_pos(0), C_ARRAY):
        raise Error("map: first argument must be an array")
    if h.nodes[a.args[1]].kind != V_KWARG:
        raise Error("\x03NotImplemented: map: filter-mapping not implemented")
    var val = a.get_pos(0)
    var attribute = kwarg_or_pos(h, a, "attribute", 1)
    var attr_is_int = h.nodes[attribute].kind == V_INT
    if h.nodes[attribute].kind != V_STRING and not attr_is_int:
        raise Error("map: attribute must be string or integer")
    var attr_int = h.nodes[attribute].i if attr_is_int else 0
    var default_val = a.get_kwarg(h, "default", h.undef())
    var out = List[Int]()
    for item in as_array(h, val):
        var attr_val = default_val
        if attr_is_int:
            if is_array_kind(h.nodes[item].kind):
                attr_val = _array_at_default(h, item, attr_int, default_val)
        elif h.nodes[item].kind == V_OBJECT:
            attr_val = obj_at(h, item, attribute, default_val)
        out.append(attr_val)
    return _copy_kind(h, val, out)


# --- objects -------------------------------------------------------------------------------------


def _object_method(mut h: Heap, name: String, mut a: FuncArgs) raises -> Int:
    if name == "get":
        a.ensure_count(2, 3)
        if h.nodes[a.get_pos(0)].kind != V_OBJECT:
            raise Error("get: first argument must be an object")
        if h.nodes[a.get_pos(1)].kind != V_STRING:
            raise Error("get: second argument must be a string (key)")
        var default_val = h.none()
        if len(a.args) == 3:
            default_val = a.get_pos(2)
        return obj_at(h, a.get_pos(0), a.get_pos(1), default_val)
    if name == "keys" or name == "values" or name == "items":
        ev1(h, a, C_OBJECT)
        var obj = a.get_pos(0)
        var keys = h.nodes[obj].keys.copy()
        var vals = h.nodes[obj].vals.copy()
        var result = h.array()
        for i in range(len(keys)):
            if name == "keys":
                h.nodes[result].arr.append(keys[i])
            elif name == "values":
                h.nodes[result].arr.append(vals[i])
            else:
                var pair = List[Int]()
                pair.append(keys[i])
                pair.append(vals[i])
                var t = h.tuple(pair)
                h.nodes[result].arr.append(t)
        return result
    if name == "tojson":
        ev1(h, a, C_OBJECT)
        return tojson(h, a)
    if name == "string":
        ev1(h, a, C_OBJECT)
        return h.string(as_string(h, a.get_pos(0)))
    if name == "length":
        ev1(h, a, C_OBJECT)
        return h.integer(len(h.nodes[a.get_pos(0)].keys))
    if name == "dictsort":
        var val_input = a.get_pos(0)
        _ = kwarg_or_pos(h, a, "case_sensitive", 1)
        var val_by = kwarg_or_pos(h, a, "by", 2)
        var reverse = as_bool(h, kwarg_or_pos(h, a, "reverse", 3))
        var by_value = h.nodes[val_by].kind == V_STRING and h.nodes[val_by].s.str() == "value"
        var keys = h.nodes[val_input].keys.copy()
        var vals = h.nodes[val_input].vals.copy()
        var op = OP_GT if reverse else OP_LT
        for i in range(1, len(keys)):
            var j = i
            while j > 0:
                var less: Bool
                if by_value:
                    less = value_compare(h, vals[j], vals[j - 1], op)
                else:
                    less = value_compare(h, keys[j], keys[j - 1], op)
                if not less:
                    break
                var tk = keys[j]
                keys[j] = keys[j - 1]
                keys[j - 1] = tk
                var tv = vals[j]
                vals[j] = vals[j - 1]
                vals[j - 1] = tv
                j -= 1
        var out = h.object()
        h.nodes[out].keys = keys^
        h.nodes[out].vals = vals^
        return out
    if name == "join":
        raise Error("\x03NotImplemented: object join not implemented")
    raise Error("Unknown builtin '" + name + "'")
