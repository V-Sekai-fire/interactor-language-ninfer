"""Parser combinators for Mojo, in the style of dashbitco/nimble_parsec.

This is not NimbleParsec. It borrows its combinator vocabulary and, in `parse` mode, its
observable behaviour, which `tests/test_combinator.py` checks against the real library; `scan` and
`uclass` below are this project's own. Combinators are composed at run time into a tree,
errors report `{reason, rest, line, offset}` with NimbleParsec's reason text, and the parser
backtracks exactly where the compiled clauses do (choice, repeat, times, lookahead).

The Elixir original compiles combinators into binary-matching clauses; here a matcher walks the
tree instead. What it keeps is the observable behaviour, including how consecutive "bound"
combinators (strings, character classes, bytes, eos) fail together as one unit and report the
start of that unit. The test runs both implementations on the same combinators.
Context is always the empty map and callbacks are the built-in operations below.
"""

from std.collections import Dict, List

from ninfer.text.unicode import Unicode, utf8_codepoint_at

# Terms (what a parser emits).
comptime T_INT = 0
comptime T_BIN = 1
comptime T_LIST = 2
comptime T_TUPLE = 3
comptime T_ATOM = 4

# Combinator elements.
comptime E_STRING = 0
comptime E_CHAR = 1
comptime E_BYTES = 2
comptime E_EOS = 3
comptime E_LABEL = 4
comptime E_TRAVERSE = 5
comptime E_CHOICE = 6
comptime E_REPEAT = 7
comptime E_TIMES = 8
comptime E_EVENTUALLY = 9
comptime E_LOOKAHEAD = 10
comptime E_CALL = 11  # run another parser by name (NimbleParsec's `parsec(:name)`)
comptime E_UCLASS = 13  # one Unicode code point by category; only `scan` recognises it
comptime E_SLICE = 12  # ascii_string / utf8_string with min/max whose class cannot match a newline

# Unicode categories for `uclass` (the groups the Qwen split pattern names).
comptime U_LETTER = 1  # \p{L}
comptime U_MARK = 2  # \p{M}
comptime U_NUMBER = 4  # \p{N}
comptime U_SPACE = 8  # \s
comptime U_NEWLINE = 16  # [\r\n]

# Traverse operations.
comptime OP_IGNORE = 0
comptime OP_REPLACE = 1
comptime OP_TAG = 2
comptime OP_UNWRAP_TAG = 3
comptime OP_WRAP = 4
comptime OP_LINE = 5
comptime OP_BYTE_OFFSET = 6
comptime OP_DIGITS = 7  # compile-time integer(n): digit codepoints -> one integer
comptime OP_FOLD_INT = 8  # runtime integer(min..max)
comptime OP_TO_STRING = 9  # ascii_string / utf8_string: codepoints and binaries -> one binary
comptime OP_REDUCE_SUM = 10
comptime OP_MAP_INC = 11
comptime OP_POST_COUNT = 12  # unbound (a user `post_traverse` callback)

comptime _UNBOUND_OP_START = 12


struct Term(Copyable, Movable):
    var kind: Int
    var i: Int
    var s: String
    var kids: List[Int]

    def __init__(out self, kind: Int):
        self.kind = kind
        self.i = 0
        self.s = String()
        self.kids = List[Int]()


struct Elem(Copyable, Movable):
    var kind: Int
    var s: String  # string / label / called parser name / tag atom
    var inc: List[Int]  # char ranges as lo,hi pairs
    var exc: List[Int]
    var utf8: Bool
    var n: Int  # bytes count, times count, op
    var m: Int
    var kids: List[Int]  # list ids: traverse/label/repeat inner = kids[0]; choice alternatives
    var positive: Bool
    var value: Int  # replace constant term id

    def __init__(out self, kind: Int):
        self.kind = kind
        self.s = String()
        self.inc = List[Int]()
        self.exc = List[Int]()
        self.utf8 = False
        self.n = 0
        self.m = 0
        self.kids = List[Int]()
        self.positive = True
        self.value = -1


struct Failure(Copyable, Movable):
    var reason: String
    var pos: Int
    var line: Int
    var line_off: Int

    def __init__(out self):
        self.reason = String()
        self.pos = 0
        self.line = 1
        self.line_off = 0


struct Result(Movable):
    var ok: Bool
    var acc: List[Int]  # term ids, in order
    var rest: String
    var line: Int
    var line_off: Int
    var offset: Int
    var reason: String

    def __init__(out self):
        self.ok = False
        self.acc = List[Int]()
        self.rest = String()
        self.line = 1
        self.line_off = 0
        self.offset = 0
        self.reason = String()


struct _State(Copyable, Movable):
    var pos: Int
    var line: Int
    var line_off: Int
    var acc: List[Int]  # emission order (NimbleParsec keeps it reversed; callbacks here are built in)

    def __init__(out self):
        self.pos = 0
        self.line = 1
        self.line_off = 0
        self.acc = List[Int]()


def _inspect_binary(s: String) -> String:
    """Elixir `inspect/1` of a binary that is valid UTF-8."""
    var out = String('"')
    for cp in s.codepoints():
        var c = Int(cp.to_u32())
        if c == 0x22:
            out += '\\"'
        elif c == 0x5C:
            out += "\\\\"
        elif c == 0x0A:
            out += "\\n"
        elif c == 0x0D:
            out += "\\r"
        elif c == 0x09:
            out += "\\t"
        elif c == 0x1B:
            out += "\\e"
        elif c == 0x07:
            out += "\\a"
        elif c == 0x08:
            out += "\\b"
        elif c == 0x0C:
            out += "\\f"
        elif c == 0x0B:
            out += "\\v"
        elif c == 0x00:
            out += "\\0"
        elif c == 0x23:
            out += "#"
        elif c < 0x20 or c == 0x7F or (c >= 0x80 and c < 0xA0):
            out += "\\x" + _hex(c)
        else:
            out += String(chr(c))
    out += '"'
    return out^


def _hex(v: Int) -> String:
    var digits = "0123456789ABCDEF"
    var out = String()
    var started = False
    for shift in [4, 0]:
        var d = (v >> shift) & 15
        if d != 0 or started or shift == 0:
            out += digits[byte=d : d + 1]
            started = True
    return "{" + out + "}"


struct Parser(Movable):
    var terms: List[Term]
    var elems: List[Elem]
    var lists: List[List[Int]]  # a combinator is a list of element ids
    var named: Dict[String, Int]
    var eos_seen: Bool  # an `eos()` inside the bound run being matched

    def __init__(out self):
        self.eos_seen = False
        self.terms = List[Term]()
        self.elems = List[Elem]()
        self.lists = List[List[Int]]()
        self.named = Dict[String, Int]()

    # --- terms -----------------------------------------------------------------------------

    def t_int(mut self, v: Int) -> Int:
        var t = Term(T_INT)
        t.i = v
        self.terms.append(t^)
        return len(self.terms) - 1

    def t_bin(mut self, v: String) -> Int:
        var t = Term(T_BIN)
        t.s = v
        self.terms.append(t^)
        return len(self.terms) - 1

    def t_atom(mut self, v: String) -> Int:
        var t = Term(T_ATOM)
        t.s = v
        self.terms.append(t^)
        return len(self.terms) - 1

    def t_list(mut self, kids: List[Int]) -> Int:
        var t = Term(T_LIST)
        t.kids = kids.copy()
        self.terms.append(t^)
        return len(self.terms) - 1

    def t_tuple(mut self, kids: List[Int]) -> Int:
        var t = Term(T_TUPLE)
        t.kids = kids.copy()
        self.terms.append(t^)
        return len(self.terms) - 1

    def show(self, t: Int) -> String:
        """Stable text form shared with the Elixir oracle."""
        var k = self.terms[t].kind
        if k == T_INT:
            return "i" + String(self.terms[t].i)
        if k == T_BIN:
            return "b" + _hex_str(self.terms[t].s)
        if k == T_ATOM:
            return "a" + self.terms[t].s
        var out = String("[" if k == T_LIST else "{")
        for i in range(len(self.terms[t].kids)):
            if i > 0:
                out += ","
            out += self.show(self.terms[t].kids[i])
        out += "]" if k == T_LIST else "}"
        return out^

    # --- combinators ------------------------------------------------------------------------

    def _list(mut self, items: List[Int]) -> Int:
        self.lists.append(items.copy())
        return len(self.lists) - 1

    def _one(mut self, e: Elem) -> Int:
        self.elems.append(e.copy())
        var items = List[Int]()
        items.append(len(self.elems) - 1)
        return self._list(items)

    def empty(mut self) -> Int:
        return self._list(List[Int]())

    def concat(mut self, a: Int, b: Int) -> Int:
        var items = self.lists[a].copy()
        for x in self.lists[b]:
            items.append(x)
        return self._list(items)

    def string(mut self, s: String) -> Int:
        var e = Elem(E_STRING)
        e.s = s
        return self._one(e)

    def _char(mut self, ranges: List[Int], excluded: List[Int], utf8: Bool) -> Int:
        var e = Elem(E_CHAR)
        e.inc = ranges.copy()
        e.exc = excluded.copy()
        e.utf8 = utf8
        return self._one(e)

    def ascii_char(mut self, inc: List[Int], exc: List[Int] = List[Int]()) -> Int:
        return self._char(inc, exc, False)

    def utf8_char(mut self, inc: List[Int], exc: List[Int] = List[Int]()) -> Int:
        return self._char(inc, exc, True)

    def uclass(mut self, include: Int, exclude: Int = 0) -> Int:
        """One code point that is in any `U_*` category of `include` (any code point when 0) and in
        none of `exclude`. Recognised by `scan` only."""
        var e = Elem(E_UCLASS)
        e.n = include
        e.m = exclude
        return self._one(e)

    def bytes(mut self, n: Int) -> Int:
        var e = Elem(E_BYTES)
        e.n = n
        return self._one(e)

    def eos(mut self) -> Int:
        return self._one(Elem(E_EOS))

    def duplicate(mut self, c: Int, n: Int) -> Int:
        var items = List[Int]()
        for _ in range(n):
            for x in self.lists[c]:
                items.append(x)
        return self._list(items)

    def _traverse(mut self, c: Int, op: Int, atom: String = "", value: Int = -1) -> Int:
        var e = Elem(E_TRAVERSE)
        e.kids.append(c)
        e.n = op
        e.s = atom
        e.value = value
        return self._one(e)

    def ignore(mut self, c: Int) -> Int:
        return self._traverse(c, OP_IGNORE)

    def replace(mut self, c: Int, value: Int) -> Int:
        return self._traverse(c, OP_REPLACE, "", value)

    def tag(mut self, c: Int, atom: String) -> Int:
        return self._traverse(c, OP_TAG, atom)

    def unwrap_and_tag(mut self, c: Int, atom: String) -> Int:
        return self._traverse(c, OP_UNWRAP_TAG, atom)

    def wrap(mut self, c: Int) -> Int:
        return self._traverse(c, OP_WRAP)

    def line(mut self, c: Int) -> Int:
        return self._traverse(c, OP_LINE)

    def byte_offset(mut self, c: Int) -> Int:
        return self._traverse(c, OP_BYTE_OFFSET)

    def reduce_sum(mut self, c: Int) -> Int:
        return self._traverse(c, OP_REDUCE_SUM)

    def map_inc(mut self, c: Int) -> Int:
        return self._traverse(c, OP_MAP_INC)

    def post_count(mut self, c: Int) -> Int:
        return self._traverse(c, OP_POST_COUNT)

    def label(mut self, c: Int, text: String) -> Int:
        var e = Elem(E_LABEL)
        e.kids.append(c)
        e.s = text
        return self._one(e)

    def choice(mut self, alts: List[Int]) -> Int:
        var e = Elem(E_CHOICE)
        e.kids = alts.copy()
        return self._one(e)

    def optional(mut self, c: Int) -> Int:
        var alts = List[Int]()
        alts.append(c)
        alts.append(self.empty())
        return self.choice(alts)

    def repeat(mut self, c: Int) -> Int:
        var e = Elem(E_REPEAT)
        e.kids.append(c)
        return self._one(e)

    def eventually(mut self, c: Int) -> Int:
        var e = Elem(E_EVENTUALLY)
        e.kids.append(c)
        return self._one(e)

    def lookahead(mut self, c: Int) -> Int:
        var e = Elem(E_LOOKAHEAD)
        e.kids.append(c)
        return self._one(e)

    def lookahead_not(mut self, c: Int) -> Int:
        var e = Elem(E_LOOKAHEAD)
        e.kids.append(c)
        e.positive = False
        return self._one(e)

    def call(mut self, name: String) -> Int:
        """Run the parser registered under `name` (NimbleParsec's `parsec(:name)`)."""
        var e = Elem(E_CALL)
        e.s = name
        return self._one(e)

    def times(mut self, c: Int, min: Int, max: Int = -1) -> Int:
        """`times(c, n)` is `times(c, min: n, max: n)`; `max < 0` means unbounded."""
        var out = self.duplicate(c, min)
        if max < 0:
            return self.concat(out, self.repeat(c))
        if max - min > 0:
            var e = Elem(E_TIMES)
            e.kids.append(c)
            e.n = max - min
            out = self.concat(out, self._one(e))
        return out

    def integer(mut self, min: Int, max: Int = -2) -> Int:
        """`integer(n)` when `max == -2`, else `integer(min: min, max: max)` (`max < 0`: no limit)."""
        var digit_ranges = List[Int]()
        digit_ranges.append(0x30)
        digit_ranges.append(0x39)
        var digit = self.ascii_char(digit_ranges)
        if max == -2:
            return self._traverse(self.duplicate(digit, min), OP_DIGITS)
        var head = self.empty()
        if min > 0:
            head = self._traverse(self.duplicate(digit, min), OP_DIGITS)
        var tail = self.empty()
        if max >= 0:
            if max - min > 0:
                var e = Elem(E_TIMES)
                e.kids.append(digit)
                e.n = max - min
                tail = self._one(e)
        else:
            tail = self.repeat(digit)
        return self._traverse(self.concat(head, tail), OP_FOLD_INT)

    def _string_of(mut self, chars: Int, min: Int, max: Int, exact: Bool) -> Int:
        if exact:
            return self._traverse(self.duplicate(chars, min), OP_TO_STRING)
        ref class_elem = self.elems[self.lists[chars][0]]
        if not _newline_possible(class_elem.inc, class_elem.exc):
            var e = Elem(E_SLICE)
            e.kids.append(chars)
            e.kids.append(self.duplicate(chars, min))
            e.n = min
            e.m = max
            return self._one(e)
        var head = self.empty()
        if min > 0:
            head = self._traverse(self.duplicate(chars, min), OP_TO_STRING)
        var tail = self.empty()
        if max >= 0:
            if max - min > 0:
                var e = Elem(E_TIMES)
                e.kids.append(chars)
                e.n = max - min
                tail = self._one(e)
        else:
            tail = self.repeat(chars)
        return self._traverse(self.concat(head, tail), OP_TO_STRING)

    def ascii_string(mut self, inc: List[Int], min: Int, max: Int = -2) -> Int:
        return self._string_of(self.ascii_char(inc), min, max, max == -2)

    def utf8_string(mut self, inc: List[Int], min: Int, max: Int = -2) -> Int:
        return self._string_of(self.utf8_char(inc), min, max, max == -2)

    def define(mut self, name: String, c: Int):
        self.named[name] = c

    # --- labels -----------------------------------------------------------------------------

    def _range_text(self, lo: Int, hi: Int, printable: Bool) -> String:
        _ = printable
        if lo == hi:
            return " equal to " + _inspect_binary(String(chr(lo)))
        return " in the range " + _inspect_binary(String(chr(lo))) + " to " + _inspect_binary(String(chr(hi)))

    def _elem_label(self, e: Int) -> String:
        ref el = self.elems[e]
        if el.kind == E_STRING:
            return "string " + _inspect_binary(el.s)
        if el.kind == E_LABEL:
            return el.s
        if el.kind == E_CHAR:
            var printable = True
            var inc = String()
            var i = 0
            while i < len(el.inc):
                if i > 0:
                    inc += " or"
                inc += self._range_text(el.inc[i], el.inc[i + 1], True)
                if not _printable(el.inc[i]) or not _printable(el.inc[i + 1]):
                    printable = False
                i += 2
            var out = inc
            var j = 0
            while j < len(el.exc):
                out += ", and not" + self._range_text(el.exc[j], el.exc[j + 1], True)
                if not _printable(el.exc[j]) or not _printable(el.exc[j + 1]):
                    printable = False
                j += 2
            var prefix = String("utf8 codepoint")
            if not el.utf8:
                prefix = "ASCII character" if printable else "byte"
            return prefix + out
        if el.kind == E_EOS:
            return "end of string"
        if el.kind == E_BYTES:
            return String(el.n) + " bytes"
        if el.kind == E_SLICE:
            return self._elem_label(self.lists[el.kids[0]][0])
        if el.kind == E_LOOKAHEAD or el.kind == E_REPEAT or el.kind == E_TIMES or el.kind == E_TRAVERSE:
            return self._labels(self.lists[el.kids[0]])
        if el.kind == E_EVENTUALLY:
            return self._labels(self.lists[el.kids[0]]) + " eventually"
        if el.kind == E_CHOICE:
            var out = String()
            for k in range(len(el.kids)):
                if k > 0:
                    out += " or "
                out += self._labels(self.lists[el.kids[k]])
            return out^
        return el.s  # called parser name

    def _labels(self, items: List[Int]) -> String:
        if len(items) == 0:
            return "nothing"
        var out = String()
        for i in range(len(items)):
            if i > 0:
                out += ", followed by "
            out += self._elem_label(items[i])
        return out^

    def _reason(self, items: List[Int], labels: List[String], positive: Bool) -> String:
        var body = String()
        if len(labels) == 0:
            body = self._labels(items)
        elif len(labels) == 1:
            body = labels[0]
        else:
            body = labels[0] + " while processing "
            for i in range(1, len(labels)):
                if i > 1:
                    body += " inside "
                body += labels[i]
        return ("expected " if positive else "did not expect ") + body

    # --- boundedness ------------------------------------------------------------------------

    def _bound_list(self, items: List[Int]) -> Bool:
        """`take_bound_combinators` consumes the whole list; a direct `eos()` ends the run, so
        anything after it (but more `eos()`) leaves the list unbound."""
        var i = 0
        while i < len(items):
            if not self._bound(items[i]):
                return False
            if self.elems[items[i]].kind == E_EOS:
                i += 1
                while i < len(items) and self.elems[items[i]].kind == E_EOS:
                    i += 1
                return i >= len(items)
            i += 1
        return True

    def _bound(self, e: Int) -> Bool:
        ref el = self.elems[e]
        var k = el.kind
        if k == E_STRING or k == E_CHAR or k == E_BYTES or k == E_EOS:
            return True
        if k == E_LABEL:
            return self._bound_list(self.lists[el.kids[0]])
        if k == E_TRAVERSE:
            return el.n < _UNBOUND_OP_START and self._bound_list(self.lists[el.kids[0]])
        return False

    # --- matching ---------------------------------------------------------------------------

    def parse(mut self, name: String, input: String) raises -> Result:
        var items = self.lists[self.named[name]].copy()
        var st = _State()
        var fail = Failure()
        var bytes = List[UInt8]()
        for b in input.as_bytes():
            bytes.append(b)
        var labels = List[String]()
        var res = Result()
        if self._run_list(items, st, labels, bytes, fail):
            res.ok = True
            res.acc = st.acc.copy()
            res.rest = _tail(bytes, st.pos)
            res.line = st.line
            res.line_off = st.line_off
            res.offset = st.pos
        else:
            res.reason = fail.reason
            res.rest = _tail(bytes, fail.pos)
            res.line = fail.line
            res.line_off = fail.line_off
            res.offset = fail.pos
        return res^

    def _fail(
        self, mut fail: Failure, items: List[Int], labels: List[String], positive: Bool, at: _State
    ):
        fail.reason = self._reason(items, labels, positive)
        fail.pos = at.pos
        fail.line = at.line
        fail.line_off = at.line_off

    def _run_list(
        mut self, items: List[Int], mut st: _State, labels: List[String], input: List[UInt8], mut fail: Failure
    ) raises -> Bool:
        var i = 0
        while i < len(items):
            if self._bound(items[i]):
                var j = i
                while j < len(items) and self._bound(items[j]):
                    var was_eos = self.elems[items[j]].kind == E_EOS
                    j += 1
                    if was_eos:
                        # a direct `eos()` closes the run, and so do the `eos()` that follow it
                        while j < len(items) and self.elems[items[j]].kind == E_EOS:
                            j += 1
                        break
                var run = List[Int]()
                for k in range(i, j):
                    run.append(items[k])
                var start = st.copy()
                self.eos_seen = False
                var matched = self._match_bound(run, st, input)
                if matched and self.eos_seen and st.pos != len(input):
                    matched = False
                if not matched:
                    st = start.copy()
                    self._fail(fail, run, labels, True, st)
                    return False
                i = j
            else:
                if not self._run_elem(items[i], st, labels, input, fail):
                    return False
                i += 1
        return True

    def _match_bound(mut self, run: List[Int], mut st: _State, input: List[UInt8]) raises -> Bool:
        for e in run:
            if not self._match_bound_elem(e, st, input):
                return False
        return True

    def _advance(self, mut st: _State, n: Int, input: List[UInt8]):
        for _ in range(n):
            st.pos += 1
            if input[st.pos - 1] == 0x0A:
                st.line += 1
                st.line_off = st.pos

    def _match_bound_elem(mut self, e: Int, mut st: _State, input: List[UInt8]) raises -> Bool:
        var el = self.elems[e].copy()
        var k = el.kind
        if k == E_STRING:
            var sb = el.s.as_bytes()
            if st.pos + len(sb) > len(input):
                return False
            for q in range(len(sb)):
                if input[st.pos + q] != sb[q]:
                    return False
            self._advance(st, len(sb), input)
            var t = self.t_bin(el.s)
            st.acc.append(t)
            return True
        if k == E_CHAR:
            if st.pos >= len(input):
                return False
            var cp = 0
            var width = 1
            if el.utf8:
                var lead = Int(input[st.pos])
                if lead < 0x80:
                    cp = lead
                else:
                    if lead < 0xC2 or lead > 0xF4:
                        return False
                    width = 2 if lead < 0xE0 else (3 if lead < 0xF0 else 4)
                    if st.pos + width > len(input):
                        return False
                    var piece = List[UInt8]()
                    for q in range(width):
                        piece.append(input[st.pos + q])
                    try:
                        cp = Int(ord(String(from_utf8=piece)))
                    except:
                        return False
            else:
                cp = Int(input[st.pos])
            if not _class_matches(el, cp):
                return False
            self._advance(st, width, input)
            st.acc.append(self.t_int(cp))
            return True
        if k == E_BYTES:
            if st.pos + el.n > len(input):
                return False
            var piece = List[UInt8]()
            for q in range(el.n):
                piece.append(input[st.pos + q])
            st.pos += el.n
            st.acc.append(self.t_bin(String(unsafe_from_utf8=piece)))
            return True
        if k == E_EOS:
            # NimbleParsec folds `eos()` into the end of the bound run's binary pattern
            self.eos_seen = True
            return True
        if k == E_LABEL:
            return self._match_bound(self.lists[el.kids[0]].copy(), st, input)
        # a bound traverse
        var inner = _State()
        inner.pos = st.pos
        inner.line = st.line
        inner.line_off = st.line_off
        if not self._match_bound(self.lists[el.kids[0]].copy(), inner, input):
            return False
        st.pos = inner.pos
        st.line = inner.line
        st.line_off = inner.line_off
        self._apply_op(el, inner.acc, st)
        return True

    def _apply_op(mut self, el: Elem, user_acc: List[Int], mut st: _State):
        """`user_acc` is the traversed combinator's output in emission order; results are appended."""
        var op = el.n
        if op == OP_IGNORE:
            return
        if op == OP_REPLACE:
            st.acc.append(el.value)
            return
        var natural = user_acc.copy()
        if op == OP_WRAP:
            st.acc.append(self.t_list(natural))
        elif op == OP_TAG:
            var pair = List[Int]()
            pair.append(self.t_atom(el.s))
            pair.append(self.t_list(natural))
            st.acc.append(self.t_tuple(pair))
        elif op == OP_UNWRAP_TAG:
            var pair = List[Int]()
            pair.append(self.t_atom(el.s))
            pair.append(natural[0] if len(natural) == 1 else self.t_list(natural))
            st.acc.append(self.t_tuple(pair))
        elif op == OP_LINE:
            var pair = List[Int]()
            pair.append(self.t_list(natural))
            var ln = List[Int]()
            ln.append(self.t_int(st.line))
            ln.append(self.t_int(st.line_off))
            pair.append(self.t_tuple(ln))
            st.acc.append(self.t_tuple(pair))
        elif op == OP_BYTE_OFFSET:
            var pair = List[Int]()
            pair.append(self.t_list(natural))
            pair.append(self.t_int(st.pos))
            st.acc.append(self.t_tuple(pair))
        elif op == OP_DIGITS:
            var v = 0
            for t in natural:
                v = v * 10 + (self.terms[t].i - 0x30)
            st.acc.append(self.t_int(v))
        elif op == OP_FOLD_INT:
            var v = 0
            var inner_items = self.lists[el.kids[0]].copy()
            var has_head = len(inner_items) > 0 and self.elems[inner_items[0]].kind == E_TRAVERSE and self.elems[inner_items[0]].n == OP_DIGITS
            var first = True
            for t in natural:
                if first:
                    v = self.terms[t].i
                    first = False
                    if not has_head:
                        v = v - 0x30
                else:
                    v = (self.terms[t].i - 0x30) + v * 10
            st.acc.append(self.t_int(v))
        elif op == OP_TO_STRING:
            var out = List[UInt8]()
            for t in natural:
                if self.terms[t].kind == T_BIN:
                    for b in self.terms[t].s.as_bytes():
                        out.append(b)
                else:
                    for b in chr(self.terms[t].i).as_bytes():
                        out.append(b)
            st.acc.append(self.t_bin(String(unsafe_from_utf8=out)))
        elif op == OP_REDUCE_SUM:
            var v = 0
            for t in natural:
                v += self.terms[t].i
            st.acc.append(self.t_int(v))
        elif op == OP_MAP_INC:
            for t in natural:
                st.acc.append(self.t_int(self.terms[t].i + 1))
        elif op == OP_POST_COUNT:
            st.acc.append(self.t_int(len(natural)))

    def _run_elem(
        mut self, e: Int, mut st: _State, labels: List[String], input: List[UInt8], mut fail: Failure
    ) raises -> Bool:
        var el = self.elems[e].copy()
        var k = el.kind
        if k == E_LABEL:
            var l2 = List[String]()
            l2.append(el.s)
            for x in labels:
                l2.append(x)
            return self._run_list(self.lists[el.kids[0]].copy(), st, l2, input, fail)
        if k == E_TRAVERSE:
            var inner = _State()
            inner.pos = st.pos
            inner.line = st.line
            inner.line_off = st.line_off
            if not self._run_list(self.lists[el.kids[0]].copy(), inner, labels, input, fail):
                return False
            st.pos = inner.pos
            st.line = inner.line
            st.line_off = inner.line_off
            self._apply_op(el, inner.acc, st)
            return True
        if k == E_CALL:
            var none = List[String]()
            return self._run_list(self.lists[self.named[el.s]].copy(), st, none, input, fail)
        if k == E_CHOICE:
            var l2 = labels.copy()
            if len(l2) == 0:
                l2.append(self._elem_label(e))
            var all_bound = True
            var has_empty = False
            for alt in el.kids:
                if not self._bound_list(self.lists[alt]):
                    all_bound = False
                if len(self.lists[alt]) == 0:
                    has_empty = True
            var start = st.copy()
            for ai in range(len(el.kids)):
                var alt = self.lists[el.kids[ai]].copy()
                var attempt = start.copy()
                if self._run_list(alt, attempt, l2, input, fail):
                    st = attempt^
                    return True
            if all_bound:
                var only = List[Int]()
                only.append(e)
                self._fail(fail, only, labels, True, start)
            return False
        if k == E_REPEAT:
            var inner_items = self.lists[el.kids[0]].copy()
            while True:
                var attempt = st.copy()
                var scratch = Failure()
                if not self._run_list(inner_items, attempt, labels, input, scratch):
                    break
                if attempt.pos == st.pos:
                    st = attempt^
                    break
                st = attempt^
            return True
        if k == E_TIMES:
            var inner_items = self.lists[el.kids[0]].copy()
            for _ in range(el.n):
                var attempt = st.copy()
                var scratch = Failure()
                if not self._run_list(inner_items, attempt, labels, input, scratch):
                    break
                st = attempt^
            return True
        if k == E_EVENTUALLY:
            var inner_items = self.lists[el.kids[0]].copy()
            while True:
                var attempt = st.copy()
                var scratch = Failure()
                if self._run_list(inner_items, attempt, labels, input, scratch):
                    st = attempt^
                    return True
                if st.pos >= len(input):
                    self._fail(fail, inner_items, labels, True, st)
                    return False
                self._advance(st, 1, input)
        if k == E_SLICE:
            var char_items = self.lists[el.kids[0]].copy()
            var start = st.copy()
            var taken = 0
            while el.m < 0 or taken < el.m:
                var attempt = st.copy()
                if not self._match_bound(char_items, attempt, input):
                    break
                # `_match_bound` pushed the codepoint; the slice outputs one binary instead
                _ = attempt.acc.pop()
                st.pos = attempt.pos
                st.line = attempt.line
                st.line_off = attempt.line_off
                taken += 1
            if taken < el.n:
                st = start.copy()
                self._fail(fail, self.lists[el.kids[1]], labels, True, start)
                return False
            var piece = List[UInt8]()
            for q in range(start.pos, st.pos):
                piece.append(input[q])
            st.acc.append(self.t_bin(String(unsafe_from_utf8=piece)))
            return True
        if k == E_LOOKAHEAD:
            var inner_items = self.lists[el.kids[0]].copy()
            var attempt = st.copy()
            var scratch = Failure()
            var matched = self._run_list(inner_items, attempt, labels, input, scratch)
            if matched == el.positive:
                return True
            self._fail(fail, inner_items, labels, el.positive, st)
            return False
        return False


    # --- recognising --------------------------------------------------------------------------
    # `scan` answers "how far does this parser reach from here" without building terms. It is what
    # the tokenizer's pre-tokenizer runs per word, so it borrows the tree instead of copying it.

    def scan(self, name: String, input: List[UInt8], pos: Int, unicode: Unicode, context: String) raises -> Int:
        """End offset of the match of parser `name` at `pos`, or -1 when it does not match."""
        return self._scan_list(self.lists[self.named[name]], pos, input, unicode, context)

    def _scan_list(
        self, items: List[Int], pos_in: Int, input: List[UInt8], unicode: Unicode, context: String
    ) raises -> Int:
        var pos = pos_in
        for e in items:
            pos = self._scan_elem(e, pos, input, unicode, context)
            if pos < 0:
                return -1
        return pos

    def _scan_elem(
        self, e: Int, pos: Int, input: List[UInt8], unicode: Unicode, context: String
    ) raises -> Int:
        ref el = self.elems[e]
        var k = el.kind
        if k == E_STRING:
            var sb = el.s.as_bytes()
            if pos + len(sb) > len(input):
                return -1
            for q in range(len(sb)):
                if input[pos + q] != sb[q]:
                    return -1
            return pos + len(sb)
        if k == E_CHAR:
            if pos >= len(input):
                return -1
            if el.utf8:
                var span = utf8_codepoint_at(input, pos, context)
                if not _class_matches(el, span.value):
                    return -1
                return pos + span.length
            if not _class_matches(el, Int(input[pos])):
                return -1
            return pos + 1
        if k == E_UCLASS:
            if pos >= len(input):
                return -1
            var span = utf8_codepoint_at(input, pos, context)
            var cp = span.value
            var cats = 0
            if unicode.is_letter(cp):
                cats |= U_LETTER
            if unicode.is_mark(cp):
                cats |= U_MARK
            if unicode.is_number(cp):
                cats |= U_NUMBER
            if unicode.is_whitespace(cp):
                cats |= U_SPACE
            if cp == 0x0A or cp == 0x0D:
                cats |= U_NEWLINE
            if el.n != 0 and (cats & el.n) == 0:
                return -1
            if (cats & el.m) != 0:
                return -1
            return pos + span.length
        if k == E_EOS:
            return pos if pos == len(input) else -1
        if k == E_LABEL or k == E_TRAVERSE:
            return self._scan_list(self.lists[el.kids[0]], pos, input, unicode, context)
        if k == E_CALL:
            return self._scan_list(self.lists[self.named[el.s]], pos, input, unicode, context)
        if k == E_CHOICE:
            for alt in el.kids:
                var end = self._scan_list(self.lists[alt], pos, input, unicode, context)
                if end >= 0:
                    return end
            return -1
        if k == E_REPEAT:
            var at = pos
            while True:
                var end = self._scan_list(self.lists[el.kids[0]], at, input, unicode, context)
                if end < 0:
                    break
                if end == at:
                    break
                at = end
            return at
        if k == E_TIMES:
            var at = pos
            for _ in range(el.n):
                var end = self._scan_list(self.lists[el.kids[0]], at, input, unicode, context)
                if end < 0:
                    break
                at = end
            return at
        if k == E_LOOKAHEAD:
            var end = self._scan_list(self.lists[el.kids[0]], pos, input, unicode, context)
            return pos if (end >= 0) == el.positive else -1
        raise Error("Parser.scan does not support this combinator")


def _newline_possible(inc: List[Int], exc: List[Int]) -> Bool:
    var allowed = len(inc) == 0
    var i = 0
    while i < len(inc):
        if inc[i] <= 10 and 10 <= inc[i + 1]:
            allowed = True
        i += 2
    var forbidden = False
    var j = 0
    while j < len(exc):
        if exc[j] <= 10 and 10 <= exc[j + 1]:
            forbidden = True
        j += 2
    return allowed and not forbidden


def _printable(c: Int) -> Bool:
    """`List.ascii_printable?/1`: 0x20..0x7E plus bell, backspace, tab, newline, vtab, formfeed, return, escape."""
    return (c >= 0x20 and c <= 0x7E) or (c >= 0x07 and c <= 0x0D) or c == 0x1B


def _class_matches(el: Elem, cp: Int) -> Bool:
    var ok = len(el.inc) == 0
    var i = 0
    while i < len(el.inc):
        if cp >= el.inc[i] and cp <= el.inc[i + 1]:
            ok = True
            break
        i += 2
    if not ok:
        return False
    var j = 0
    while j < len(el.exc):
        if cp >= el.exc[j] and cp <= el.exc[j + 1]:
            return False
        j += 2
    return True


def _tail(bytes: List[UInt8], pos: Int) -> String:
    var out = List[UInt8]()
    for i in range(pos, len(bytes)):
        out.append(bytes[i])
    return String(unsafe_from_utf8=out)


def _hex_str(s: String) -> String:
    var digits = "0123456789abcdef"
    var out = String()
    for b in s.as_bytes():
        out += digits[byte=Int(b >> 4) : Int(b >> 4) + 1]
        out += digits[byte=Int(b & 15) : Int(b & 15) + 1]
    return out^
