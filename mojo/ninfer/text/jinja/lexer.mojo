"""Jinja lexer; mirrors `third_party/llama-jinja/jinja/lexer.{h,cpp}` (the NInfer fork of
llama.cpp's `common/jinja`).

Same token table, same chat-template whitespace rules (`lstrip_blocks` and `trim_blocks` on,
`-` trimming, a single trailing newline dropped, CRLF folded to LF) and the same error text.
The source is handled as bytes, as in C++: `src[pos]` one past the end reads as NUL.
"""

from std.collections import List

# token types, in the C++ enum order
comptime T_EOF = 0
comptime T_TEXT = 1
comptime T_NUMERIC_LITERAL = 2
comptime T_STRING_LITERAL = 3
comptime T_IDENTIFIER = 4
comptime T_EQUALS = 5
comptime T_OPEN_PAREN = 6
comptime T_CLOSE_PAREN = 7
comptime T_OPEN_STATEMENT = 8
comptime T_CLOSE_STATEMENT = 9
comptime T_OPEN_EXPRESSION = 10
comptime T_CLOSE_EXPRESSION = 11
comptime T_OPEN_SQUARE = 12
comptime T_CLOSE_SQUARE = 13
comptime T_OPEN_CURLY = 14
comptime T_CLOSE_CURLY = 15
comptime T_COMMA = 16
comptime T_DOT = 17
comptime T_COLON = 18
comptime T_PIPE = 19
comptime T_CALL_OPERATOR = 20
comptime T_ADDITIVE = 21
comptime T_MULTIPLICATIVE = 22
comptime T_COMPARISON = 23
comptime T_UNARY = 24
comptime T_COMMENT = 25


def type_to_string(t: Int) -> String:
    var names = [
        "eof", "text", "numeric_literal", "string_literal", "identifier", "equals", "open_paren",
        "close_paren", "open_statement", "close_statement", "open_expression", "close_expression",
        "open_square_bracket", "close_square_bracket", "open_curly_bracket", "close_curly_bracket",
        "comma", "dot", "colon", "pipe", "call_operator", "additive_binary_operator",
        "multiplicative_binary_operator", "comparison_binary_operator", "unary_operator", "comment",
    ]
    if t >= 0 and t < len(names):
        return String(names[t])
    return "unknown"


struct Token(Copyable, Movable):
    var t: Int
    var value: String
    var pos: Int

    def __init__(out self, t: Int, var value: String, pos: Int):
        self.t = t
        self.value = value^
        self.pos = pos


struct LexerResult(Movable):
    var tokens: List[Token]
    var source: List[UInt8]  # normalized source (CRLF folded, trailing newline dropped)

    def __init__(out self):
        self.tokens = List[Token]()
        self.source = List[UInt8]()


def _replace_newlines(text: List[UInt8]) -> String:
    """`peak_source` shows a newline as U+21B5."""
    var out = List[UInt8]()
    for b in text:
        if b == 0x0A:
            out.append(0xE2)
            out.append(0x86)
            out.append(0xB5)
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


def peak_source(source: List[UInt8], pos: Int, max_peak_chars: Int = 40) -> String:
    if len(source) == 0:
        return "(no source available)"
    var start = pos - max_peak_chars if pos >= max_peak_chars else 0
    var end = min(pos + max_peak_chars, len(source))
    var sub = List[UInt8]()
    for i in range(start, end):
        sub.append(source[i])
    var output = "..." + _replace_newlines(sub) + "...\n"
    output += " " * (pos - start + 3)
    return output + "^"


def fmt_error_with_source(tag: String, msg: String, source: List[UInt8], pos: Int) -> String:
    return tag + ": " + msg + "\n" + peak_source(source, pos)


def _is_word(c: UInt8) -> Bool:
    return (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F


def _is_digit(c: UInt8) -> Bool:
    return c >= 0x30 and c <= 0x39


def _is_space(c: UInt8) -> Bool:
    """`std::isspace` in the C locale."""
    return c == 0x20 or (c >= 0x09 and c <= 0x0D)


def _escape(c: UInt8) -> Int:
    if c == 0x6E:
        return 0x0A
    if c == 0x74:
        return 0x09
    if c == 0x72:
        return 0x0D
    if c == 0x62:
        return 0x08
    if c == 0x66:
        return 0x0C
    if c == 0x76:
        return 0x0B
    if c == 0x5C or c == 0x27 or c == 0x22:
        return Int(c)
    return -1


struct _Cursor(Movable):
    var src: List[UInt8]
    var original: List[UInt8]
    var pos: Int

    def __init__(out self, var src: List[UInt8], original: List[UInt8]):
        self.src = src^
        self.original = original.copy()
        self.pos = 0

    def at(self, i: Int) -> UInt8:
        return self.src[i] if i < len(self.src) else 0

    def next_pos_is(self, chars: List[UInt8], n: Int = 1) -> Bool:
        if self.pos + n >= len(self.src):
            return False
        for c in chars:
            if self.src[self.pos + n] == c:
                return True
        return False

    def fail(self, msg: String, pos: Int) -> Error:
        return Error(fmt_error_with_source("lexer", msg, self.original, pos))

    def consume_while(mut self, kind: Int, quote: UInt8 = 0) raises -> List[UInt8]:
        """kind: 0 whitespace, 1 word, 2 digits, 3 'not the quote'."""
        var out = List[UInt8]()
        while True:
            var c = self.at(self.pos)
            var ok: Bool
            if kind == 0:
                ok = _is_space(c)
            elif kind == 1:
                ok = _is_word(c)
            elif kind == 2:
                ok = _is_digit(c)
            else:
                ok = c != quote
            if not ok:
                break
            if self.src[self.pos] == 0x5C if self.pos < len(self.src) else False:
                self.pos += 1
                if self.pos >= len(self.src):
                    raise self.fail("unexpected end of input after escape character", self.pos)
                var escaped = self.src[self.pos]
                self.pos += 1
                var mapped = _escape(escaped)
                if mapped < 0:
                    raise self.fail("unknown escape character \\" + String(chr(Int(escaped))), self.pos)
                out.append(UInt8(mapped))
                continue
            out.append(self.at(self.pos))
            self.pos += 1
            if self.pos > len(self.src):
                raise self.fail("unexpected end of input during consume_while", self.pos)
        return out^

    def consume_numeric(mut self) raises -> List[UInt8]:
        var num = self.consume_while(2)
        if (
            self.pos < len(self.src)
            and self.src[self.pos] == 0x2E
            and self.pos + 1 < len(self.src)
            and _is_digit(self.src[self.pos + 1])
        ):
            self.pos += 1
            var frac = self.consume_while(2)
            num.append(0x2E)
            num.extend(frac^)
        return num^


def _str(b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=b)


def _lstrip_ws(mut text: List[UInt8]):
    var i = 0
    while i < len(text) and (text[i] == 0x20 or text[i] == 0x09 or text[i] == 0x0D or text[i] == 0x0A):
        i += 1
    if i > 0:
        var rest = List[UInt8]()
        for k in range(i, len(text)):
            rest.append(text[k])
        text = rest^


def _rstrip_ws(mut text: List[UInt8]):
    while len(text) > 0:
        var c = text[len(text) - 1]
        if c == 0x20 or c == 0x09 or c == 0x0D or c == 0x0A:
            _ = text.pop()
        else:
            break


def _table_seq() -> List[String]:
    var out = List[String]()
    for s in "{%- -%} {{- -}} {% %} {{ }} ( ) { } [ ] , . : | <= >= == != < > + - ~ * / % =".split(" "):
        out.append(String(s))
    return out^


def _table_type() -> List[Int]:
    var out = List[Int]()
    out.append(T_OPEN_STATEMENT)
    out.append(T_CLOSE_STATEMENT)
    out.append(T_OPEN_EXPRESSION)
    out.append(T_CLOSE_EXPRESSION)
    out.append(T_OPEN_STATEMENT)
    out.append(T_CLOSE_STATEMENT)
    out.append(T_OPEN_EXPRESSION)
    out.append(T_CLOSE_EXPRESSION)
    out.append(T_OPEN_PAREN)
    out.append(T_CLOSE_PAREN)
    out.append(T_OPEN_CURLY)
    out.append(T_CLOSE_CURLY)
    out.append(T_OPEN_SQUARE)
    out.append(T_CLOSE_SQUARE)
    out.append(T_COMMA)
    out.append(T_DOT)
    out.append(T_COLON)
    out.append(T_PIPE)
    for _ in range(6):
        out.append(T_COMPARISON)
    for _ in range(3):
        out.append(T_ADDITIVE)
    for _ in range(3):
        out.append(T_MULTIPLICATIVE)
    out.append(T_EQUALS)
    return out^


def tokenize(source: List[UInt8]) raises -> LexerResult:
    var result = LexerResult()
    if len(source) == 0:
        return result^
    # Normalize \r\n and \r to \n.
    var src = List[UInt8]()
    var i = 0
    while i < len(source):
        if source[i] == 0x0D and i + 1 < len(source) and source[i + 1] == 0x0A:
            i += 1
            continue
        src.append(0x0A if source[i] == 0x0D else source[i])
        i += 1
    if source[len(source) - 1] == 0x0A:
        _ = src.pop()
    var cur = _Cursor(src^, source)
    var table_seq = _table_seq()
    var table_type = _table_type()
    var tokens = List[Token]()
    var start_pos = 0
    var curly_depth = 0
    var is_lstrip_block = False
    var is_rstrip_block = False

    while cur.pos < len(cur.src):
        start_pos = cur.pos
        var last_type = T_CLOSE_STATEMENT if len(tokens) == 0 else tokens[len(tokens) - 1].t
        if last_type == T_CLOSE_STATEMENT or last_type == T_CLOSE_EXPRESSION or last_type == T_COMMENT:
            var last_block_can_rm_newline = False
            is_rstrip_block = False
            if cur.pos > 3:
                var c0 = cur.src[cur.pos - 3]
                var c1 = cur.src[cur.pos - 2]
                var c2 = cur.src[cur.pos - 1]
                is_rstrip_block = c0 == 0x2D and (c1 == 0x25 or c1 == 0x7D or c1 == 0x23) and c2 == 0x7D
                last_block_can_rm_newline = (c1 == 0x23 or c1 == 0x25 or c1 == 0x2D) and c2 == 0x7D
            var start = cur.pos
            var end = start
            var open_set = List[UInt8]()
            open_set.append(0x25)
            open_set.append(0x7B)
            open_set.append(0x23)
            while cur.pos < len(cur.src) and not (cur.src[cur.pos] == 0x7B and cur.next_pos_is(open_set)):
                cur.pos += 1
                end = cur.pos
            # lstrip_blocks: text\n[space]{block} --> text\n{block}
            var lstrip_set = List[UInt8]()
            lstrip_set.append(0x25)
            lstrip_set.append(0x23)
            lstrip_set.append(0x2D)
            if cur.at(cur.pos) == 0x7B and cur.next_pos_is(lstrip_set):
                var current = end
                while current > start:
                    var c = cur.src[current - 1]
                    if current == 1:
                        end = 0
                        break
                    if c == 0x0A:
                        end = current
                        break
                    if not _is_space(c):
                        break
                    current -= 1
            var text = List[UInt8]()
            for k in range(start, end):
                text.append(cur.src[k])
            if last_block_can_rm_newline:  # trim_blocks
                if len(text) > 0 and text[0] == 0x0A:
                    var rest = List[UInt8]()
                    for k in range(1, len(text)):
                        rest.append(text[k])
                    text = rest^
            if is_rstrip_block:
                _lstrip_ws(text)
            var brace_set = List[UInt8]()
            brace_set.append(0x7B)
            brace_set.append(0x25)
            brace_set.append(0x23)
            var dash = List[UInt8]()
            dash.append(0x2D)
            is_lstrip_block = cur.at(cur.pos) == 0x7B and cur.next_pos_is(brace_set) and cur.next_pos_is(dash, 2)
            if is_lstrip_block:
                _rstrip_ws(text)
            if len(text) > 0:
                tokens.append(Token(T_TEXT, _str(text), start_pos))
                continue

        # comment
        var hash_set = List[UInt8]()
        hash_set.append(0x23)
        if cur.at(cur.pos) == 0x7B and cur.next_pos_is(hash_set):
            start_pos = cur.pos
            cur.pos += 2
            var comment = List[UInt8]()
            var close = List[UInt8]()
            close.append(0x7D)
            while not (cur.at(cur.pos) == 0x23 and cur.next_pos_is(close)):
                if cur.pos + 2 >= len(cur.src):
                    raise cur.fail("missing end of comment tag", cur.pos)
                comment.append(cur.src[cur.pos])
                cur.pos += 1
            tokens.append(Token(T_COMMENT, _str(comment), start_pos))
            cur.pos += 2
            continue

        if cur.at(cur.pos) == 0x2D and (last_type == T_OPEN_EXPRESSION or last_type == T_OPEN_STATEMENT):
            cur.pos += 1
            if cur.pos >= len(cur.src):
                break

        _ = cur.consume_while(0)
        if cur.pos >= len(cur.src):
            break

        var ch = cur.src[cur.pos]
        var close_set = List[UInt8]()
        close_set.append(0x25)
        close_set.append(0x7D)
        var is_closing_block = ch == 0x2D and cur.next_pos_is(close_set)

        if not is_closing_block and (ch == 0x2D or ch == 0x2B):
            start_pos = cur.pos
            var last = T_EOF if len(tokens) == 0 else tokens[len(tokens) - 1].t
            if last == T_TEXT or last == T_EOF:
                raise cur.fail("unexpected character: " + String(chr(Int(ch))), cur.pos)
            if (
                last == T_IDENTIFIER
                or last == T_NUMERIC_LITERAL
                or last == T_STRING_LITERAL
                or last == T_CLOSE_PAREN
                or last == T_CLOSE_SQUARE
            ):
                pass  # binary operator: fall through to the table
            else:
                cur.pos += 1
                var num = cur.consume_numeric()
                var value = String(chr(Int(ch))) + _str(num)
                var t = T_UNARY if len(num) == 0 else T_NUMERIC_LITERAL
                tokens.append(Token(t, value^, start_pos))
                continue

        var matched = False
        for k in range(len(table_seq)):
            var seq = table_seq[k]
            start_pos = cur.pos
            if seq == "}}" and curly_depth > 0:
                continue
            var sb = seq.as_bytes()
            if cur.pos + len(sb) <= len(cur.src):
                var same = True
                for q in range(len(sb)):
                    if cur.src[cur.pos + q] != sb[q]:
                        same = False
                        break
                if same:
                    var typ = table_type[k]
                    tokens.append(Token(typ, seq, start_pos))
                    if typ == T_OPEN_EXPRESSION:
                        curly_depth = 0
                    elif typ == T_OPEN_CURLY:
                        curly_depth += 1
                    elif typ == T_CLOSE_CURLY:
                        curly_depth -= 1
                    cur.pos += len(sb)
                    matched = True
                    break
        if matched:
            continue

        if ch == 0x27 or ch == 0x22:
            start_pos = cur.pos
            cur.pos += 1
            var s = cur.consume_while(3, ch)
            tokens.append(Token(T_STRING_LITERAL, _str(s), start_pos))
            cur.pos += 1
            continue

        if _is_digit(ch):
            start_pos = cur.pos
            var num = cur.consume_numeric()
            tokens.append(Token(T_NUMERIC_LITERAL, _str(num), start_pos))
            continue

        if _is_word(ch):
            start_pos = cur.pos
            var word = cur.consume_while(1)
            tokens.append(Token(T_IDENTIFIER, _str(word), start_pos))
            continue

        raise cur.fail("unexpected character: " + String(chr(Int(ch))), cur.pos)

    result.tokens = tokens^
    result.source = cur.src.copy()
    return result^
