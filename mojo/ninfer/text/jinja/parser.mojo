"""Jinja parser; mirrors `third_party/llama-jinja/jinja/parser.cpp` and the statement
definitions of `runtime.h`.

The C++ AST is a class hierarchy; here it is one arena of `Node` records addressed by index,
which keeps the recursive tree flat and cheap to share with macros. The meaning of the
generic fields per kind is listed next to each constant.
"""

from ninfer.text.jinja.error import JinjaError, jinja_error, parse_float, parse_int
from std.collections import List

from ninfer.text.jinja.lexer import (
    Token,
    LexerResult,
    fmt_error_with_source,
    T_EOF,
    T_TEXT,
    T_NUMERIC_LITERAL,
    T_STRING_LITERAL,
    T_IDENTIFIER,
    T_EQUALS,
    T_OPEN_PAREN,
    T_CLOSE_PAREN,
    T_OPEN_STATEMENT,
    T_CLOSE_STATEMENT,
    T_OPEN_EXPRESSION,
    T_CLOSE_EXPRESSION,
    T_OPEN_SQUARE,
    T_CLOSE_SQUARE,
    T_OPEN_CURLY,
    T_CLOSE_CURLY,
    T_COMMA,
    T_DOT,
    T_COLON,
    T_PIPE,
    T_ADDITIVE,
    T_MULTIPLICATIVE,
    T_COMPARISON,
    T_COMMENT,
    type_to_string,
)

# node kinds -- fields used: a b c (child ids, -1 = none), l1 l2 (child lists), s, i, f, flag
comptime N_OUTPUT = 0  # a=operand
comptime N_IF = 1  # a=test l1=body l2=alternate
comptime N_FOR = 2  # a=loopvar b=iterable l1=body l2=default block
comptime N_BREAK = 3
comptime N_CONTINUE = 4
comptime N_NOOP = 5
comptime N_SET = 6  # a=assignee b=value l1=body
comptime N_MACRO = 7  # a=name l1=args l2=body
comptime N_COMMENT = 8
comptime N_BLANK = 9
comptime N_MEMBER = 10  # a=object b=property flag=computed
comptime N_CALL = 11  # a=callee l1=args
comptime N_IDENT = 12  # s
comptime N_INT = 13  # i
comptime N_FLOAT = 14  # f
comptime N_STRING = 15  # s
comptime N_ARRAY = 16  # l1
comptime N_TUPLE = 17  # l1
comptime N_OBJECT = 18  # l1=keys l2=values
comptime N_BINARY = 19  # s=op a=left b=right
comptime N_FILTER = 20  # a=operand b=filter
comptime N_FILTER_STMT = 21  # a=filter l1=body
comptime N_SELECT = 22  # a=lhs b=test
comptime N_TEST = 23  # a=operand flag=negate b=test
comptime N_UNARY = 24  # s=op a=argument
comptime N_SLICE = 25  # a=start b=stop c=step
comptime N_KWARG = 26  # a=key b=value
comptime N_SPREAD = 27  # a
comptime N_CALL_STMT = 28  # a=call l1=caller args l2=body
comptime N_TERNARY = 29  # a=condition b=true c=false

comptime EXPRESSION_KINDS_START = 100  # (documentation only)


struct Node(Copyable, Movable):
    var kind: Int
    var pos: Int
    var a: Int
    var b: Int
    var c: Int
    var l1: List[Int]
    var l2: List[Int]
    var s: String
    var i: Int
    var f: Float64
    var flag: Bool

    def __init__(out self, kind: Int, pos: Int):
        self.kind = kind
        self.pos = pos
        self.a = -1
        self.b = -1
        self.c = -1
        self.l1 = List[Int]()
        self.l2 = List[Int]()
        self.s = String()
        self.i = 0
        self.f = 0.0
        self.flag = False


def type_name(kind: Int) -> String:
    """`statement::type()` strings used in error messages."""
    if kind == N_OUTPUT:
        return "Output"
    if kind == N_IF:
        return "If"
    if kind == N_FOR:
        return "For"
    if kind == N_BREAK:
        return "Break"
    if kind == N_CONTINUE:
        return "Continue"
    if kind == N_NOOP:
        return "Noop"
    if kind == N_SET:
        return "Set"
    if kind == N_MACRO:
        return "Macro"
    if kind == N_COMMENT:
        return "Comment"
    if kind == N_BLANK:
        return "BlankExpression"
    if kind == N_MEMBER:
        return "MemberExpression"
    if kind == N_CALL:
        return "CallExpression"
    if kind == N_IDENT:
        return "Identifier"
    if kind == N_INT:
        return "IntegerLiteral"
    if kind == N_FLOAT:
        return "FloatLiteral"
    if kind == N_STRING:
        return "StringLiteral"
    if kind == N_ARRAY:
        return "ArrayLiteral"
    if kind == N_TUPLE:
        return "TupleLiteral"
    if kind == N_OBJECT:
        return "ObjectLiteral"
    if kind == N_BINARY:
        return "BinaryExpression"
    if kind == N_FILTER:
        return "FilterExpression"
    if kind == N_FILTER_STMT:
        return "FilterStatement"
    if kind == N_SELECT:
        return "SelectExpression"
    if kind == N_TEST:
        return "TestExpression"
    if kind == N_UNARY:
        return "UnaryExpression"
    if kind == N_SLICE:
        return "SliceExpression"
    if kind == N_KWARG:
        return "KeywordArgumentExpression"
    if kind == N_SPREAD:
        return "SpreadExpression"
    if kind == N_CALL_STMT:
        return "CallStatement"
    if kind == N_TERNARY:
        return "Ternary"
    return "Statement"


struct Ast(Copyable, Movable):
    var nodes: List[Node]
    var body: List[Int]  # top-level statements

    def __init__(out self):
        self.nodes = List[Node]()
        self.body = List[Int]()


struct _Parser(Movable):
    var tokens: List[Token]
    var current: Int
    var source: List[UInt8]
    var ast: Ast

    def __init__(out self, var tokens: List[Token], var source: List[UInt8]):
        self.tokens = tokens^
        self.current = 0
        self.source = source^
        self.ast = Ast()

    def finish(self) -> Ast:
        return self.ast.copy()

    # --- helpers ---------------------------------------------------------------------------

    def mk(mut self, start_pos: Int, kind: Int) -> Int:
        var n = Node(kind, self.tokens[start_pos].pos if start_pos < len(self.tokens) else 0)
        self.ast.nodes.append(n^)
        return len(self.ast.nodes) - 1

    def peek_type(self, offset: Int = 0) -> Int:
        if self.current + offset >= len(self.tokens):
            return T_EOF
        return self.tokens[self.current + offset].t

    def peek_value(self, offset: Int = 0) -> String:
        if self.current + offset >= len(self.tokens):
            return String()
        return self.tokens[self.current + offset].value

    def peek_pos(self, offset: Int = 0) -> Int:
        if self.current + offset >= len(self.tokens):
            return 0
        return self.tokens[self.current + offset].pos

    def next(mut self) raises JinjaError -> Token:
        if self.current >= len(self.tokens):
            raise jinja_error(
                fmt_error_with_source(
                    "parser",
                    "Parser Error: Unexpected EOF",
                    self.source,
                    0 if len(self.tokens) == 0 else self.tokens[len(self.tokens) - 1].pos,
                )
            )
        var t = self.tokens[self.current].copy()
        self.current += 1
        return t^

    def expect(mut self, type: Int, error: String) raises JinjaError:
        if self.peek_type() != type:
            raise jinja_error(
                fmt_error_with_source(
                    "parser",
                    "Parser Error: " + error + " (Got " + self.peek_value() + ")",
                    self.source,
                    self.peek_pos(),
                )
            )
        self.current += 1

    def expect_identifier(mut self, name: String) raises JinjaError:
        if self.peek_type() != T_IDENTIFIER or self.peek_value() != name:
            raise jinja_error(
                fmt_error_with_source("parser", "Expected identifier: " + name, self.source, self.peek_pos())
            )
        self.current += 1

    def is_(self, type: Int) -> Bool:
        return self.peek_type() == type

    def is_identifier(self, name: String) -> Bool:
        return self.peek_type() == T_IDENTIFIER and self.peek_value() == name

    def is_statement(self, names: List[String]) -> Bool:
        if self.peek_type(0) != T_OPEN_STATEMENT or self.peek_type(1) != T_IDENTIFIER:
            return False
        var v = self.peek_value(1)
        for n in names:
            if n == v:
                return True
        return False

    def one(self, a: String) -> List[String]:
        var l = List[String]()
        l.append(a)
        return l^

    # --- statements ------------------------------------------------------------------------

    def parse_any(mut self) raises JinjaError -> Int:
        var start = self.current
        var t = self.peek_type()
        if t == T_COMMENT:
            var v = self.next().value
            var n = self.mk(start, N_COMMENT)
            self.ast.nodes[n].s = v^
            return n
        if t == T_TEXT:
            var v = self.next().value
            var n = self.mk(start, N_STRING)
            self.ast.nodes[n].s = v^
            return n
        if t == T_OPEN_STATEMENT:
            return self.parse_jinja_statement()
        if t == T_OPEN_EXPRESSION:
            return self.parse_jinja_expression()
        raise jinja_error("Unexpected token type")

    def parse_jinja_expression(mut self) raises JinjaError -> Int:
        var pos = self.current
        self.expect(T_OPEN_EXPRESSION, "Expected {{")
        var result = self.parse_expression()
        self.expect(T_CLOSE_EXPRESSION, "Expected }}")
        var n = self.mk(pos, N_OUTPUT)
        self.ast.nodes[n].a = result
        return n

    def parse_body_until(mut self, names: List[String]) raises JinjaError -> List[Int]:
        var body = List[Int]()
        while not self.is_statement(names):
            body.append(self.parse_any())
        return body^

    def expect_end(mut self, name: String) raises JinjaError:
        self.expect(T_OPEN_STATEMENT, "Expected {%")
        self.expect_identifier(name)
        self.expect(T_CLOSE_STATEMENT, "Expected %}")

    def parse_jinja_statement(mut self) raises JinjaError -> Int:
        self.expect(T_OPEN_STATEMENT, "Expected {%")
        if self.peek_type() != T_IDENTIFIER:
            raise jinja_error("Unknown statement")
        var start = self.current
        var name = self.next().value
        var result = -1
        if name == "set":
            result = self.parse_set_statement(start)
        elif name == "if":
            result = self.parse_if_statement(start)
            self.expect_end("endif")
        elif name == "macro":
            result = self.parse_macro_statement(start)
            self.expect_end("endmacro")
        elif name == "for":
            result = self.parse_for_statement(start)
            self.expect_end("endfor")
        elif name == "break":
            self.expect(T_CLOSE_STATEMENT, "Expected %}")
            result = self.mk(start, N_BREAK)
        elif name == "continue":
            self.expect(T_CLOSE_STATEMENT, "Expected %}")
            result = self.mk(start, N_CONTINUE)
        elif name == "call":
            var caller_args = List[Int]()
            if self.is_(T_OPEN_PAREN):
                caller_args = self.parse_args()
            var callee = self.parse_primary_expression()
            if self.ast.nodes[callee].kind != N_IDENT:
                raise jinja_error("Expected identifier")
            var call_args = self.parse_args()
            self.expect(T_CLOSE_STATEMENT, "Expected %}")
            var names = List[String]()
            names.append("endcall")
            var body = self.parse_body_until(names)
            self.expect_end("endcall")
            var call_expr = self.mk(start, N_CALL)
            self.ast.nodes[call_expr].a = callee
            self.ast.nodes[call_expr].l1 = call_args^
            result = self.mk(start, N_CALL_STMT)
            self.ast.nodes[result].a = call_expr
            self.ast.nodes[result].l1 = caller_args^
            self.ast.nodes[result].l2 = body^
        elif name == "filter":
            var filter_node = self.parse_primary_expression()
            if self.ast.nodes[filter_node].kind == N_IDENT and self.is_(T_OPEN_PAREN):
                filter_node = self.parse_call_expression(filter_node)
            self.expect(T_CLOSE_STATEMENT, "Expected %}")
            var names = List[String]()
            names.append("endfilter")
            var body = self.parse_body_until(names)
            self.expect_end("endfilter")
            result = self.mk(start, N_FILTER_STMT)
            self.ast.nodes[result].a = filter_node
            self.ast.nodes[result].l1 = body^
        elif name == "generation" or name == "endgeneration":
            result = self.mk(start, N_NOOP)
            self.current += 1
        else:
            raise jinja_error("Unknown statement: " + name)
        return result

    def parse_set_statement(mut self, start: Int) raises JinjaError -> Int:
        var left = self.parse_expression_sequence()
        var value = -1
        var body = List[Int]()
        if self.is_(T_EQUALS):
            self.current += 1
            value = self.parse_expression_sequence()
        else:
            self.expect(T_CLOSE_STATEMENT, "Expected %}")
            var names = List[String]()
            names.append("endset")
            body = self.parse_body_until(names)
            self.expect(T_OPEN_STATEMENT, "Expected {%")
            self.expect_identifier("endset")
        self.expect(T_CLOSE_STATEMENT, "Expected %}")
        var n = self.mk(start, N_SET)
        self.ast.nodes[n].a = left
        self.ast.nodes[n].b = value
        self.ast.nodes[n].l1 = body^
        return n

    def parse_if_statement(mut self, start: Int) raises JinjaError -> Int:
        var test = self.parse_expression()
        self.expect(T_CLOSE_STATEMENT, "Expected %}")
        var stops = List[String]()
        stops.append("elif")
        stops.append("else")
        stops.append("endif")
        var body = self.parse_body_until(stops)
        var alternate = List[Int]()
        var elif_names = self.one("elif")
        var else_names = self.one("else")
        var endif_names = self.one("endif")
        if self.is_statement(elif_names):
            var pos0 = self.current
            self.current += 2
            alternate.append(self.parse_if_statement(pos0))
        elif self.is_statement(else_names):
            self.current += 2
            self.expect(T_CLOSE_STATEMENT, "Expected %}")
            alternate = self.parse_body_until(endif_names)
        var n = self.mk(start, N_IF)
        self.ast.nodes[n].a = test
        self.ast.nodes[n].l1 = body^
        self.ast.nodes[n].l2 = alternate^
        return n

    def parse_macro_statement(mut self, start: Int) raises JinjaError -> Int:
        var name = self.parse_primary_expression()
        var args = self.parse_args()
        self.expect(T_CLOSE_STATEMENT, "Expected %}")
        var names = self.one("endmacro")
        var body = self.parse_body_until(names)
        var n = self.mk(start, N_MACRO)
        self.ast.nodes[n].a = name
        self.ast.nodes[n].l1 = args^
        self.ast.nodes[n].l2 = body^
        return n

    def parse_expression_sequence(mut self, primary: Bool = False) raises JinjaError -> Int:
        var start = self.current
        var exprs = List[Int]()
        exprs.append(self.parse_primary_expression() if primary else self.parse_expression())
        var is_tuple = self.is_(T_COMMA)
        while self.is_(T_COMMA):
            self.current += 1
            exprs.append(self.parse_primary_expression() if primary else self.parse_expression())
        if is_tuple:
            var n = self.mk(start, N_TUPLE)
            self.ast.nodes[n].l1 = exprs^
            return n
        return exprs[0]

    def parse_for_statement(mut self, start: Int) raises JinjaError -> Int:
        var loop_var = self.parse_expression_sequence(True)
        if not self.is_identifier("in"):
            raise jinja_error("Expected 'in'")
        self.current += 1
        var iterable = self.parse_expression()
        self.expect(T_CLOSE_STATEMENT, "Expected %}")
        var stops = List[String]()
        stops.append("endfor")
        stops.append("else")
        var body = self.parse_body_until(stops)
        var alternate = List[Int]()
        var else_names = self.one("else")
        if self.is_statement(else_names):
            self.current += 2
            self.expect(T_CLOSE_STATEMENT, "Expected %}")
            var endfor = self.one("endfor")
            alternate = self.parse_body_until(endfor)
        var n = self.mk(start, N_FOR)
        self.ast.nodes[n].a = loop_var
        self.ast.nodes[n].b = iterable
        self.ast.nodes[n].l1 = body^
        self.ast.nodes[n].l2 = alternate^
        return n

    # --- expressions -----------------------------------------------------------------------

    def parse_expression(mut self) raises JinjaError -> Int:
        return self.parse_if_expression()

    def parse_if_expression(mut self) raises JinjaError -> Int:
        var a = self.parse_logical_or_expression()
        if self.is_identifier("if"):
            var start = self.current
            self.current += 1
            var test = self.parse_logical_or_expression()
            if self.is_identifier("else"):
                var pos0 = self.current
                self.current += 1
                var false_expr = self.parse_if_expression()
                var n = self.mk(pos0, N_TERNARY)
                self.ast.nodes[n].a = test
                self.ast.nodes[n].b = a
                self.ast.nodes[n].c = false_expr
                return n
            var n = self.mk(start, N_SELECT)
            self.ast.nodes[n].a = a
            self.ast.nodes[n].b = test
            return n
        return a

    def binary(mut self, start: Int, op: String, left: Int, right: Int) -> Int:
        var n = self.mk(start, N_BINARY)
        self.ast.nodes[n].s = op
        self.ast.nodes[n].a = left
        self.ast.nodes[n].b = right
        return n

    def parse_logical_or_expression(mut self) raises JinjaError -> Int:
        var left = self.parse_logical_and_expression()
        while self.is_identifier("or"):
            var start = self.current
            var op = self.next().value
            var right = self.parse_logical_and_expression()
            left = self.binary(start, op, left, right)
        return left

    def parse_logical_and_expression(mut self) raises JinjaError -> Int:
        var left = self.parse_logical_negation_expression()
        while self.is_identifier("and"):
            var start = self.current
            var op = self.next().value
            var right = self.parse_logical_negation_expression()
            left = self.binary(start, op, left, right)
        return left

    def parse_logical_negation_expression(mut self) raises JinjaError -> Int:
        if self.is_identifier("not"):
            var start = self.current
            var op = self.next().value
            var arg = self.parse_logical_negation_expression()
            var n = self.mk(start, N_UNARY)
            self.ast.nodes[n].s = op
            self.ast.nodes[n].a = arg
            return n
        return self.parse_comparison_expression()

    def parse_comparison_expression(mut self) raises JinjaError -> Int:
        var left = self.parse_additive_expression()
        while True:
            var op = String()
            var start = self.current
            if self.is_identifier("not") and self.peek_type(1) == T_IDENTIFIER and self.peek_value(1) == "in":
                op = "not in"
                self.current += 2
            elif self.is_identifier("in"):
                op = self.next().value
            elif self.is_(T_COMPARISON):
                op = self.next().value
            else:
                break
            var right = self.parse_additive_expression()
            left = self.binary(start, op, left, right)
        return left

    def parse_additive_expression(mut self) raises JinjaError -> Int:
        var left = self.parse_multiplicative_expression()
        while self.is_(T_ADDITIVE):
            var start = self.current
            var op = self.next().value
            var right = self.parse_multiplicative_expression()
            left = self.binary(start, op, left, right)
        return left

    def parse_multiplicative_expression(mut self) raises JinjaError -> Int:
        var left = self.parse_test_expression()
        while self.is_(T_MULTIPLICATIVE):
            var start = self.current
            var op = self.next().value
            var right = self.parse_test_expression()
            left = self.binary(start, op, left, right)
        return left

    def parse_test_expression(mut self) raises JinjaError -> Int:
        var operand = self.parse_filter_expression()
        while self.is_identifier("is"):
            var start = self.current
            self.current += 1
            var negate = False
            if self.is_identifier("not"):
                self.current += 1
                negate = True
            var test_id = self.parse_primary_expression()
            if self.is_(T_OPEN_PAREN):
                test_id = self.parse_call_expression(test_id)
            var n = self.mk(start, N_TEST)
            self.ast.nodes[n].a = operand
            self.ast.nodes[n].flag = negate
            self.ast.nodes[n].b = test_id
            operand = n
        return operand

    def parse_filter_expression(mut self) raises JinjaError -> Int:
        var operand = self.parse_call_member_expression()
        while self.is_(T_PIPE):
            var start = self.current
            self.current += 1
            var filter = self.parse_primary_expression()
            if self.is_(T_OPEN_PAREN):
                filter = self.parse_call_expression(filter)
            var n = self.mk(start, N_FILTER)
            self.ast.nodes[n].a = operand
            self.ast.nodes[n].b = filter
            operand = n
        return operand

    def parse_call_member_expression(mut self) raises JinjaError -> Int:
        var primary = self.parse_primary_expression()
        var member = self.parse_member_expression(primary)
        if self.is_(T_OPEN_PAREN):
            return self.parse_call_expression(member)
        return member

    def parse_call_expression(mut self, callee: Int) raises JinjaError -> Int:
        var start = self.current
        var args = self.parse_args()
        var expr = self.mk(start, N_CALL)
        self.ast.nodes[expr].a = callee
        self.ast.nodes[expr].l1 = args^
        var member = self.parse_member_expression(expr)
        if self.is_(T_OPEN_PAREN):
            return self.parse_call_expression(member)
        return member

    def parse_args(mut self) raises JinjaError -> List[Int]:
        self.expect(T_OPEN_PAREN, "Expected (")
        var args = List[Int]()
        while not self.is_(T_CLOSE_PAREN):
            var arg = -1
            if self.peek_type() == T_MULTIPLICATIVE and self.peek_value() == "*":
                var start = self.current
                self.current += 1
                var inner = self.parse_expression()
                arg = self.mk(start, N_SPREAD)
                self.ast.nodes[arg].a = inner
            else:
                arg = self.parse_expression()
                if self.is_(T_EQUALS):
                    var start = self.current
                    self.current += 1
                    var value = self.parse_expression()
                    var k = self.mk(start, N_KWARG)
                    self.ast.nodes[k].a = arg
                    self.ast.nodes[k].b = value
                    arg = k
            args.append(arg)
            if self.is_(T_COMMA):
                self.current += 1
        self.expect(T_CLOSE_PAREN, "Expected )")
        return args^

    def parse_member_expression(mut self, object: Int) raises JinjaError -> Int:
        var start = self.current
        var obj = object
        while self.is_(T_DOT) or self.is_(T_OPEN_SQUARE):
            var op = self.next()
            var computed = op.t == T_OPEN_SQUARE
            var prop = -1
            if computed:
                prop = self.parse_member_expression_arguments()
                self.expect(T_CLOSE_SQUARE, "Expected ]")
            else:
                prop = self.parse_primary_expression()
            var n = self.mk(start, N_MEMBER)
            self.ast.nodes[n].a = obj
            self.ast.nodes[n].b = prop
            self.ast.nodes[n].flag = computed
            obj = n
        return obj

    def parse_member_expression_arguments(mut self) raises JinjaError -> Int:
        var slices = List[Int]()
        var is_slice = False
        var start = self.current
        while not self.is_(T_CLOSE_SQUARE):
            if self.is_(T_COLON):
                slices.append(-1)
                self.current += 1
                is_slice = True
            else:
                slices.append(self.parse_expression())
                if self.is_(T_COLON):
                    self.current += 1
                    is_slice = True
        if is_slice:
            var n = self.mk(start, N_SLICE)
            self.ast.nodes[n].a = slices[0] if len(slices) > 0 else -1
            self.ast.nodes[n].b = slices[1] if len(slices) > 1 else -1
            self.ast.nodes[n].c = slices[2] if len(slices) > 2 else -1
            return n
        if len(slices) == 0:
            return self.mk(start, N_BLANK)
        return slices[0]

    def parse_primary_expression(mut self) raises JinjaError -> Int:
        var start = self.current
        var t = self.next()
        if t.t == T_NUMERIC_LITERAL:
            if t.value.find(".") >= 0:
                var n = self.mk(start, N_FLOAT)
                self.ast.nodes[n].f = parse_float(t.value)
                return n
            var n = self.mk(start, N_INT)
            self.ast.nodes[n].i = parse_int(t.value)
            return n
        if t.t == T_STRING_LITERAL:
            var val = t.value
            while self.is_(T_STRING_LITERAL):
                val += self.next().value
            var n = self.mk(start, N_STRING)
            self.ast.nodes[n].s = val^
            return n
        if t.t == T_IDENTIFIER:
            var n = self.mk(start, N_IDENT)
            self.ast.nodes[n].s = t.value
            return n
        if t.t == T_OPEN_PAREN:
            var expr = self.parse_expression_sequence()
            self.expect(T_CLOSE_PAREN, "Expected )")
            return expr
        if t.t == T_OPEN_SQUARE:
            var vals = List[Int]()
            while not self.is_(T_CLOSE_SQUARE):
                vals.append(self.parse_expression())
                if self.is_(T_COMMA):
                    self.current += 1
            self.current += 1
            var n = self.mk(start, N_ARRAY)
            self.ast.nodes[n].l1 = vals^
            return n
        if t.t == T_OPEN_CURLY:
            var keys = List[Int]()
            var vals = List[Int]()
            while not self.is_(T_CLOSE_CURLY):
                var key = self.parse_expression()
                self.expect(T_COLON, "Expected :")
                keys.append(key)
                vals.append(self.parse_expression())
                if self.is_(T_COMMA):
                    self.current += 1
            self.current += 1
            var n = self.mk(start, N_OBJECT)
            self.ast.nodes[n].l1 = keys^
            self.ast.nodes[n].l2 = vals^
            return n
        raise jinja_error("Unexpected token: " + t.value + " of type " + String(t.t))


def parse_from_tokens(lexed: LexerResult) raises JinjaError -> Ast:
    var parser = _Parser(lexed.tokens.copy(), lexed.source.copy())
    while parser.current < len(parser.tokens):
        var stmt = parser.parse_any()
        parser.ast.body.append(stmt)
    return parser.finish()
