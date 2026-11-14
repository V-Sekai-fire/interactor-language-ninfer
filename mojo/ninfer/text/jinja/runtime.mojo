"""Jinja executor; mirrors `third_party/llama-jinja/jinja/runtime.{h,cpp}`."""

from ninfer.text.jinja.error import JinjaError, jinja_error, K_ERROR, K_BREAK, K_CONTINUE, K_RETHROWN
from std.collections import List

from ninfer.text.jinja.jstring import JString, JU, characters, codepoint_length
from ninfer.text.jinja.lexer import peak_source
from ninfer.text.jinja.parser import (
    Ast, Node, type_name,
    N_OUTPUT, N_IF, N_FOR, N_BREAK, N_CONTINUE, N_NOOP, N_SET, N_MACRO, N_COMMENT, N_BLANK,
    N_MEMBER, N_CALL, N_IDENT, N_INT, N_FLOAT, N_STRING, N_ARRAY, N_TUPLE, N_OBJECT, N_BINARY,
    N_FILTER, N_FILTER_STMT, N_SELECT, N_TEST, N_UNARY, N_SLICE, N_KWARG, N_SPREAD, N_CALL_STMT,
    N_TERNARY,
)
from ninfer.text.jinja.value import (
    Heap, FuncArgs, Ctx,
    V_NONE, V_UNDEF, V_BOOL, V_INT, V_FLOAT, V_STRING, V_ARRAY, V_TUPLE, V_OBJECT, V_FUNC, V_KWARG,
    F_GLOBAL, F_METHOD, F_MACRO, F_CALLER,
    vtype, as_string, as_bool, as_int, as_float, as_array, equivalent, obj_find, obj_insert,
    obj_insert_str, obj_at, is_hashable, is_immutable, _f2i,
)
from ninfer.text.jinja.builtins import (
    has_global, has_method, call_global, call_method, call_test, is_class, C_ARRAY, out_push,
)





def _get_line_col(source: List[UInt8], pos: Int) -> String:
    var line = 1
    var col = 1
    var i = 0
    while i < pos and i < len(source):
        if source[i] == 0x0A:
            line += 1
            col = 1
        else:
            col += 1
        i += 1
    return "line " + String(line) + ", column " + String(col)


struct Runtime(Movable):
    var h: Heap
    var ast: Ast
    var src: List[UInt8]

    def __init__(out self, var ast: Ast, var src: List[UInt8]):
        self.h = Heap(JU())
        self.ast = ast^
        self.src = src^

    # --- context ---------------------------------------------------------------------------

    def new_env(mut self) raises JinjaError -> Int:
        var env = self.h.object()
        self.h.nodes[env].has_builtins = False
        obj_insert_str(self.h, env, "true", self.h.boolean(True))
        obj_insert_str(self.h, env, "True", self.h.boolean(True))
        obj_insert_str(self.h, env, "false", self.h.boolean(False))
        obj_insert_str(self.h, env, "False", self.h.boolean(False))
        obj_insert_str(self.h, env, "none", self.h.none())
        obj_insert_str(self.h, env, "None", self.h.none())
        return env

    def copy_env(mut self, parent: Int) -> Int:
        var env = self.h.object()
        self.h.nodes[env].has_builtins = False
        self.h.nodes[env].keys = self.h.nodes[parent].keys.copy()
        self.h.nodes[env].vals = self.h.nodes[parent].vals.copy()
        return env

    def get_val(mut self, env: Int, name: String) raises JinjaError -> Int:
        var key = self.h.text(name)
        var dflt = self.h.undef(name)
        return obj_at(self.h, env, key, dflt)

    def set_val(mut self, env: Int, name: String, v: Int) raises JinjaError:
        obj_insert_str(self.h, env, name, v)

    # --- execution -------------------------------------------------------------------------

    def execute_program(mut self, env: Int, time: Int) raises JinjaError -> Int:
        var results = self.h.array()
        for stmt in self.ast.body.copy():
            var res = self.execute(stmt, env, time)
            out_push(self.h, results, res)
        return self.gather(results)

    def gather(mut self, v: Int) raises JinjaError -> Int:
        var parts = JString()
        self._gather(v, parts)
        return self.h.string(parts^)

    def _gather(mut self, v: Int, mut parts: JString) raises JinjaError:
        var k = self.h.nodes[v].kind
        if k == V_STRING:
            parts.append(self.h.nodes[v].s)
        elif k == V_INT or k == V_FLOAT or k == V_BOOL:
            parts.append(as_string(self.h, v))
        elif k == V_ARRAY or k == V_TUPLE:
            for item in self.h.nodes[v].arr.copy():
                self._gather(item, parts)

    def exec_statements(mut self, stmts: List[Int], env: Int, time: Int) raises JinjaError -> Int:
        var result = self.h.array()
        for s in stmts:
            var r = self.execute(s, env, time)
            out_push(self.h, result, r)
        return self.gather(result)

    def execute(mut self, node: Int, env: Int, time: Int) raises JinjaError -> Int:
        try:
            return self.execute_impl(node, env, time)
        except e:
            if e.kind != K_ERROR:
                raise e^
            var msg = e.message
            var n_pos = self.ast.nodes[node].pos
            var ty = type_name(self.ast.nodes[node].kind)
            if len(self.src) == 0:
                raise JinjaError(
                    K_RETHROWN, "\nError executing " + ty + " at position " + String(n_pos) + ": " + msg
                )
            raise JinjaError(
                K_RETHROWN, "\n------------\nWhile executing " + ty + " at " + _get_line_col(self.src, n_pos)
                + " in source:\n" + peak_source(self.src, n_pos) + "\nError: " + msg
            )

    def execute_impl(mut self, node: Int, env: Int, time: Int) raises JinjaError -> Int:
        var n = self.ast.nodes[node].copy()
        var k = n.kind
        if k == N_STRING:
            return self.h.text(n.s)
        if k == N_INT:
            return self.h.integer(n.i)
        if k == N_FLOAT:
            return self.h.float(n.f)
        if k == N_OUTPUT:
            var r = self.execute(n.a, env, time)
            return self.h.string(as_string(self.h, r))
        if k == N_IDENT:
            var it = self.get_val(env, n.s)
            if self.h.nodes[it].kind != V_UNDEF:
                return it
            if has_global(n.s):
                return self.h.func(n.s, F_GLOBAL)
            return self.h.undef(n.s)
        if k == N_ARRAY or k == N_TUPLE:
            var arr = self.h.array()
            for item in n.l1:
                var v = self.execute(item, env, time)
                self.h.nodes[arr].arr.append(v)
            if k == N_TUPLE:
                var items_copy = self.h.nodes[arr].arr.copy()
                return self.h.tuple(items_copy)
            return arr
        if k == N_OBJECT:
            var obj = self.h.object()
            for i in range(len(n.l1)):
                var key = self.execute(n.l1[i], env, time)
                var val = self.execute(n.l2[i], env, time)
                obj_insert(self.h, obj, key, val)
            return obj
        if k == N_BINARY:
            return self.exec_binary(n, env, time)
        if k == N_FILTER:
            var operand = self.execute(n.a, env, time)
            return self.apply_filter(operand, n.b, env, time)
        if k == N_FILTER_STMT:
            var body = self.exec_statements(n.l1, env, time)
            return self.apply_filter(body, n.a, env, time)
        if k == N_TEST:
            return self.exec_test(n, env, time)
        if k == N_UNARY:
            var operand = self.execute(n.a, env, time)
            if n.s == "not":
                return self.h.boolean(not as_bool(self.h, operand))
            if n.s == "-":
                if self.h.nodes[operand].kind == V_INT:
                    return self.h.integer(-as_int(self.h, operand))
                if self.h.nodes[operand].kind == V_FLOAT:
                    return self.h.float(-as_float(self.h, operand))
                raise jinja_error("Unary - operator requires numeric operand")
            raise jinja_error("Unknown unary operator '" + n.s + "'")
        if k == N_IF:
            var test = self.execute(n.a, env, time)
            var out = self.h.array()
            var branch = n.l1.copy() if as_bool(self.h, test) else n.l2.copy()
            for s in branch:
                var r = self.execute(s, env, time)
                out_push(self.h, out, r)
            return self.gather(out)
        if k == N_FOR:
            return self.exec_for(n, env, time)
        if k == N_SET:
            return self.exec_set(n, env, time)
        if k == N_MACRO:
            if self.ast.nodes[n.a].kind != N_IDENT:
                raise jinja_error("Macro name must be an identifier")
            var name = self.ast.nodes[n.a].s
            var f = self.h.func(name, F_MACRO)
            self.h.nodes[f].node = node
            self.h.nodes[f].scope = env
            self.set_val(env, name, f)
            return self.h.undef()
        if k == N_CALL_STMT:
            return self.exec_call_stmt(node, n, env, time)
        if k == N_MEMBER:
            return self.exec_member(n, env, time)
        if k == N_CALL:
            var args = FuncArgs(Ctx(env, time))
            for a in n.l1:
                var v = self.execute(a, env, time)
                args.push_back(v)
            var callee = self.execute(n.a, env, time)
            if self.h.nodes[callee].kind != V_FUNC:
                raise jinja_error("Callee is not a function: got " + vtype(self.h, callee))
            return self.invoke(callee, args)
        if k == N_KWARG:
            if self.ast.nodes[n.a].kind != N_IDENT:
                raise jinja_error("Keyword argument key must be identifiers")
            var key = self.ast.nodes[n.a].s
            var v = self.execute(n.b, env, time)
            return self.h.kwarg(key, v)
        if k == N_TERNARY:
            var cond = self.execute(n.a, env, time)
            if as_bool(self.h, cond):
                return self.execute(n.b, env, time)
            return self.execute(n.c, env, time)
        if k == N_SELECT:
            var predicate = self.execute_impl(n.b, env, time)
            if not as_bool(self.h, predicate):
                return self.h.undef()
            return self.execute_impl(n.a, env, time)
        if k == N_BREAK:
            raise JinjaError(K_BREAK, "")
        if k == N_CONTINUE:
            raise JinjaError(K_CONTINUE, "")
        if k == N_NOOP or k == N_COMMENT or k == N_BLANK:
            return self.h.undef()
        if k == N_SLICE:
            raise jinja_error("must be handled by MemberExpression")
        raise jinja_error("cannot exec " + type_name(k))

    # --- binary ----------------------------------------------------------------------------

    def is_in(mut self, left: Int, right: Int, env: Int, time: Int) raises JinjaError -> Bool:
        var a = FuncArgs(Ctx(env, time))
        a.push_back(left)
        a.push_back(right)
        var r = call_test(self.h, "in", a)
        return as_bool(self.h, r)

    def exec_binary(mut self, n: Node, env: Int, time: Int) raises JinjaError -> Int:
        var op = n.s
        var left = self.execute(n.a, env, time)
        if op == "and":
            if as_bool(self.h, left):
                return self.execute(n.b, env, time)
            return left
        if op == "or":
            if as_bool(self.h, left):
                return left
            return self.execute(n.b, env, time)
        var right = self.execute(n.b, env, time)
        if op == "==":
            return self.h.boolean(equivalent(self.h, left, right))
        if op == "!=":
            return self.h.boolean(not equivalent(self.h, left, right))
        if op == "~":
            var s = as_string(self.h, left)
            s.append(as_string(self.h, right))
            return self.h.string(s^)
        var lk = self.h.nodes[left].kind
        var rk = self.h.nodes[right].kind
        if lk == V_UNDEF or rk == V_UNDEF:
            if rk == V_UNDEF and (op == "in" or op == "not in"):
                return self.h.boolean(op == "not in")
            raise jinja_error("Cannot perform operation " + op + " on undefined values")
        elif lk == V_NONE or rk == V_NONE:
            if rk != V_NONE and (op == "in" or op == "not in"):
                var member = self.is_in(left, right, env, time)
                return self.h.boolean(member if op == "in" else not member)
            raise jinja_error("Cannot perform operation on null values")
        if (lk == V_INT or lk == V_FLOAT) and (rk == V_INT or rk == V_FLOAT):
            var a = self.h.nodes[left].f
            var b = self.h.nodes[right].f
            var is_float = lk == V_FLOAT or rk == V_FLOAT
            if op == "+" or op == "-" or op == "*":
                var res = a + b if op == "+" else (a - b if op == "-" else a * b)
                if is_float:
                    return self.h.float(res)
                return self.h.integer(_f2i(res))
            if op == "/":
                return self.h.float(a / b)
            if op == "%":
                var rem = _fmod(a, b)
                if is_float:
                    return self.h.float(rem)
                return self.h.integer(_f2i(rem))
            if op == "<":
                return self.h.boolean(a < b)
            if op == ">":
                return self.h.boolean(a > b)
            if op == ">=":
                return self.h.boolean(a >= b)
            if op == "<=":
                return self.h.boolean(a <= b)
        var left_arr = lk == V_ARRAY or lk == V_TUPLE
        var right_arr = rk == V_ARRAY or rk == V_TUPLE
        if left_arr and right_arr:
            if op == "+":
                var result = self.h.array()
                for item in self.h.nodes[left].arr.copy():
                    self.h.nodes[result].arr.append(item)
                for item in self.h.nodes[right].arr.copy():
                    self.h.nodes[result].arr.append(item)
                return result
        elif right_arr:
            var member = self.is_in(left, right, env, time)
            if op == "in":
                return self.h.boolean(member)
            elif op == "not in":
                return self.h.boolean(not member)
        if lk == V_STRING and rk == V_STRING and op == "+":
            var s = as_string(self.h, left)
            s.append(as_string(self.h, right))
            return self.h.string(s^)
        if op == "*" and ((lk == V_STRING and rk == V_INT) or (lk == V_INT and rk == V_STRING)):
            var str = as_string(self.h, left) if lk == V_STRING else as_string(self.h, right)
            var repeat = as_int(self.h, right) if rk == V_INT else as_int(self.h, left)
            var res = JString()
            if repeat <= 0:
                return self.h.string(res^)
            for _ in range(repeat):
                res.append(str)
            return self.h.string(res^)
        if lk == V_STRING and rk == V_STRING:
            var member = self.is_in(left, right, env, time)
            if op == "in":
                return self.h.boolean(member)
            elif op == "not in":
                return self.h.boolean(not member)
        if rk == V_OBJECT:
            var member = self.is_in(left, right, env, time)
            if op == "in":
                return self.h.boolean(member)
            elif op == "not in":
                return self.h.boolean(not member)
        raise jinja_error("Unknown operator \"" + op + "\" between " + vtype(self.h, left) + " and " + vtype(self.h, right))

    # --- filters and tests -----------------------------------------------------------------

    def try_builtin_func(mut self, name: String, input: Int, undef_on_missing: Bool) raises JinjaError -> Int:
        if has_method(self.h, input, name):
            var f = self.h.func(name, F_METHOD, input)
            return f
        if undef_on_missing:
            return self.h.undef(name)
        raise jinja_error("Unknown (built-in) filter '" + name + "' for type " + vtype(self.h, input))

    def apply_filter(mut self, input_in: Int, filter: Int, env: Int, time: Int) raises JinjaError -> Int:
        var input = input_in
        var fnode = self.ast.nodes[filter].copy()
        if fnode.kind == N_IDENT:
            var id = fnode.s
            id = _alias(id)
            if self.h.nodes[input].kind != V_UNDEF and self.h.nodes[input].kind != V_STRING:
                if (
                    id == "capitalize" or id == "lower" or id == "replace" or id == "strip" or id == "title"
                    or id == "upper" or id == "wordcount"
                ):
                    input = self.h.string(as_string(self.h, input))
            var f = self.try_builtin_func(id, input, False)
            return self.invoke(f, FuncArgs(Ctx(env, time)))
        if fnode.kind == N_CALL:
            if self.ast.nodes[fnode.a].kind != N_IDENT:
                raise jinja_error("Filter callee must be an identifier")
            var id = _alias(self.ast.nodes[fnode.a].s)
            var args = FuncArgs(Ctx(env, time))
            for arg in fnode.l1:
                var v = self.execute(arg, env, time)
                args.push_back(v)
            var f = self.try_builtin_func(id, input, False)
            return self.invoke(f, args)
        raise jinja_error("Invalid filter expression")

    def exec_test(mut self, n: Node, env: Int, time: Int) raises JinjaError -> Int:
        var input = self.execute(n.a, env, time)
        var args = FuncArgs(Ctx(env, time))
        args.push_back(input)
        var tn = self.ast.nodes[n.b].copy()
        var test_id = String()
        if tn.kind == N_IDENT:
            test_id = tn.s
        elif tn.kind == N_CALL:
            if self.ast.nodes[tn.a].kind != N_IDENT:
                raise jinja_error("Test callee must be an identifier")
            test_id = self.ast.nodes[tn.a].s
            for arg in tn.l1:
                var v = self.execute(arg, env, time)
                args.push_back(v)
        else:
            raise jinja_error("Invalid test expression")
        if not has_global("test_is_" + test_id):
            raise jinja_error("Unknown test '" + test_id + "'")
        args.func_name = "test_is_" + test_id
        var res = call_test(self.h, test_id, args)
        if n.flag:
            return self.h.boolean(not as_bool(self.h, res))
        return res

    # --- invocation ------------------------------------------------------------------------

    def invoke(mut self, f: Int, args_in: FuncArgs) raises JinjaError -> Int:
        var fk = self.h.nodes[f].fkind
        var name = self.h.nodes[f].name
        var args = args_in.copy()
        args.func_name = name
        if fk == F_GLOBAL:
            return call_global(self.h, name, args)
        if fk == F_METHOD:
            args.push_front(self.h.nodes[f].bound)
            return call_method(self.h, name, args)
        if fk == F_MACRO:
            return self.invoke_macro(f, args)
        if fk == F_CALLER:
            return self.invoke_caller(f, args)
        raise jinja_error("Function is not callable")

    def bind_parameters(mut self, name: String, params: List[Int], args: FuncArgs, env: Int) raises JinjaError:
        var positional = List[Int]()
        var named_keys = List[String]()
        var named_vals = List[Int]()
        for input in args.args:
            if self.h.nodes[input].kind == V_KWARG:
                var key = self.h.nodes[input].name
                for k in named_keys:
                    if k == key:
                        raise jinja_error("duplicate macro argument: " + key)
                named_keys.append(key)
                named_vals.append(self.h.nodes[input].kwval)
            else:
                positional.append(input)
        if len(positional) > len(params):
            raise jinja_error("too many arguments to macro " + name)
        for i in range(len(params)):
            var p = self.ast.nodes[params[i]].copy()
            var plain = p.kind == N_IDENT
            var with_default = p.kind == N_KWARG
            if not plain and (not with_default or self.ast.nodes[p.a].kind != N_IDENT):
                raise jinja_error("invalid parameter in macro " + name)
            var key = p.s if plain else self.ast.nodes[p.a].s
            var input = -1
            var found = -1
            for j in range(len(named_keys)):
                if named_keys[j] == key:
                    found = j
                    break
            if found >= 0:
                if i < len(positional):
                    raise jinja_error("duplicate macro argument: " + key)
                input = named_vals[found]
                _ = named_keys.pop(found)
                _ = named_vals.pop(found)
            elif i < len(positional):
                input = positional[i]
            elif with_default:
                input = self.execute(p.b, env, args.ctx.time)
            else:
                input = self.h.undef(key)
            self.set_val(env, key, input)
        if len(named_keys) > 0:
            var first = named_keys[0]
            for k in named_keys:
                if k < first:
                    first = k
            raise jinja_error("unknown macro argument: " + first)

    def invoke_macro(mut self, f: Int, args: FuncArgs) raises JinjaError -> Int:
        var macro_node = self.ast.nodes[self.h.nodes[f].node].copy()
        var env = self.copy_env(self.h.nodes[f].scope)
        var caller = self.get_val(args.ctx.env, "caller")
        if self.h.nodes[caller].kind != V_UNDEF:
            self.set_val(env, "caller", caller)
        var macro_name = self.h.nodes[f].name
        self.bind_parameters(macro_name, macro_node.l1, args, env)
        return self.exec_statements(macro_node.l2, env, args.ctx.time)

    def invoke_caller(mut self, f: Int, args: FuncArgs) raises JinjaError -> Int:
        var call_node = self.ast.nodes[self.h.nodes[f].node].copy()
        var env = self.copy_env(self.h.nodes[f].scope)
        self.bind_parameters("caller", call_node.l1, args, env)
        return self.exec_statements(call_node.l2, env, args.ctx.time)

    def exec_call_stmt(mut self, node: Int, n: Node, env: Int, time: Int) raises JinjaError -> Int:
        var call = self.ast.nodes[n.a].copy()
        if self.ast.nodes[n.a].kind != N_CALL:
            raise jinja_error("Call statement requires a valid call expression")
        var callee = self.execute(call.a, env, time)
        if self.h.nodes[callee].kind != V_FUNC:
            raise jinja_error("Callee is not a function: got " + vtype(self.h, callee))
        var call_env = self.copy_env(env)
        var caller = self.h.func("caller", F_CALLER)
        self.h.nodes[caller].node = node
        self.h.nodes[caller].scope = env
        self.set_val(call_env, "caller", caller)
        var args = FuncArgs(Ctx(call_env, time))
        for a in call.l1:
            var v = self.execute(a, env, time)
            args.push_back(v)
        return self.invoke(callee, args)

    # --- statements -------------------------------------------------------------------------

    def exec_set(mut self, n: Node, env: Int, time: Int) raises JinjaError -> Int:
        var rhs = -1
        if n.b >= 0:
            rhs = self.execute(n.b, env, time)
        else:
            rhs = self.exec_statements(n.l1, env, time)
        var assignee = self.ast.nodes[n.a].copy()
        if assignee.kind == N_IDENT:
            self.set_val(env, assignee.s, rhs)
        elif assignee.kind == N_TUPLE:
            var rk = self.h.nodes[rhs].kind
            if rk != V_ARRAY and rk != V_TUPLE:
                raise jinja_error("Cannot unpack non-iterable type in set: " + vtype(self.h, rhs))
            var arr = self.h.nodes[rhs].arr.copy()
            if len(arr) != len(assignee.l1):
                raise jinja_error("Too " + ("few" if len(assignee.l1) > len(arr) else "many") + " items to unpack in set")
            for i in range(len(assignee.l1)):
                var elem = self.ast.nodes[assignee.l1[i]].copy()
                if elem.kind != N_IDENT:
                    raise jinja_error("Cannot unpack to non-identifier in set: " + type_name(elem.kind))
                self.set_val(env, elem.s, arr[i])
        elif assignee.kind == N_MEMBER:
            if assignee.flag:
                raise jinja_error("Cannot assign to computed member")
            if self.ast.nodes[assignee.b].kind != N_IDENT:
                raise jinja_error("Cannot assign to member with non-identifier property")
            var prop = self.ast.nodes[assignee.b].s
            var object = self.execute(assignee.a, env, time)
            if self.h.nodes[object].kind != V_OBJECT:
                raise jinja_error("Cannot assign to member of non-object")
            obj_insert_str(self.h, object, prop, rhs)
        else:
            raise jinja_error("Invalid LHS inside assignment expression: " + type_name(assignee.kind))
        return self.h.undef()

    def exec_for(mut self, n: Node, env: Int, time: Int) raises JinjaError -> Int:
        var scope = self.copy_env(env)
        var iter_node = n.b
        var test_node = -1
        var is_select = self.ast.nodes[n.b].kind == N_SELECT
        if is_select:
            iter_node = self.ast.nodes[n.b].a
            test_node = self.ast.nodes[n.b].b
        var iterable = self.execute(iter_node, scope, time)
        if self.h.nodes[iterable].kind == V_UNDEF:
            iterable = self.h.array()
        var ik = self.h.nodes[iterable].kind
        if not (ik == V_ARRAY or ik == V_TUPLE or ik == V_OBJECT or ik == V_STRING):
            raise jinja_error("Expected iterable or object type in for loop: got " + vtype(self.h, iterable))
        var items = List[Int]()
        if ik == V_OBJECT:
            var keys = self.h.nodes[iterable].keys.copy()
            var vals = self.h.nodes[iterable].vals.copy()
            for i in range(len(keys)):
                var pair = List[Int]()
                pair.append(keys[i])
                pair.append(vals[i])
                items.append(self.h.tuple(pair))
        elif ik == V_STRING:
            var str = self.h.nodes[iterable].s.copy()
            for ch in characters(self.h.ju, str.str()):
                items.append(self.h.string(str.cut_bytes(ch.begin, ch.end)))
        else:
            items = self.h.nodes[iterable].arr.copy()

        var loopvar = self.ast.nodes[n.a].copy()
        var filtered = List[Int]()
        var filtered_idx = List[Int]()
        for i in range(len(items)):
            var loop_scope = self.copy_env(scope)
            var current = items[i]
            if loopvar.kind == N_IDENT:
                if ik == V_OBJECT:
                    current = self.h.nodes[items[i]].arr[0]
            elif loopvar.kind == N_TUPLE:
                var ck = self.h.nodes[current].kind
                if ck != V_ARRAY and ck != V_TUPLE:
                    raise jinja_error("Cannot unpack non-iterable type: " + vtype(self.h, current))
                var c_arr = self.h.nodes[current].arr.copy()
                if len(loopvar.l1) != len(c_arr):
                    raise jinja_error("Too " + ("few" if len(loopvar.l1) > len(c_arr) else "many") + " items to unpack")
            else:
                raise jinja_error("Invalid loop variable(s): " + type_name(loopvar.kind))
            if is_select and test_node >= 0:
                self.apply_loop_vars(loopvar, items[i], ik, loop_scope)
                var tv = self.execute(test_node, loop_scope, time)
                if not as_bool(self.h, tv):
                    continue
            filtered.append(current)
            filtered_idx.append(i)

        var result = self.h.array()
        var no_iteration = True
        var count = len(filtered)
        for i in range(count):
            var loop_obj = self.h.object()
            self.h.nodes[loop_obj].has_builtins = False
            obj_insert_str(self.h, loop_obj, "index", self.h.integer(i + 1))
            obj_insert_str(self.h, loop_obj, "index0", self.h.integer(i))
            obj_insert_str(self.h, loop_obj, "revindex", self.h.integer(count - i))
            obj_insert_str(self.h, loop_obj, "revindex0", self.h.integer(count - i - 1))
            obj_insert_str(self.h, loop_obj, "first", self.h.boolean(i == 0))
            obj_insert_str(self.h, loop_obj, "last", self.h.boolean(i == count - 1))
            obj_insert_str(self.h, loop_obj, "length", self.h.integer(count))
            obj_insert_str(self.h, loop_obj, "previtem", filtered[i - 1] if i > 0 else self.h.undef("previtem"))
            obj_insert_str(self.h, loop_obj, "nextitem", filtered[i + 1] if i < count - 1 else self.h.undef("nextitem"))
            self.set_val(scope, "loop", loop_obj)
            self.apply_loop_vars(loopvar, items[filtered_idx[i]], ik, scope)
            var skipped = False
            var stopped = False
            for stmt in n.l1:
                try:
                    var val = self.execute(stmt, scope, time)
                    out_push(self.h, result, val)
                except e:
                    if e.kind == K_CONTINUE:
                        skipped = True
                        break
                    if e.kind == K_BREAK:
                        stopped = True
                        break
                    raise e^
            if skipped:
                continue
            if stopped:
                break
            no_iteration = False
        if no_iteration:
            for stmt in n.l2:
                var val = self.execute(stmt, env, time)
                out_push(self.h, result, val)
        return self.gather(result)

    def apply_loop_vars(mut self, loopvar: Node, item: Int, iterable_kind: Int, scope: Int) raises JinjaError:
        if loopvar.kind == N_IDENT:
            if iterable_kind == V_OBJECT:
                self.set_val(scope, loopvar.s, self.h.nodes[item].arr[0])
            else:
                self.set_val(scope, loopvar.s, item)
        else:
            var c_arr = self.h.nodes[item].arr.copy()
            for j in range(len(loopvar.l1)):
                var el = self.ast.nodes[loopvar.l1[j]].copy()
                if el.kind != N_IDENT:
                    raise jinja_error("Cannot unpack non-identifier type: " + type_name(el.kind))
                self.set_val(scope, el.s, c_arr[j])

    # --- member access ----------------------------------------------------------------------

    def exec_member(mut self, n: Node, env: Int, time: Int) raises JinjaError -> Int:
        var object = self.execute(n.a, env, time)
        var property = -1
        var pn = self.ast.nodes[n.b].copy()
        if n.flag:
            if pn.kind == N_SLICE:
                var step_val = self.execute(pn.c, env, time) if pn.c >= 0 else self.h.integer(1)
                var start_val = self.execute(pn.a, env, time) if pn.a >= 0 else self.h.undef()
                var stop_val = self.execute(pn.b, env, time) if pn.b >= 0 else self.h.undef()
                var f = self.try_builtin_func("slice", object, False)
                var args = FuncArgs(Ctx(env, time))
                args.push_back(start_val)
                args.push_back(stop_val)
                args.push_back(step_val)
                return self.invoke(f, args)
            property = self.execute(n.b, env, time)
        elif pn.kind == N_INT:
            property = self.h.integer(pn.i)
            if pn.i < 0:
                raise jinja_error("Static member property cannot be negative")
        else:
            if pn.kind != N_IDENT:
                raise jinja_error("Static member property must be an identifier")
            property = self.h.text(pn.s)
            var v = self.try_builtin_func(pn.s, object, True)
            if self.h.nodes[v].kind != V_UNDEF:
                return v
        var val = self.h.undef("object_property")
        if self.h.nodes[property].kind == V_UNDEF:
            return val
        if not is_hashable(self.h, property):
            raise jinja_error("Type: " + vtype(self.h, property) + " is not allowed as object key")
        var ok = self.h.nodes[object].kind
        if ok == V_UNDEF:
            return val
        if ok == V_OBJECT:
            var key = as_string(self.h, property).str()
            val = obj_at(self.h, object, property, val)
            if self.h.nodes[val].kind == V_UNDEF:
                val = self.try_builtin_func(key, object, True)
        elif ok == V_ARRAY or ok == V_TUPLE or ok == V_STRING:
            var pk = self.h.nodes[property].kind
            if pk == V_INT:
                var index = self.h.nodes[property].i
                if ok != V_STRING:
                    var size = len(self.h.nodes[object].arr)
                    if index < 0:
                        index += size
                    if index >= 0 and index < size:
                        val = self.h.nodes[object].arr[index]
                else:
                    var str = self.h.nodes[object].s.copy()
                    var size = str.length()
                    if index < 0:
                        index += size
                    if index >= 0 and index < size:
                        val = self.h.string(str.slice(self.h.ju, Optional[Int](index), Optional[Int](index + 1), 1))
            elif pk == V_STRING:
                var key = self.h.nodes[property].s.str()
                val = self.try_builtin_func(key, object, True)
            else:
                raise jinja_error("Cannot access property with non-string/non-number: got " + vtype(self.h, property))
        else:
            if self.h.nodes[property].kind != V_STRING:
                raise jinja_error("Cannot access property with non-string: got " + vtype(self.h, property))
            var key = self.h.nodes[property].s.str()
            val = self.try_builtin_func(key, object, True)
        return val


def _fmod(a: Float64, b: Float64) -> Float64:
    """C `fmod`: the result takes the sign of the dividend."""
    if b == 0.0 or a != a or b != b or a - a != 0.0:
        return Float64.MAX * 2.0 - Float64.MAX * 2.0
    if b - b != 0.0:
        return a
    var m = a % b
    if m != 0.0 and ((m < 0.0) != (a < 0.0)):
        m -= b
    return m


def _alias(id: String) -> String:
    if id == "count":
        return "length"
    if id == "d":
        return "default"
    if id == "e":
        return "escape"
    if id == "trim":
        return "strip"
    return id
