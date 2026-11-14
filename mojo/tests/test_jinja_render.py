"""Parity: Mojo Jinja renderer against src/text/jinja.cpp (llama-jinja runtime, nlohmann input).

Cases are rendered by both and diffed: text, literal spans and provenance regions, or the error
message with its source excerpt. Corpus: curated expressions over a rich context (every builtin
filter and test, member access, loops, macros, namespaces), generated templates, and the three
chat templates the repo ships rendered against realistic conversations.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
import shutil
import subprocess

import pytest

from test_jinja_lexer import ROOT, HERE, MOJO, CXX, build_jinja_oracle

pytestmark = pytest.mark.skipif(MOJO is None or CXX is None, reason="needs mojo and a C++ compiler")

CONTEXT = {
    "s": "Hello World", "e": "", "u": "h\u00e9llo \u0391\u03a3 stra\u00dfe", "n": 42, "neg": -7, "f": 3.5, "b": True, "nul": None,
    "l": [3, 1, 2], "ls": ["b", "a", "c"], "d": {"x": 1, "y": "two", "z": [1, 2]},
    "objs": [{"k": "a", "v": 1}, {"k": "b", "v": 2}, {"k": "c", "v": 3}], "nested": [[1, 2], [3, 4]],
    "words": "a b  c d", "padded": "  xx  ", "multi": "l1\nl2\n\nl4", "big": 9007199254740993, "fl": [1.0, 2.5, 1e20, 1e-7, 0.1],
}
EXPRS = r"""
s|s|upper|s|lower|s|title|s|capitalize|u|upper|u|lower|u|title|u|capitalize|s|length|u|length|s|trim|padded|strip
padded|lstrip|padded|rstrip|s|replace('l','L')|s|replace('l','L',1)|s|replace('','-')|s|replace('o','0',count=1)
s|split(' ')|words|split|words|split(' ')|words|split(None,1)|words|rsplit(None,1)|s|split('o',1)|s|rsplit('o',1)
s|startswith('He')|s|endswith('ld')|s|startswith(1)|'{} and {}'|format(1,'x')|'a{}'|format('b')|'{x}'|format(1)
'12'|int|'0x1f'|int(base=16)|'abc'|int(default=7)|'3.5e2x'|float|'zz'|float(default=2.5)|n|int|n|float|f|int|f|float|b|int|b|float
n|abs|neg|abs|f|abs|b|string|n|string|f|string|nul|string|d|string|l|string|ls|string|s|string
l|first|l|last|l|length|l|sort|l|sort(reverse=true)|ls|sort|l|reverse|l|min|l|max|l|list|e|list
ls|join(',')|ls|join|l|join('-')|objs|join(',',attribute='k')|objs|map(attribute='k')|objs|map(attribute='z',default='dflt')
nested|map(attribute=0)|objs|selectattr('k')|objs|selectattr('v','gt',1)|objs|rejectattr('v','gt',1)|l|select('odd')|l|reject('odd')
objs|sort(attribute='v',reverse=true)|d|keys|d|values|d|items|d|length|d|get('x')|d|get('q')|d|get('q',5)|d|dictsort|d|dictsort(by='value')
d|tojson|d|tojson(indent=2)|d|tojson(separators=(',',':'))|s|tojson|u|tojson|u|tojson(ensure_ascii=true)|fl|tojson|f|tojson|nul|tojson|big|tojson
l[0]|l[-1]|l[5]|l[1:]|l[:2]|l[::-1]|l[::2]|s[0]|s[-1]|s[1:4]|s[::-1]|u[1:3]|d.x|d['y']|d.z[1]|d.q|d.items|d.keys()|objs[1].k|objs.0.k
n+1|n-1|n*2|n/5|n%5|neg%5|f*2|f+1|n+f|'a'+'b'|'ab'*3|3*'ab'|l+l|n==42|n!=42|n<50|n>=42|f<=3.5|'a'<'b'|s~n|s~nul|1/0|n%0
n in l|1 in l|'a' in ls|'He' in s|'x' in d|nul in d|n not in l|1 in undefined_var|none in d|n is odd|n is even|n is divisibleby(7)
n is integer|f is float|n is number|s is string|l is iterable|d is mapping|nul is none|undefined_var is defined|undefined_var is undefined
s is lower|s is upper|n is eq(42)|n is ne(1)|n is gt(1)|n is lt(1)|n is ge(42)|n is le(1)|b is true|b is false|n is callable|s is sameas(s)
not b|-n|-f|n if b else 0|n if not b else 'no'|undefined_var|undefined_var.x|undefined_var|default('d')|nul|default('d')|e|default('d',true)
range(3)|range(2,5)|range(5,0,-2)|range(0)|range(1)|range(1,2,0)|namespace(a=1).a|namespace(a=1,b=2)|namespace(1)|raise_exception('boom')
l|slice(1,2)|s|slice(1,3)|s|indent(2)|multi|indent(2)|multi|indent(2,true)|multi|indent(width=1,first=true,blank=true)|s|safe|n|safe|b|safe
l|append(4)|l|pop|l|pop(0)|ls|unique|ls|string|d|join|s|join|n|bogus|s|nosuch|nul|items|nul|default('q')|undefined_var|upper|undefined_var|length
n|upper|n|trim|f|capitalize|l|upper|d|upper|nul|upper|(1,2)|(1,2)|length|(1,)|[1,2]|{'a':1}|{'a':{'b':2}}|{1:2}|{[1]:2}|{(1,2):3}
l|map(1)|l|map('upper')|ls|map('upper')|l|first|first|s|x|l|sort(attribute='q')|objs|sort(attribute='k')|l|min(attribute='q')|nul|select
s.upper()|s.split()|s.lower().upper()|s|upper|lower|(s|upper)|length|"a" "b"|'it''s'|'\n'|'a\tb'|'q\'q'|'\u00e9'|'\\'
macro_missing()|s()|d.x()|n.x|s.x|l.x|nul.x|b.x|f.x|l['x']|d[1]|d[nul]|d[[1]]|d[l]|s[1.5]|s[nul]|1.5|007|1e3|.5
"""
STATEMENTS = [
    "{% set x = 5 %}{{ x }}", "{% set a, b = [1,2] %}{{ a }}{{ b }}", "{% set a, b = [1] %}", "{% set a, b = 3 %}", "{% set ns = namespace(c=0) %}{% for i in l %}{% set ns.c = ns.c + i %}{% endfor %}{{ ns.c }}",
    "{% set x %}captured {{ n }}{% endset %}[{{ x }}]", "{% for i in l %}{{ i }},{% endfor %}", "{% for i in l %}{{ loop.index }}/{{ loop.length }}{{ loop.first }}{{ loop.last }}{{ loop.revindex }}{{ loop.previtem }}{{ loop.nextitem }};{% endfor %}",
    "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}", "{% for k in d %}{{ k }};{% endfor %}", "{% for k, v in d %}{{ k }};{% endfor %}", "{% for c in s %}[{{ c }}]{% endfor %}", "{% for c in u %}[{{ c }}]{% endfor %}",
    "{% for i in l if i > 1 %}{{ i }}{% endfor %}", "{% for i in e %}x{% else %}empty{% endfor %}", "{% for i in l %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% endfor %}", "{% for i in l %}{% if i == 1 %}{% break %}{% endif %}{{ i }}{% endfor %}",
    "{% for i, j in nested %}{{ i }}{{ j }};{% endfor %}", "{% for i, j in l %}x{% endfor %}", "{% for i in n %}x{% endfor %}", "{% for i in undefined_var %}x{% else %}none{% endfor %}", "{% for a, b, c in nested %}x{% endfor %}",
    "{% for i in l %}{% for j in l %}{{ i }}{{ j }}{% endfor %}|{% endfor %}", "{% for i in l %}{% set t = i %}{% endfor %}[{{ t }}]", "{% for i in range(3) %}{{ i }}{% endfor %}",
    "{% if n > 40 %}big{% elif n > 10 %}mid{% else %}small{% endif %}", "{% if n < 40 %}a{% elif n < 50 %}b{% else %}c{% endif %}", "{% if nul %}a{% else %}b{% endif %}", "{% if s and n %}t{% endif %}", "{% if e or n %}t{% endif %}",
    "{% macro m(a, b=2) %}{{ a }}-{{ b }}{% endmacro %}{{ m(1) }}{{ m(1,3) }}{{ m(b=9,a=0) }}", "{% macro m(a) %}{{ a }}{% endmacro %}{{ m() }}{{ m(1,2) }}", "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(c=1) }}", "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1,a=2) }}",
    "{% macro fact(k) %}{% if k <= 1 %}1{% else %}{{ k * fact(k - 1)|int }}{% endif %}{% endmacro %}{{ fact(5) }}", "{% macro outer() %}{{ n }}{% endmacro %}{% set n = 1 %}{{ outer() }}",
    "{% macro wrap() %}<{{ caller() }}>{% endmacro %}{% call wrap() %}inner {{ n }}{% endcall %}", "{% macro wrap(x) %}<{{ caller(x) }}>{% endmacro %}{% call(v) wrap(5) %}got {{ v }}{% endcall %}",
    "{% filter upper %}shout {{ s }}{% endfilter %}", "{% filter replace('a','b') %}banana{% endfilter %}", "{% filter bogus %}x{% endfilter %}", "{# comment #}after", "a  {%- if true -%}  b  {%- endif -%}  c", "{% if true %}\n  x\n{% endif %}\n",
    "{{ 'x' }}\n{{ 'y' }}\n", "line1\n  {% if true %}\n  line2\n  {% endif %}\nline3", "{% generation %}g{% endgeneration %}", "{% set a = 1 %}{% set a = a + 1 %}{{ a }}", "{{ n is not odd }}{{ n is not even }}",
    "{% set x = [1,2] %}{% set _ = x.append(3) %}{{ x }}", "{{ [1,2].append(3) }}", "{% set t = (1,2) %}{{ t.append(3) }}", "{% if %}x{% endif %}", "{% foo %}", "{% for %}x{% endfor %}", "{{ }}", "{{ 1 + }}", "{% if true %}x", "{% endif %}", "{{ 'unterminated", "{{ n | }}",
    "{% set x = n %}{{ x if x else 'no' }}", "{{ (n if b else f) + 1 }}", "{% for i in l %}{{ i }}{{ ', ' if not loop.last }}{% endfor %}", "{{ l | select('odd') | list }}", "{{ l | map('string') | list }}",
    "{{ d | tojson }}", "{% for m in objs %}{{ m.k }}:{{ m['v'] }}{% endfor %}", "{{ objs | selectattr('k', 'equalto', 'b') | list | length }}", "{{ nested[1][0] }}{{ nested[-1][-1] }}",
    "{% set d2 = {'a': 1} %}{% set _ = d2.update({'b': 2}) %}", "{{ s is string and n is number }}", "{{ 'a' in 'abc' }}{{ 'z' in 'abc' }}{{ 1 in 'abc' }}", "{{ 1 ~ 2 }}{{ 1 + '2' }}", "{{ [1] + 2 }}", "{{ -s }}", "{{ not s }}{{ not l }}{{ not d }}",
]
REGION_CASES = [
    ('{{ user }}', {"user": "alice"}, [{"pointer": "/user", "tag": 7}], []),
    ('{{ user|upper }}{{ user|trim }}{{ user|tojson }}', {"user": " alice "}, [{"pointer": "/user", "tag": 7}], []),
    ('[{{ a }}|{{ b }}]', {"a": "x", "b": "y"}, [{"pointer": "/a", "tag": 1}, {"pointer": "/b", "tag": 2}], []),
    ('{{ m.content }}{{ m.content[1:3] }}{{ m.content|replace("l","L") }}', {"m": {"content": "hello"}}, [{"pointer": "/m/content", "tag": 3}], []),
    ('{% for x in xs %}<{{ x }}>{% endfor %}', {"xs": ["a", "b"]}, [{"pointer": "/xs/0", "tag": 4}, {"pointer": "/xs/1", "tag": 5}], []),
    ('{{ tok }}{{ other }}', {"tok": "<|im_start|>", "other": "plain"}, [{"pointer": "/other", "tag": 9}], ["tok"]),
    ('{{ d|tojson }}', {"d": {"k": "v"}}, [{"pointer": "/d", "tag": 6}], []),
    ('{{ ~ }}', {}, [], []),
    ('{{ s ~ s }}{{ s + s }}{{ s * 2 }}', {"s": "ab"}, [{"pointer": "/s", "tag": 8}], []),
    ('{{ s|indent(2) }}{{ s|split(",")|join("-") }}', {"s": "a,b\nc"}, [{"pointer": "/s", "tag": 8}], []),
    ('{{ s|upper }}{{ s|lower }}{{ s|title }}{{ s|capitalize }}{{ s|length }}', {"s": "mIx"}, [{"pointer": "/s", "tag": 8}], []),
    ('{{ big }}', {"big": 18446744073709551615}, [], []),
    ('{{ x }}', {"x": -9223372036854775808}, [], []),
]


def case_line(template, context=None, regions=(), controls=()):
    return json.dumps({"template": template, "context": CONTEXT if context is None else context, "regions": list(regions),
                       "controls": list(controls), "timestamp": 1700000000})


def normalize(blocks):
    # the libc++ spelling of typeid(std::shared_ptr<T>).name() on macOS -> libstdc++ spelling the port emits
    out = []
    for b in blocks:
        if b.startswith("err "):
            raw = bytes.fromhex(b[4:].strip()).decode("utf-8", "replace")
            raw = re.sub(r"NSt3__110shared_ptrIN5jinja(\d+\w+)EEE", r"St10shared_ptrIN5jinja\1EE", raw)
            b = "err " + raw
        out.append(b)
    return out


def split_cases(text):
    blocks = ("\n" + text).split("\ncase ")[1:]
    out = []
    for blk in blocks:
        lines = blk.strip().split("\n")[1:]
        out.append("\n".join(f"err {bytes.fromhex(l[4:]).decode('utf-8','replace')}" if l.startswith("err ") else l for l in lines))
    return out


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    return build_jinja_oracle(tmp_path_factory.mktemp("jr"), HERE / "jinja_render_oracle.cpp", "render_oracle", extra=[ROOT / "src/text/jinja.cpp"])


def run(oracle, tmp_path, lines):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n")
    mine = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(HERE / "jinja_render_probe.mojo"), str(path)], capture_output=True, text=True)
    assert mine.returncode == 0, mine.stderr[-2000:]
    theirs = subprocess.run([str(oracle), str(path)], capture_output=True, text=True, check=True)
    a, b = split_cases(mine.stdout), split_cases(theirs.stdout)
    na = [re.sub(r"NSt3__110shared_ptrIN5jinja(\d+\w+)EEE", r"St10shared_ptrIN5jinja\1EE", x) for x in b]
    return a, na


def check(oracle, tmp_path, lines, min_errors=0):
    a, b = run(oracle, tmp_path, lines)
    assert len(a) == len(b) == len(lines)
    bad = [i for i in range(len(lines)) if a[i] != b[i]]
    assert not bad, [(json.loads(lines[i])["template"], a[i][:400], b[i][:400]) for i in bad[:4]] + [f"{len(bad)} of {len(lines)} differ"]
    assert sum(x.startswith("err") for x in b) >= min_errors
    return b


def test_expressions(oracle, tmp_path):
    expressions = set()
    for line in EXPRS.strip().split("\n"):
        parts = line.split("|")
        buf = parts[0]
        for p in parts[1:]:
            # `operand|filter(...)`: a part naming a filter joins the previous operand with a pipe
            if p.split("(")[0] in FILTERS and re.match(r"^[a-z_]+(\(.*\))?$", p):
                buf += "|" + p
            else:
                expressions.add(buf)
                buf = p
        expressions.add(buf)
    lines = [case_line("{{ " + e + " }}") for e in sorted(expressions) if e.strip()]
    assert len(lines) > 250
    check(oracle, tmp_path, lines, min_errors=20)


FILTERS = set("""upper lower title capitalize length trim strip lstrip rstrip replace split rsplit startswith endswith format int float abs string
first last sort reverse min max list join map selectattr rejectattr select reject keys values items get dictsort tojson slice indent safe append pop
unique default count d e""".split())


def test_statements(oracle, tmp_path):
    check(oracle, tmp_path, [case_line(t) for t in STATEMENTS], min_errors=10)


def test_provenance_regions(oracle, tmp_path):
    lines = [case_line(t, c, r, k) for t, c, r, k in REGION_CASES]
    check(oracle, tmp_path, lines)


def test_generated_templates(oracle, tmp_path):
    rng = random.Random(20261114)
    frags = ["{{ s }}", "{{ n + 1 }}", "{% if b %}", "{% else %}", "{% endif %}", "{% for i in l %}", "{% endfor %}", "text ", "\n", "  ", "{{- s -}}", "{%- if n -%}",
             "{{ d.x }}", "{# c #}", "{% set z = n %}", "{{ z }}", "{{ l|join(',') }}", "{{ objs[0].k|upper }}", "{{ i }}"]
    lines = []
    for _ in range(1500):
        lines.append(case_line("".join(rng.choice(frags) for _ in range(rng.randint(1, 9)))))
    check(oracle, tmp_path, lines, min_errors=50)


CONVERSATION = {
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2? \u00e9\u4e2d"},
        {"role": "assistant", "content": "<think>\nsimple\n</think>\n\nIt is 4.", "reasoning_content": "simple"},
        {"role": "user", "content": [{"type": "text", "text": "and 3+3?"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": {"expression": "3+3", "precise": True}}}]},
        {"role": "tool", "content": "6", "tool_call_id": "c1"},
    ],
    "tools": [{"type": "function", "function": {"name": "calc", "description": "Evaluate", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}],
    "add_generation_prompt": True, "enable_thinking": True, "bos_token": "<s>", "eos_token": "<|im_end|>",
}


@pytest.mark.parametrize("name", ["qwen3_6", "qwen3_8", "froggeric_v22_5"])
def test_shipped_chat_templates(oracle, tmp_path, name):
    template = (ROOT / "tools/chat_templates" / f"{name}.jinja").read_text()
    variants = []
    for gen in (True, False):
        for think in (True, False):
            for tools in (True, False):
                ctx = json.loads(json.dumps(CONVERSATION))
                ctx["add_generation_prompt"], ctx["enable_thinking"] = gen, think
                if not tools:
                    ctx.pop("tools")
                variants.append(ctx)
    short = json.loads(json.dumps(CONVERSATION))
    short["messages"] = short["messages"][:2]
    variants.append(short)
    regions = [{"pointer": "/messages/1/content", "tag": 11}, {"pointer": "/messages/0/content", "tag": 12}]
    lines = [case_line(template, c, regions, ["bos_token", "eos_token"]) for c in variants]
    got = check(oracle, tmp_path, lines)
    assert sum(x.startswith("text") for x in got) >= 6, got[0][:300]
