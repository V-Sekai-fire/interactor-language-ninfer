"""Parity: the Mojo Qwen tool-call parser and streaming decoder against the engine's own C++
(`src/models/qwen3_5/frontend/tool_call_parser.cpp`, built with the system C++ compiler).

Inputs: a hand-written corpus of well-formed and broken calls under several tool contracts, random
edits of those texts (so the recovery pass sees every kind of damage), and the same texts fed to
the decoder in random chunks, with and without a forced opener."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from test_jinja_lexer import CXX, MOJO, ROOT

HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(MOJO is None or CXX is None, reason="needs mojo and a C++ compiler")


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    work = tmp_path_factory.mktemp("toolcall")
    exe = work / "oracle"
    subprocess.run(
        [CXX, "-std=c++20", "-O1", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}", f"-I{ROOT / 'third_party'}",
         str(HERE / "tool_call_oracle.cpp"), str(ROOT / "src/models/qwen3_5/frontend/tool_call_parser.cpp"), "-o", str(exe)],
        check=True,
    )
    return exe


def tool(name, props):
    return json.dumps({"type": "function", "function": {"name": name, "description": "d", "parameters": {
        "type": "object", "properties": props, "required": []}}})


CALC = tool("calc", {"expression": {"type": "string"}, "precision": {"type": "integer"}, "exact": {"type": "boolean"}})
TYPED = tool("typed", {"n": {"type": "number"}, "i": {"type": "integer"}, "b": {"type": "boolean"}, "s": {"type": "string"},
                       "o": {"type": "object"}, "a": {"type": "array"}, "z": {"type": "null"},
                       "u": {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "m": {"type": ["string", "null"]},
                       "legacy": {"description": "no type"}, "bad": {"type": "weird"}, "e": {"type": []}})
DUP_A = tool("dup", {"x": {"type": "string"}})
DUP_B = tool("dup", {"x": {"type": "integer"}})
BARE = tool("bare", {})
NOPARAMS = json.dumps({"type": "function", "function": {"name": "np"}})
CONTRACTS = {
    "none": [],
    "calc": [CALC],
    "typed": [CALC, TYPED, BARE, NOPARAMS],
    "dup": [DUP_A, DUP_B, CALC],
    "junk": ["not json", "[]", "{}", json.dumps({"function": 3}), CALC],
}

FN = lambda name, params, close="</function>": f"<function={name}>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params) + close
CALL = lambda name, params: "<tool_call>\n" + FN(name, params) + "</tool_call>"

TEXTS = [
    "plain answer with no call",
    "Let me compute.\n" + CALL("calc", [("expression", "1+2")]),
    CALL("calc", [("expression", "2*(3+4)"), ("precision", " 5 "), ("exact", "True")]),
    CALL("calc", [("expression", "x"), ("expression", "y")]),
    CALL("calc", []) + "\n" + CALL("typed", [("n", "1e3"), ("i", "10.0"), ("b", "false"), ("s", " keep "), ("o", '{"a":1}'), ("a", "[1,2]"), ("z", "null")]),
    CALL("typed", [("n", "abc"), ("i", "1.5"), ("b", "maybe"), ("o", "[1]"), ("a", "{}"), ("u", "7"), ("u", "null"), ("m", "text"), ("m", "null")]),
    CALL("typed", [("legacy", "12"), ("legacy", "not json"), ("bad", "x"), ("e", "y"), ("nope", "z")]),
    CALL("typed", [("n", ""), ("i", "  "), ("s", ""), ("b", "\n")]),
    CALL("typed", [("n", "-0"), ("n", "1E+400"), ("i", "100e-2"), ("i", "1.50e1"), ("i", "-0.0"), ("i", "0.000e5"), ("i", "12e-1")]),
    CALL("calc", [("expression", "é中😀 \u0001 \"quoted\" \\ back")]),
    "<tool_call>\n<function=calc>\n<parameter=expression>\r\n1+1\r\n</parameter>\n</function>\n</tool_call>",
    "<tool_call><function=calc><parameter=expression>1+1</parameter></function></tool_call>trailing",
    '<tool_call>\n<function name="calc">\n<parameter name="expression">3</parameter>\n</function>\n</tool_call>',
    '<function name="calc"><param name=\'expression\'>4</param></function>',
    '<invoke name="calc"><parameter name="expression">5</parameter></invoke>',
    '<function_calls>\n<invoke name="calc"><param name="expression">6</param></invoke>\n<invoke name="calc"><param name="expression">7</param></invoke>\n</function_calls>',
    "<function_calls></function_calls>",
    "<function=calc>\n<parameter=expression>8</parameter>\n</function>\n<function=calc>\n<parameter=expression>9</parameter>\n</function>",
    "before <tool_call> quoted marker, then the real one:\n" + CALL("calc", [("expression", "10")]),
    "<tool_call> and <tool_call> and again " + CALL("calc", [("expression", "11")]),
    "<tool_call>\n<function=calc>\n<parameter=expression>\n12\n</parameter>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\n13\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\n14",
    "<tool_call>\n<function=calc>\n",
    "<tool_call>\n<function=calc",
    "<tool_call>\n<function=>\n</function></tool_call>",
    "<tool_call>\n<function=bad name!>\n</function></tool_call>",
    "<tool_call>\n<function=" + "a" * 70 + ">\n</function></tool_call>",
    "<tool_call>\n<function=unknown>\n<parameter=x>1</parameter>\n</function>\n</tool_call>",
    CALL("calc", [("expression", "a")]) + "\nI imagine the result is 3",
    CALL("calc", [("expression", "a")]) + "<tool_call>\n<function=calc>\n<parameter=expression>broken",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nuses </parameter> inside\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\n<parameter=nested>x</parameter>\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\n<html><body>x</parameter> y</body></html>\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\n1</parameter><think>\n</function></tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nx</parameter></think>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nx</parameter><think>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nx</parameter></function_calls>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nx</parameter></function><function=calc>\n</function></tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nx</parameter></function> trailing words</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nx</parameter></think>\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=expression>\nx</parameter></think>\ny</parameter>\n</function>\n</tool_call>",
    "<think>\nthinking about <tool_call> markers\n</think>\n" + CALL("calc", [("expression", "15")]),
    "<tool_call>\n<function=calc>\n<parameter>\nnoname</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\nstray text\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n</function>\n<function=calc>\n</function>\n</tool_call>",
    "<tool_call>\n<function=calc>\n<parameter=filename=\"x\" name=\"expression\">v</parameter>\n</function>\n</tool_call>",
    "<function ",
    "<function",
    "<tool_call>",
    "text <",
    b"x<tool_call>\n<function=calc>\n<parameter=expression>\nbad utf8 \xff\xfe\n</parameter>\n</function>\n</tool_call>",
    b"\xc3\x28 before " + CALL("calc", [("expression", "ok")]).encode(),
    "<tool_call>\n<function=calc>\n<parameter=expression>\n{\"a\": [1, 2, {\"b\": null}]}\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=typed>\n<parameter=o>\n{\"a\": [1, 2,}\n</parameter>\n<parameter=a>\n[1, 2\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=dup>\n<parameter=x>7</parameter>\n</function>\n</tool_call>",
    "   \n  " + CALL("calc", [("expression", "lead")]) + "  \n ",
]
MARKUP = ["<tool_call>", "</tool_call>", "<function=calc>", "</function>", "<parameter=expression>", "</parameter>", "<function_calls>",
          "</function_calls>", "<invoke name=\"calc\">", "</invoke>", "<param name=\"x\">", "</param>", "<think>", "</think>", "\n", " ", "\r\n", "<", ">", "=", "\"", "'", "{", "}"]


def mutate(rng, data: bytes) -> bytes:
    data = bytearray(data)
    for _ in range(rng.randint(1, 3)):
        kind = rng.randint(0, 4)
        pos = rng.randint(0, len(data))
        if kind == 0 and data:
            del data[pos:pos + rng.randint(1, 12)]
        elif kind == 1:
            data[pos:pos] = rng.choice(MARKUP).encode()
        elif kind == 2 and len(data) > 4:
            a, b = sorted(rng.sample(range(len(data)), 2))
            data[a:a] = data[b:b + rng.randint(1, 20)]
        elif kind == 3:
            data[pos:pos] = bytes([rng.choice([0xFF, 0xC3, 0x80, 0x00, 0x7F])])
        elif kind == 4 and data:
            del data[pos:]
    return bytes(data)


def hx(b: bytes) -> str:
    return b.hex()


def case(kind, contract, forced, max_len, chunks, enabled=True):
    tools = ",".join(hx(t.encode()) for t in contract)
    fields = [kind, "1" if enabled else "0", hx(forced.encode()), str(max_len), tools, ",".join(chunks)]
    return "\t".join(fields)


def run_both(oracle, tmp_path, lines):
    src = tmp_path / "cases.txt"
    src.write_text("\n".join(lines) + "\n")
    c = subprocess.run([str(oracle), str(src)], capture_output=True, text=True, timeout=300)
    assert c.returncode == 0, c.stderr[-500:]
    m = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(HERE / "tool_call_probe.mojo"), str(src)],
                       capture_output=True, text=True, timeout=1500)
    assert m.returncode == 0, m.stderr[-3000:]
    want, got = c.stdout.strip("\n").split("\n"), m.stdout.strip("\n").split("\n")
    assert len(want) == len(got) == len(lines), (len(want), len(got), len(lines))
    def first_diff(a, b):
        k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        return f"@{k}: C++ ...{a[max(0, k - 40):k + 60]} | Mojo ...{b[max(0, k - 40):k + 60]}"

    bad = [(i, lines[i].split("\t")[5][:200], first_diff(want[i], got[i]))
           for i in range(len(lines)) if want[i] != got[i]]
    assert not bad, f"{len(bad)} of {len(lines)} differ; first: {bad[:2]}"
    return want


def corpus_bytes():
    return [t if isinstance(t, bytes) else t.encode() for t in TEXTS]


def test_parse_corpus(oracle, tmp_path):
    texts = corpus_bytes()
    lines = [case("parse", contract, "", 64, [hx(t)]) for contract in CONTRACTS.values() for t in texts]
    lines += [case("parse", CONTRACTS["calc"], "", 8, [hx(t)]) for t in texts]
    lines += [case("parse", [], "", 64, [hx(texts[1])], enabled=False)]
    got = run_both(oracle, tmp_path, lines)
    assert sum(x.startswith("parse resp=1") for x in got) > 100
    assert sum(x == "EXC" for x in got) >= 3  # invalid UTF-8 in an encoded value throws in both


@pytest.mark.parametrize("seed", [1, 2])
def test_parse_damaged(oracle, tmp_path, seed):
    rng = random.Random(seed)
    base = corpus_bytes()
    lines = []
    for _ in range(1200):
        t = mutate(rng, rng.choice(base))
        lines.append(case("parse", CONTRACTS[rng.choice(list(CONTRACTS))], "", rng.choice([64, 64, 6]), [hx(t)]))
    got = run_both(oracle, tmp_path, lines)
    assert sum(x.startswith("parse resp=1") for x in got) > 300


@pytest.mark.parametrize("forced", ["", "calc"])
def test_stream_decoder(oracle, tmp_path, forced):
    rng = random.Random(7 if forced else 8)
    base = corpus_bytes()
    lines = []
    for _ in range(900):
        t = mutate(rng, rng.choice(base)) if rng.random() < 0.6 else rng.choice(base)
        if forced:
            # the prompt carries the opener, so the model continues inside the call
            t = rng.choice([b"<parameter=expression>\n1+2\n</parameter>\n</function>\n</tool_call>",
                            b"<parameter=expression>\n1+2\n</parameter>\n</function>",
                            b"<parameter=precision>\n3\n</parameter>\n", b"text only", t])
        cuts = sorted(rng.sample(range(len(t) + 1), min(len(t) + 1, rng.randint(0, 5)))) if t else []
        chunks, last = [], 0
        for c in cuts + [len(t)]:
            chunks.append(hx(t[last:c]))
            last = c
        if rng.random() < 0.15:
            chunks.insert(rng.randint(0, len(chunks)), "R")
        lines.append(case("stream", CONTRACTS[rng.choice(["calc", "typed", "none", "dup"])], forced, 64, chunks))
    got = run_both(oracle, tmp_path, lines)
    assert sum("T content" in x for x in got) > 800


def test_structured_format(oracle, tmp_path):
    """The constrained-decoding grammar, byte for byte, for every contract (including the ones the
    C++ rejects: ambiguous duplicate definitions and parameter names holding '>')."""
    gt = tool("gt", {"a>b": {"type": "string"}})
    named = tool("tëst_é", {"é": {"type": "string"}, 'q"uote': {"type": "integer"}, "b\\s": {}})
    contracts = dict(CONTRACTS, gt=[gt], named=[named, CALC], only_dup=[DUP_A, DUP_B], empty_name=[tool("x", {"": {"type": "string"}})])
    lines = [case("format", c, "", 64, ["00"]) for c in contracts.values()]
    lines += [case("format", [], "", 64, ["00"], enabled=False)]
    got = run_both(oracle, tmp_path, lines)
    assert sum(x.startswith("format ") and len(x) > 20 for x in got) >= 4
    assert sum(x == "EXC" for x in got) >= 3


def test_deeply_nested_values(oracle, tmp_path):
    """`Json::accept` has no depth limit; neither does the Mojo reader on the tool-call path."""
    texts = []
    for depth in (100, 511, 513, 5000, 60000):
        for open_, close in (("[", "]"), ('{"a":', "}")):
            value = open_ * depth + "1" + close * depth
            texts.append(f"<tool_call>\n<function=typed>\n<parameter=a>\n{value}\n</parameter>\n</function>\n</tool_call>".encode())
    texts.append(b"<tool_call>\n<function=typed>\n<parameter=a>\n" + b"[" * 3000 + b"\n</parameter>\n</function>\n</tool_call>")
    lines = [case("parse", CONTRACTS["typed"], "", 64, [hx(t)]) for t in texts]
    got = run_both(oracle, tmp_path, lines)
    assert sum(x.startswith("parse resp=1") for x in got) == len(texts)


def _deep_tool(depth):
    nested = '{"a":' * depth + "{}" + "}" * depth
    return tool("deep", {"x": {"type": "string"}}).replace('"type": "string"', '"type": "string", "extra": ' + nested)


def test_deeply_nested_tool_definitions(oracle, tmp_path):
    """Deep tools read as the C++ reads them, up to well below where the C++ itself runs out of stack."""
    lines = [case("parse", [_deep_tool(d)], "", 64, [hx(CALL("deep", [("x", "v")]).encode())])
             for d in (100, 511, 513, 2000, 15000, 36000)]
    got = run_both(oracle, tmp_path, lines)
    assert all(x.startswith("parse resp=1") for x in got), got[:2]


def test_tool_nested_past_the_limit_is_discarded(tmp_path):
    """The limit is the last depth the C++ oracle reads (37,319 containers: 6 around a schema nested 37,313
    deep). One level more and the tool is discarded, however far it goes, without overflowing our stack."""
    call = hx(CALL("deep", [("x", "v")]).encode())
    src = tmp_path / "c.txt"
    src.write_text("\n".join(case("parse", [_deep_tool(d)], "", 64, [call]) for d in (37313, 37314, 1_000_000)) + "\n")
    r = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(HERE / "tool_call_probe.mojo"), str(src)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    at, past, far = r.stdout.strip("\n").split("\n")
    assert past == far and past != at, "37,313 deep is a tool; 37,314 and beyond is not"
