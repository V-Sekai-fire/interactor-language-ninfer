"""Parity: the Mojo parser combinators (`ninfer.combinator`) against dashbitco/nimble_parsec itself.

Random combinator trees are rendered twice: as Elixir `defparsec` modules compiled against
the real library, and as specs the Mojo port builds. Every input must give the same result,
rest, line, byte offset and, on failure, the same error text.
Needs `elixir` and a checkout of nimble_parsec (NIMBLE_PARSEC_DIR, or a clone made on demand).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOJO = shutil.which("mojo")
ELIXIR = shutil.which("elixir")
pytestmark = pytest.mark.skipif(MOJO is None or ELIXIR is None or shutil.which("git") is None,
                                reason="needs mojo, elixir and git")

ALPHABET = list("ab1 -\n,\u00e9xZ9")


def nimble_dir(tmp_path_factory) -> Path:
    given = os.environ.get("NIMBLE_PARSEC_DIR")
    path = Path(given) if given else Path(tmp_path_factory.mktemp("nimble")) / "nimble_parsec"
    if not path.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/dashbitco/nimble_parsec", str(path)], check=True)
    if not (path / "_build/prod/lib/nimble_parsec/ebin").exists():
        subprocess.run(["mix", "compile"], cwd=path, env={**os.environ, "MIX_ENV": "prod"}, check=True, capture_output=True)
    return path


# --- spec generation -----------------------------------------------------------------------------


class Gen:
    """Builds specs together with what each produces: `min` bytes consumed, `ints` (only integers
    are emitted) and `single` (exactly one token)."""

    def __init__(self, rng, defs_available):
        self.rng = rng
        self.defs = defs_available
        # NimbleParsec 1.4 rejects some inputs that `lookahead` inside `eventually` should accept
        # (a lookahead after an unbound choice there drops the retry); the port implements the
        # documented behaviour, so the generator keeps the two apart.
        self.no_lookahead = False

    def ranges(self, utf8):
        pools = [(48, 57), (97, 122), (97, 99), (65, 90), (32, 32), (45, 45), (10, 10), (44, 44)]
        if utf8:
            pools.append((0xE9, 0xE9))
            pools.append((0x100, 0x2FF))
        inc = self.rng.sample(pools, self.rng.randint(0 if utf8 else 1, 2))
        exc = self.rng.sample([(98, 98), (49, 50), (10, 10)], self.rng.randint(0, 1)) if inc else []
        return [list(r) for r in inc], [list(r) for r in exc]

    def leaf(self, want_ints=False):
        r = self.rng
        kind = r.choice(["ascii_char", "utf8_char", "integer", "integer_range"] if want_ints else
                        ["string", "string", "ascii_char", "utf8_char", "integer", "integer_range", "ascii_string", "utf8_string", "bytes", "eos"])
        if kind == "string":
            s = "".join(r.choice(ALPHABET) for _ in range(r.randint(1, 3)))
            return {"op": "string", "v": s}, dict(min=len(s.encode()), ints=False, single=True)
        if kind in ("ascii_char", "utf8_char"):
            inc, exc = self.ranges(kind == "utf8_char")
            return {"op": kind, "incl": inc, "excl": exc}, dict(min=1, ints=True, single=True)
        if kind == "integer":
            return {"op": "integer", "n": r.randint(1, 3)}, dict(min=1, ints=True, single=True)
        if kind == "integer_range":
            mn = r.randint(1, 2)
            return {"op": "integer", "min": mn, "max": r.choice([None, mn + r.randint(1, 2)])}, dict(min=1, ints=True, single=True)
        if kind in ("ascii_string", "utf8_string"):
            inc, _ = self.ranges(kind == "utf8_string")
            if r.random() < 0.4:
                return {"op": kind, "incl": inc, "n": r.randint(1, 3)}, dict(min=1, ints=False, single=True)
            mn = r.randint(0, 2)
            mx = r.choice([None, mn + r.randint(1, 2)])
            return {"op": kind, "incl": inc, "min": mn, "max": mx}, dict(min=mn, ints=False, single=True)
        if kind == "bytes":
            n = r.randint(1, 3)
            return {"op": "bytes", "n": n}, dict(min=n, ints=False, single=True)
        if kind == "eos":
            return {"op": "eos"}, dict(min=0, ints=False, single=False)
        return {"op": "eos"}, dict(min=0, ints=False, single=False)

    def make(self, depth, want_ints=False, need_nonempty=False, need_single=False):
        r = self.rng
        for _ in range(30):
            spec, info = self._make(depth, want_ints)
            if need_nonempty and info["min"] == 0:
                continue
            if need_single and not info["single"]:
                continue
            return spec, info
        return {"op": "ascii_char", "incl": [[97, 122]], "excl": []}, dict(min=1, ints=True, single=True)

    def _make(self, depth, want_ints):
        r = self.rng
        if depth <= 0 or r.random() < 0.3:
            return self.leaf(want_ints)
        pool = ["concat", "concat", "choice", "optional", "times", "repeat", "eventually", "lookahead", "lookahead_not",
                       "ignore", "replace", "tag", "unwrap_and_tag", "wrap", "label", "line", "byte_offset", "reduce_sum", "map_inc",
                       "post_count", "duplicate", "parsec"]
        if self.no_lookahead:
            pool = [x for x in pool if not x.startswith("lookahead")]
        op = r.choice(pool if not want_ints else ["concat", "choice", "optional", "times", "reduce_sum", "map_inc", "ignore"])
        if op == "parsec":
            if not self.defs:
                return self.leaf(want_ints)
            name = r.choice(sorted(self.defs))
            return {"op": "parsec", "name": name}, dict(self.defs[name])
        if op == "concat":
            parts = [self.make(depth - 1, want_ints) for _ in range(r.randint(2, 3))]
            return {"op": "concat", "cs": [p[0] for p in parts]}, dict(min=sum(p[1]["min"] for p in parts), ints=all(p[1]["ints"] for p in parts), single=False)
        if op == "choice":
            parts = [self.make(depth - 1, want_ints) for _ in range(r.randint(2, 3))]
            return {"op": "choice", "cs": [p[0] for p in parts]}, dict(min=min(p[1]["min"] for p in parts), ints=all(p[1]["ints"] for p in parts), single=False)
        c, ci = self.make(depth - 1, want_ints)
        if op == "optional":
            return {"op": "optional", "c": c}, dict(min=0, ints=ci["ints"], single=False)
        if op in ("repeat", "eventually", "times", "duplicate"):
            saved = self.no_lookahead
            self.no_lookahead = saved or op == "eventually"
            c, ci = self.make(depth - 1, want_ints, need_nonempty=True)
            self.no_lookahead = saved
            if op == "repeat":
                return {"op": "repeat", "c": c}, dict(min=0, ints=ci["ints"], single=False)
            if op == "eventually":
                return {"op": "eventually", "c": c}, dict(min=0, ints=False, single=False)
            if op == "duplicate":
                n = r.randint(1, 3)
                return {"op": "duplicate", "c": c, "n": n}, dict(min=ci["min"] * n, ints=ci["ints"], single=False)
            if r.random() < 0.4:
                n = r.randint(1, 3)
                return {"op": "times", "c": c, "n": n}, dict(min=ci["min"] * n, ints=ci["ints"], single=False)
            mn = r.randint(0, 2)
            mx = r.choice([None, mn + r.randint(1, 2)])
            return {"op": "times", "c": c, "min": mn, "max": mx}, dict(min=ci["min"] * mn, ints=ci["ints"], single=False)
        if op in ("lookahead", "lookahead_not"):
            return {"op": op, "c": c}, dict(min=0, ints=True, single=False)
        if op == "ignore":
            return {"op": "ignore", "c": c}, dict(min=ci["min"], ints=True, single=False)
        if op == "replace":
            return {"op": "replace", "c": c, "v": r.choice(["R", 7])}, dict(min=ci["min"], ints=False, single=True)
        if op == "tag":
            return {"op": "tag", "c": c, "tag": r.choice(["t", "u"])}, dict(min=ci["min"], ints=False, single=True)
        if op == "unwrap_and_tag":
            c, ci = self.make(depth - 1, want_ints, need_single=True)
            return {"op": "unwrap_and_tag", "c": c, "tag": "w"}, dict(min=ci["min"], ints=False, single=True)
        if op == "wrap":
            return {"op": "wrap", "c": c}, dict(min=ci["min"], ints=False, single=True)
        if op == "label":
            return {"op": "label", "c": c, "label": r.choice(["thing", "a b", "x\"y"])}, dict(min=ci["min"], ints=ci["ints"], single=ci["single"])
        if op in ("line", "byte_offset"):
            return {"op": op, "c": c}, dict(min=ci["min"], ints=False, single=True)
        if op in ("reduce_sum", "map_inc"):
            c, ci = self.make(depth - 1, True)
            if op == "reduce_sum":
                return {"op": op, "c": c}, dict(min=ci["min"], ints=True, single=True)
            return {"op": op, "c": c}, dict(min=ci["min"], ints=True, single=False)
        return {"op": "post_count", "c": c}, dict(min=ci["min"], ints=True, single=True)


def elixir_expr(spec) -> str:
    def rng_list(rs, excl=False):
        out = []
        for lo, hi in rs:
            body = str(lo) if lo == hi else f"{lo}..{hi}"
            out.append(f"{{:not, {body}}}" if excl else body)
        return out

    def count_opts(spec):
        if "n" in spec:
            return str(spec["n"])
        parts = [f"min: {spec.get('min', 0)}"] + ([f"max: {spec['max']}"] if spec.get("max") is not None else [])
        return "[" + ", ".join(parts) + "]"

    op = spec["op"]
    if op == "empty":
        return "empty()"
    if op == "string":
        return f"string({json.dumps(spec['v'], ensure_ascii=False)})"
    if op in ("ascii_char", "utf8_char"):
        return f"{op}([{', '.join(rng_list(spec['incl']) + rng_list(spec['excl'], True))}])"
    if op == "bytes":
        return f"bytes({spec['n']})"
    if op == "eos":
        return "eos()"
    if op == "integer":
        return f"integer({spec['n']})" if "n" in spec else f"integer({count_opts(spec)})"
    if op in ("ascii_string", "utf8_string"):
        return f"{op}([{', '.join(rng_list(spec['incl']))}], {count_opts(spec)})"
    if op == "concat":
        return "|> concat(".join([elixir_expr(spec["cs"][0])] + [elixir_expr(c) + ")" for c in spec["cs"][1:]]) if False else \
            _concat([elixir_expr(c) for c in spec["cs"]])
    if op == "choice":
        return "choice([" + ", ".join(elixir_expr(c) for c in spec["cs"]) + "])"
    c = elixir_expr(spec["c"]) if "c" in spec else None
    if op == "optional":
        return f"optional({c})"
    if op == "times":
        return f"times({c}, {count_opts(spec)})"
    if op == "duplicate":
        return f"duplicate({c}, {spec['n']})"
    if op in ("repeat", "eventually", "lookahead", "lookahead_not", "ignore", "wrap", "line", "byte_offset"):
        return f"{op}({c})"
    if op == "replace":
        return f"replace({c}, {json.dumps(spec['v'])})"
    if op in ("tag", "unwrap_and_tag"):
        return f"{op}({c}, :{spec['tag']})"
    if op == "label":
        return f"label({c}, {json.dumps(spec['label'])})"
    if op == "reduce_sum":
        return f"reduce({c}, {{Enum, :sum, []}})"
    if op == "map_inc":
        return f"map({c}, {{Kernel, :+, [1]}})"
    if op == "post_count":
        return f"post_traverse({c}, {{Helper, :count, []}})"
    if op == "parsec":
        return f"parsec(:{spec['name']})"
    raise AssertionError(op)


def _concat(parts):
    out = parts[0]
    for p in parts[1:]:
        out = f"concat({out}, {p})"
    return out


# --- inputs ----------------------------------------------------------------------------------------


def sample(spec, rng):
    """A string the spec is likely to accept."""
    op = spec["op"]
    if op in ("string",):
        return spec["v"]
    if op in ("ascii_char", "utf8_char"):
        if spec["incl"]:
            lo, hi = rng.choice(spec["incl"])
            cp = rng.randint(lo, hi)
        else:
            cp = rng.choice([97, 49, 32, 0xE9] if op == "utf8_char" else [97, 49, 32])
        for lo, hi in spec["excl"]:
            if lo <= cp <= hi:
                cp = 0x7A if not (spec["incl"] and not any(a <= 0x7A <= b for a, b in spec["incl"])) else cp
        return chr(cp)
    if op == "integer":
        return str(rng.randint(0, 99))
    if op in ("ascii_string", "utf8_string"):
        n = spec.get("n", spec.get("min", 0) + rng.randint(0, 2))
        lo, hi = rng.choice(spec["incl"]) if spec["incl"] else (97, 122)
        return "".join(chr(rng.randint(lo, hi)) for _ in range(n))
    if op == "bytes":
        return "q" * spec["n"]
    if op in ("concat",):
        return "".join(sample(c, rng) for c in spec["cs"])
    if op == "choice":
        return sample(rng.choice(spec["cs"]), rng)
    if op in ("optional",):
        return sample(spec["c"], rng) if rng.random() < 0.6 else ""
    if op in ("repeat", "times", "duplicate"):
        n = spec.get("n", rng.randint(0, 3))
        return "".join(sample(spec["c"], rng) for _ in range(n))
    if op == "eventually":
        return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 3))) + sample(spec["c"], rng)
    if "c" in spec:
        return sample(spec["c"], rng)
    return ""


def make_cases(seed, count):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        gen = Gen(rng, {})
        defs = {}
        for name in rng.sample(["alpha", "beta"], rng.randint(0, 2)):
            spec, info = gen.make(2, need_nonempty=True)
            defs[name] = spec
            gen.defs[name] = info
        main, _ = gen.make(rng.randint(1, 4))
        inputs = []
        for _ in range(6):
            s = sample(main, rng)
            if rng.random() < 0.35:
                s = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 8)))
            elif rng.random() < 0.3:
                s += "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 3)))
            inputs.append(s)
        cases.append({"defs": defs, "main": main, "inputs": inputs})
    return cases


def elixir_module(i, case):
    lines = [f"defmodule Generated.P{i} do", "  import NimbleParsec"]
    for name, spec in case["defs"].items():
        lines.append(f"  defcombinatorp :{name}, {elixir_expr(spec)}")
    lines.append(f"  defparsec :main, {elixir_expr(case['main'])}")
    lines.append("end")
    return "\n".join(lines)


def run_both(tmp_path, nimble, cases):
    source = tmp_path / "generated.exs"
    source.write_text("\n".join(elixir_module(i, c) for i, c in enumerate(cases)) + "\n")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("\n".join(json.dumps(c["inputs"]) for c in cases) + "\n")
    specs = tmp_path / "specs.jsonl"
    specs.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    ebin = nimble / "_build/prod/lib/nimble_parsec/ebin"
    theirs = subprocess.run([ELIXIR, "-pa", str(ebin), str(HERE / "nimble_parsec_oracle.exs"), str(source), str(inputs)], capture_output=True, text=True)
    assert theirs.returncode == 0, theirs.stderr[-3000:]
    mine = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(HERE / "combinator_probe.mojo"), str(specs)], capture_output=True, text=True)
    assert mine.returncode == 0, mine.stderr[-3000:]
    return mine.stdout.strip().split("\n"), theirs.stdout.strip().split("\n")


@pytest.fixture(scope="module")
def nimble(tmp_path_factory):
    return nimble_dir(tmp_path_factory)


def test_documented_examples(nimble, tmp_path):
    date = {"op": "concat", "cs": [{"op": "integer", "n": 4}, {"op": "ignore", "c": {"op": "string", "v": "-"}}, {"op": "integer", "n": 2},
                                   {"op": "ignore", "c": {"op": "string", "v": "-"}}, {"op": "integer", "n": 2}]}
    cases = [{"defs": {}, "main": date, "inputs": ["2010-04-17", "2010-4-17", "", "2010-04-1x"]},
             {"defs": {}, "main": {"op": "concat", "cs": [{"op": "ascii_char", "incl": [[48, 57]], "excl": []}, {"op": "ascii_char", "incl": [[97, 122]], "excl": []}]},
              "inputs": ["1a", "a1", "1"]},
             {"defs": {}, "main": {"op": "eventually", "c": {"op": "integer", "min": 1, "max": 2}}, "inputs": ["let's meet at 12?", "none"]}]
    mine, theirs = run_both(tmp_path, nimble, cases)
    assert mine == theirs
    assert any(l.startswith("err") for l in theirs) and any(l.startswith("ok") for l in theirs)


@pytest.mark.parametrize("seed", [20261114, 7, 424242])
def test_random_combinators(nimble, tmp_path, seed):
    cases = make_cases(seed, 400)
    mine, theirs = run_both(tmp_path, nimble, cases)
    assert len(mine) == len(theirs)
    bad = [i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    assert not bad, [(mine[i], theirs[i]) for i in bad[:5]] + [f"{len(bad)} of {len(theirs)} lines differ"]
    oks = sum(l.startswith("ok") for l in theirs)
    errs = sum(l.startswith("err") for l in theirs)
    assert oks > 300 and errs > 300, (oks, errs)
