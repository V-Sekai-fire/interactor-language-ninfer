#!/usr/bin/env python3
"""Generate mojo/ninfer/text/unicode_tables.mojo from the repo's own utf8proc.

The C++ engine takes its Unicode categories and NFC from third_party/utf8proc, so the Mojo
tables are read out of that library (compiled with the system C compiler) rather than from
another Unicode database that may be a different version.

    python3 mojo/tools/gen_unicode_tables.py
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
UTF8PROC = ROOT / "third_party" / "utf8proc"
OUT = ROOT / "mojo" / "ninfer" / "text" / "unicode_tables.mojo"

DUMP_C = r"""
#include "utf8proc.h"
#include <stdio.h>
int main(void) {
  printf("version %s\n", utf8proc_unicode_version());
  for (int cp = 0; cp <= 0x10FFFF; cp++) {
    if (cp >= 0xD800 && cp <= 0xDFFF) continue;
    const utf8proc_property_t *p = utf8proc_get_property(cp);
    unsigned char in[5]; int n = utf8proc_encode_char(cp, in);
    utf8proc_uint8_t *c = 0, *d = 0;
    utf8proc_ssize_t rc = utf8proc_map(in, n, &c, UTF8PROC_STABLE | UTF8PROC_COMPOSE);
    utf8proc_ssize_t rd = utf8proc_map(in, n, &d, UTF8PROC_STABLE | UTF8PROC_DECOMPOSE);
    printf("%x %d %d %x %x %x %d %d ", cp, p->category, p->combining_class,
           utf8proc_tolower(cp), utf8proc_toupper(cp), utf8proc_totitle(cp),
           utf8proc_isupper(cp) ? 1 : 0, utf8proc_islower(cp) ? 1 : 0);
    for (int i = 0; i < rc; i++) printf("%02x", c[i]);
    printf(" ");
    for (int i = 0; i < rd; i++) printf("%02x", d[i]);
    printf("\n");
  }
  return 0;
}
"""

# utf8proc_category_t -> class used by the engine: 1 letter, 2 mark, 3 number, 4 separator.
def klass(category: int) -> int:
    if 1 <= category <= 5:
        return 1
    if 6 <= category <= 8:
        return 2
    if 9 <= category <= 11:
        return 3
    if 23 <= category <= 25:
        return 4
    return 0


def decode(hexstr: str) -> list[int]:
    return [ord(ch) for ch in bytes.fromhex(hexstr).decode("utf-8")]


def ranges(values: dict[int, int]) -> str:
    out, start, prev, cur = [], None, None, None
    for cp in sorted(values):
        v = values[cp]
        if start is not None and cp == prev + 1 and v == cur:
            prev = cp
            continue
        if start is not None:
            out.append(f"{start:x} {prev:x} {cur}")
        start, prev, cur = cp, cp, v
    if start is not None:
        out.append(f"{start:x} {prev:x} {cur}")
    return "\n".join(out)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / "dump.c", Path(tmp) / "dump"
        src.write_text(DUMP_C)
        subprocess.run(
            ["cc", "-O1", f"-I{UTF8PROC}", str(src), str(UTF8PROC / "utf8proc.c"), "-o", str(exe)],
            check=True,
        )
        lines = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines()
    version = lines[0].split()[1]
    cats: dict[int, int] = {}
    ccc: dict[int, int] = {}
    nfd: dict[int, list[int]] = {}
    nfc: dict[int, list[int]] = {}
    simple_case: dict[int, tuple[int, int, int]] = {}
    is_upper: dict[int, int] = {}
    is_lower: dict[int, int] = {}
    for line in lines[1:]:
        cp_s, cat_s, ccc_s, lo_s, up_s, ti_s, isup_s, islo_s, c_hex, d_hex = line.split(" ")
        cp = int(cp_s, 16)
        if (int(lo_s, 16), int(up_s, 16), int(ti_s, 16)) != (cp, cp, cp):
            simple_case[cp] = (int(lo_s, 16), int(up_s, 16), int(ti_s, 16))
        if isup_s == "1":
            is_upper[cp] = 1
        if islo_s == "1":
            is_lower[cp] = 1
        if klass(int(cat_s)):
            cats[cp] = klass(int(cat_s))
        if int(ccc_s):
            ccc[cp] = int(ccc_s)
        d, c = decode(d_hex), decode(c_hex)
        if d != [cp]:
            nfd[cp] = d
        nfc[cp] = c
    # Hangul syllables decompose algorithmically; keep them out of the table.
    for cp in [c for c in nfd if 0xAC00 <= c <= 0xD7A3]:
        del nfd[cp]
    # Primary composites: stay themselves under NFC yet decompose. Their pair is
    # (NFC of every part but the last, last part).
    primary = {tuple(d): cp for cp, d in nfd.items() if nfc[cp] == [cp] and len(d) >= 2}
    pairs = []
    for cp, d in sorted(nfd.items(), key=lambda item: (len(item[1]), item[0])):
        if nfc[cp] != [cp] or len(d) < 2:
            continue
        prefix = tuple(d[:-1])
        head = prefix[0] if len(prefix) == 1 else primary[prefix]
        pairs.append((head, d[-1], cp))
    pairs.sort(key=lambda t: t[2])
    jinja_src = (ROOT / "third_party" / "llama-jinja" / "jinja" / "unicode_data.h").read_text()
    special = []
    for m in re.finditer(r'\{0x([0-9A-Fa-f]+), "((?:\\x[0-9a-f]{2})+)", "((?:\\x[0-9a-f]{2})+)", "((?:\\x[0-9a-f]{2})+)"\}', jinja_src):
        cp, lo, ti, up = m.group(1), *(m.group(i).replace("\\x", "") for i in (2, 3, 4))
        special.append(f"{int(cp, 16):x} {lo} {ti} {up}")
    def jinja_ranges(name: str) -> str:
        body = jinja_src.split(f"{name}[] = {{")[1].split("};")[0]
        return "\n".join(f"{int(a, 16):x} {int(b, 16):x} 1" for a, b in re.findall(r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}", body))
    case_lines = "\n".join(f"{cp:x} {lo:x} {up:x} {ti:x}" for cp, (lo, up, ti) in sorted(simple_case.items()))
    pair_lines = "\n".join(f"{a:x} {b:x} {c:x}" for a, b, c in pairs)
    decomp_lines = "\n".join(f"{cp:x}:" + " ".join(f"{x:x}" for x in d) for cp, d in sorted(nfd.items()))
    OUT.write_text(
        f'''"""Unicode {version} tables read out of third_party/utf8proc. Generated by
mojo/tools/gen_unicode_tables.py; do not edit.

Classes: 1 letter (L*), 2 mark (M*), 3 number (N*), 4 separator (Z*). Ranges are `first last value`
in hex. `DECOMPOSITIONS` is `cp:parts...` (full canonical decomposition, Hangul excluded).
`COMPOSITIONS` is `first second composite` for every primary composite.
"""

comptime UNICODE_VERSION = "{version}"

comptime CLASS_RANGES = """{ranges(cats)}
"""

comptime CCC_RANGES = """{ranges(ccc)}
"""

comptime DECOMPOSITIONS = """{decomp_lines}
"""

comptime COMPOSITIONS = """{pair_lines}
"""

# Jinja string methods (src/text/jinja via third_party/llama-jinja/jinja/unicode.cpp).
# SIMPLE_CASE: `cp lower upper title` where utf8proc's simple mapping differs from cp.
# SPECIAL_CASE: `cp lower title upper`, UTF-8 bytes as hex, from SpecialCasing.txt.
comptime SIMPLE_CASE = """{case_lines}
"""

comptime SPECIAL_CASE = """{chr(10).join(special)}
"""

comptime IS_UPPER_RANGES = """{ranges(is_upper)}
"""

comptime IS_LOWER_RANGES = """{ranges(is_lower)}
"""

comptime CASED_RANGES = """{jinja_ranges("cased")}
"""

comptime CASE_IGNORABLE_RANGES = """{jinja_ranges("case_ignorable")}
"""
'''
    )
    print(f"wrote {OUT} (Unicode {version}): {len(cats)} class cps, {len(ccc)} ccc cps, "
          f"{len(nfd)} decompositions, {len(pairs)} compositions")


if __name__ == "__main__":
    main()
