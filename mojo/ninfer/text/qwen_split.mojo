"""The Qwen3.5 pre-tokenizer split, written as parser combinators.

The tokenizer's Split pre-tokenizer is this regular expression:

    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}
    | ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

A regex engine backs off when a greedy run overshoots; combinators do not, so each place where
that matters is written as the match the backing-off produces:

- `[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+`: a mark can be the optional prefix or the first letter,
  so the alternatives are "prefix then letters" and "letters".
- `\\s*[\\r\\n]+` ends at the last newline of the whitespace run: consume whitespace while a newline
  still follows, then the newlines.
- `\\s+(?!\\S)` is the whole run at the end of the text, otherwise the run without its last
  character (so a run of one never matches).
"""

from std.collections import List

from ninfer.parsec.parsec import Parsec, U_LETTER, U_MARK, U_NUMBER, U_SPACE, U_NEWLINE
from ninfer.text.unicode import Unicode, utf8_codepoint_at

comptime _NAME = "qwen_split"


def _ci_char(mut p: Parsec, ch: Int) -> Int:
    var ranges = List[Int]()
    ranges.append(ch)
    ranges.append(ch)
    ranges.append(ch - 32)
    ranges.append(ch - 32)
    return p.ascii_char(ranges)


def _ci_word(mut p: Parsec, word: String) -> Int:
    var out = p.empty()
    for b in word.as_bytes():
        out = p.concat(out, _ci_char(p, Int(b)))
    return out


def _one_or_more(mut p: Parsec, c: Int) -> Int:
    return p.concat(c, p.repeat(c))


def _alts(mut p: Parsec, parts: List[Int]) -> Int:
    return p.choice(parts)


def build_qwen_split() -> Parsec:
    var p = Parsec()
    var space = p.uclass(U_SPACE)
    var newline = p.uclass(U_NEWLINE)
    var letter_mark = p.uclass(U_LETTER | U_MARK)

    var contraction_tail = List[Int]()
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"]:
        contraction_tail.append(_ci_word(p, String(suffix)))
    var contraction = p.concat(p.string("'"), p.choice(contraction_tail))

    var prefixed = List[Int]()
    prefixed.append(p.concat(p.uclass(0, U_NEWLINE | U_LETTER | U_NUMBER), _one_or_more(p, letter_mark)))
    prefixed.append(_one_or_more(p, letter_mark))
    var word = p.choice(prefixed)

    var number = p.uclass(U_NUMBER)

    var punctuation = p.concat(
        p.concat(p.optional(p.string(" ")), _one_or_more(p, p.uclass(0, U_SPACE | U_LETTER | U_MARK | U_NUMBER))),
        p.repeat(newline),
    )

    var newline_ahead = p.lookahead(p.concat(p.repeat(p.uclass(U_SPACE, U_NEWLINE)), newline))
    var line_break = p.concat(p.repeat(p.concat(space, newline_ahead)), _one_or_more(p, newline))

    var trailing = List[Int]()
    trailing.append(p.concat(_one_or_more(p, space), p.eos()))
    trailing.append(_one_or_more(p, p.concat(space, p.lookahead(space))))
    var spaces_before_word = p.choice(trailing)

    var any_spaces = _one_or_more(p, space)

    var top = List[Int]()
    top.append(contraction)
    top.append(word)
    top.append(number)
    top.append(punctuation)
    top.append(line_break)
    top.append(spaces_before_word)
    top.append(any_spaces)
    p.define(_NAME, p.choice(top))
    return p^


def qwen_word_end(
    split: Parsec, text: List[UInt8], begin: Int, unicode: Unicode, context: String
) raises -> Int:
    """End of the piece that starts at `begin`; a character no alternative takes is a piece alone."""
    var end = split.scan(_NAME, text, begin, unicode, context)
    if end > begin:
        return end
    return begin + _codepoint_length(text, begin, context)


def _codepoint_length(text: List[UInt8], begin: Int, context: String) raises -> Int:
    return utf8_codepoint_at(text, begin, context).length
