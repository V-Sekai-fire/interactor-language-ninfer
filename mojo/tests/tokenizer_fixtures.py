"""Synthetic tokenizer.json files and case generators for the tokenizer parity test.

Two small tokenizers are trained here so the test needs no downloads: a byte-level BPE in the
Qwen3.5 layout and a SentencePiece-style one in the Gemma 4 layout. The case generator builds
inputs that stress every branch of the pre-tokenizer, the added-token scanner and the boundary
bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re

QWEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CORPUS = (
    "The quick brown fox jumps over the lazy dog. Hello, world! It's what they're saying; we'll see.\n"
    "def add(a, b):\n    return a + b\n\nprint(add(2, 40))  # 42\n"
    "Caf\u00e9 cr\u00e8me br\u00fbl\u00e9e na\u00efve fa\u00e7ade. \u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440! \u4f60\u597d\uff0c\u4e16\u754c\u3002 \u3053\u3093\u306b\u3061\u306f\n"
    "emoji \U0001f600\U0001f680 and numbers 3.14159 1234567890 \u0663\u0664\u0665\n"
    "    indented    text\t\twith\ttabs \r\n windows line\r\n\r\nends\n"
) * 4


def byte_level_alphabet() -> list[str]:
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    out, extra = [], 0
    for byte in range(256):
        if byte in visible:
            out.append(chr(byte))
        else:
            out.append(chr(256 + extra))
            extra += 1
    return out


def train_byte_bpe(text: str, merges: int) -> tuple[dict[str, int], list[list[str]]]:
    alphabet = byte_level_alphabet()
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    words: dict[tuple[str, ...], int] = {}
    for word in re.findall(r"\s*\S+|\s+", text):
        symbols = tuple(alphabet[b] for b in word.encode())
        words[symbols] = words.get(symbols, 0) + 1
    rules: list[list[str]] = []
    for _ in range(merges):
        counts: dict[tuple[str, str], int] = {}
        for symbols, n in words.items():
            for pair in zip(symbols, symbols[1:]):
                counts[pair] = counts.get(pair, 0) + n
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        rules.append(list(best))
        vocab[best[0] + best[1]] = len(vocab)
        merged = {}
        for symbols, n in words.items():
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                    out.append(best[0] + best[1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            merged[tuple(out)] = merged.get(tuple(out), 0) + n
        words = merged
    return vocab, rules


ADDED = [
    ("<|endoftext|>", True),
    ("<|im_start|>", True),
    ("<|im_end|>", True),
    ("<tool_call>", False),
    ("<tool_call_x>", False),
]


def write_qwen(directory: Path, merge_style: str = "array") -> tuple[Path, Path, Path]:
    vocab, rules = train_byte_bpe(CORPUS, 900)
    first = len(vocab) + 7  # leave unused ids so decode must reject them
    added = [{"id": first + i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": sp} for i, (c, sp) in enumerate(ADDED)]
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN_PATTERN}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab,
                  "merges": rules if merge_style == "array" else [" ".join(r) for r in rules]},
    }
    config = {"added_tokens_decoder": {str(a["id"]): {k: a[k] for k in
              ("content", "single_word", "lstrip", "rstrip", "normalized", "special")} for a in added}}
    generation = {"eos_token_id": [added[2]["id"], added[0]["id"]]}
    paths = []
    for name, data in (("tokenizer.json", tokenizer), ("tokenizer_config.json", config), ("generation_config.json", generation)):
        path = directory / name
        path.write_text(json.dumps(data, ensure_ascii=False))
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


def write_sentencepiece(directory: Path) -> tuple[Path, Path, Path]:
    space = "\u2581"
    chars = sorted({c for c in CORPUS.replace(" ", space) if c not in "\r\n\t"} | {space})
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for c in chars:
        vocab.setdefault(c, len(vocab))
    # Merges over frequent pairs of the corpus, greedily, in rank order.
    words = [list(w) for w in CORPUS.replace(" ", space).split("\n") if w]
    rules = []
    for _ in range(300):
        counts: dict[tuple[str, str], int] = {}
        for w in words:
            for pair in zip(w, w[1:]):
                if pair[0] in vocab and pair[1] in vocab:
                    counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        if best[0] + best[1] in vocab:
            counts.pop(best)
            break
        rules.append(f"{best[0]} {best[1]}")
        vocab[best[0] + best[1]] = len(vocab)
        for w in words:
            i = 0
            while i + 1 < len(w):
                if (w[i], w[i + 1]) == best:
                    w[i:i + 2] = [best[0] + best[1]]
                else:
                    i += 1
    base = len(vocab)
    added = [{"id": base + i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": sp} for i, (c, sp) in enumerate([("<start_of_turn>", True), ("<end_of_turn>", True), ("<|tool_call>", False), ("<|tool_call>x", False)])]
    # <bos>/<eos> repeat vocabulary entries, as in Gemma.
    added += [{"id": vocab["<bos>"], "content": "<bos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}]
    tokenizer = {
        "version": "1.0", "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": space},
        "model": {"type": "BPE", "byte_fallback": True, "vocab": vocab, "merges": rules},
    }
    config = {"added_tokens_decoder": {str(a["id"]): {k: a[k] for k in
              ("content", "single_word", "lstrip", "rstrip", "normalized", "special")} for a in added}}
    generation = {"eos_token_id": 1}
    paths = []
    for name, data in (("tokenizer.json", tokenizer), ("tokenizer_config.json", config), ("generation_config.json", generation)):
        path = directory / name
        path.write_text(json.dumps(data, ensure_ascii=False))
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


SNIPPETS = [
    "Hello, world!", " It's", "they're", "I'LL", "we'd", "'", "''s", "a'b", "x", "  ", "\n", "\r\n", "\n\n", " \n ",
    "\t", "   ", "foo(bar)", "3.14", "1234", "\u0663\u0664", "caf\u00e9", "cafe\u0301", "e\u0301\u0302", "\u00e9\u0301",
    "\u4f60\u597d", "\u3053\u3093\u306b\u3061\u306f", "\U0001f600", "\U0001f1fa\U0001f1f8", "\u1100\u1161\u11a8", "\uac00",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "<tool_call_x>", "<tool", "<|im_", "<start_of_turn>",
    "<end_of_turn>", "<|tool_call>", "<|tool_call>x", "<bos>", "the", "quick", "brown fox", "def ", "return ", "print",
    "\u00a0", "\u2003", "\u2028", "\u200b", "\ufeff", "\u0301", "...", "!!!", "?!", "-", "_", "#", "{", "}", "\\", "\"",
]
ALPHABET = list("abcdefghijklmnopqrstuvwxyz ABCDEFGH0123456789.,;:!?\n\t'\u00e9\u4e2d\u0430") + ["\u0301", "\U0001f600"]


def random_text(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(0, 12)):
        if rng.random() < 0.6:
            parts.append(rng.choice(SNIPPETS))
        else:
            parts.append("".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 8))))
    return "".join(parts)


def encode_cases(rng: random.Random, count: int) -> list[str]:
    lines = []
    for _ in range(count):
        text = random_text(rng).encode()
        parse = "1" if rng.random() < 0.8 else "0"
        limit = rng.choice([99999999, 99999999, 99999999, 0, 1, 2, 5, 17])
        if rng.random() < 0.15 and text:  # exercise invalid UTF-8
            cut = rng.randrange(len(text))
            text = text[:cut] + bytes([rng.choice([0x80, 0xC0, 0xFF, 0xE2])]) + text[cut:]
        boundaries = "-"
        spans = "-"
        if rng.random() < 0.5 and text:
            offsets = sorted(rng.randint(0, len(text)) for _ in range(rng.randint(1, 4)))
            boundaries = ",".join(map(str, offsets))
        if rng.random() < 0.3 and len(text) > 2:
            cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.choice([2, 4, 6]))))
            flat = []
            for i in range(0, len(cuts) - 1, 2):
                if cuts[i] < cuts[i + 1]:
                    flat += [cuts[i], cuts[i + 1]]
            if flat:
                spans = ",".join(map(str, flat))
        if not text:
            continue
        lines.append(f"encode {text.hex()} {parse} {limit} {boundaries} {spans}")
    return lines


def decode_cases(rng: random.Random, vocab_size: int, count: int, stops: list[int]) -> list[str]:
    lines = []
    for _ in range(count):
        n = rng.randint(1, 12)
        ids = [rng.randrange(vocab_size + 20) if rng.random() < 0.97 else -1 for _ in range(n)]
        if rng.random() < 0.3:
            ids[-1] = rng.choice(stops)
        skip = "1" if rng.random() < 0.5 else "0"
        stop_arg = ",".join(map(str, stops)) if rng.random() < 0.5 else "-"
        lines.append(f"decode {','.join(map(str, ids))} {skip} {stop_arg}")
    return lines
