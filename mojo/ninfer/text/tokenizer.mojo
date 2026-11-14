"""BPE tokenizer for the Qwen3.5 family and Gemma 4; mirrors
`src/models/qwen3_5/frontend/tokenizer.{h,cpp}`.

Both flavours of the C++ class are here: the byte-level Qwen tokenizer and the SentencePiece
style one (Gemma 4). Loading checks every field the C++ loader checks and raises the same
message; encoding keeps the same pre-tokenization, merge order and boundary bookkeeping.

Text is `List[UInt8]` throughout: decoded tokens are arbitrary bytes, not always valid UTF-8.
"""

from std.collections import Dict, List, Optional

from ninfer.artifact.json import Json, parse_json, NULL, BOOL, UINT, NEG, FLOAT, STRING, ARRAY, OBJECT
from ninfer.text.unicode import (
    Unicode,
    CodepointSpan,
    append_utf8,
    utf8_codepoint_at,
    utf8_codepoints,
)

comptime MAX_TOKEN_ID = 1_000_000
comptime NO_LIMIT = Int.MAX

comptime QWEN_SPLIT_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"


def _sp_space() -> String:
    """U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's visible space."""
    return String(chr(0x2581))


def _bytes_of(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


# --- JSON accessors with the C++ loader's messages ---------------------------------------


struct _Asset(Movable):
    var doc: Json
    var root: Int
    var label: String

    def __init__(out self, var text: List[UInt8], label: String) raises:
        self.doc = Json()
        self.label = label
        try:
            self.root = parse_json(text^, label, self.doc, strict_keys=False)
        except e:
            raise Error("malformed " + label + ": " + String(e))


def _require_object_field(doc: Json, obj: Int, field: String, label: String) raises -> Int:
    if not doc.is_object(obj) or not doc.contains(obj, field):
        raise Error("missing field " + field + " in " + label)
    var v = doc.find(obj, field)
    if not doc.is_object(v):
        raise Error("field " + field + " must be object in " + label)
    return v


def _require_array_field(doc: Json, obj: Int, field: String, label: String) raises -> Int:
    if not doc.is_object(obj) or not doc.contains(obj, field):
        raise Error("missing field " + field + " in " + label)
    var v = doc.find(obj, field)
    if not doc.is_array(v):
        raise Error("field " + field + " must be array in " + label)
    return v


def _parse_token_id(doc: Json, value: Int, field: String, label: String) raises -> Int:
    if not doc.is_integer(value):
        raise Error("field " + field + " must be integer in " + label)
    var mag = doc.nodes[value].u
    if doc.is_unsigned(value):
        if mag > UInt64(MAX_TOKEN_ID):
            raise Error("field " + field + " id is out of range in " + label)
        return Int(mag)
    if mag != 0:
        raise Error("field " + field + " has negative id in " + label)
    return 0


def _require_string_field(doc: Json, obj: Int, field: String, label: String) raises -> String:
    if not doc.is_object(obj) or not doc.contains(obj, field) or not doc.is_string(doc.find(obj, field)):
        raise Error("field " + field + " must be string in " + label)
    return doc.string(doc.find(obj, field))


def _require_bool_field(doc: Json, obj: Int, field: String, label: String) raises -> Bool:
    if not doc.is_object(obj) or not doc.contains(obj, field) or doc.kind(doc.find(obj, field)) != BOOL:
        raise Error("field " + field + " must be boolean in " + label)
    return doc.nodes[doc.find(obj, field)].u != 0


def _type_is(doc: Json, v: Int, name: String) -> Bool:
    if not doc.is_object(v) or not doc.contains(v, "type"):
        return False
    var t = doc.find(v, "type")
    return doc.is_string(t) and doc.string(t) == name


def _string_member(doc: Json, v: Int, key: String) -> String:
    """`value.value(key, std::string{})` for a string member; empty otherwise."""
    if doc.is_object(v) and doc.contains(v, key) and doc.is_string(doc.find(v, key)):
        return doc.string(doc.find(v, key))
    return String()


def _bool_is(doc: Json, v: Int, key: String, expected: Bool) -> Bool:
    """`value.value(key, default) == expected` with the default being `expected`."""
    if not doc.is_object(v) or not doc.contains(v, key):
        return True
    var m = doc.find(v, key)
    return doc.kind(m) == BOOL and (doc.nodes[m].u != 0) == expected


# --- Types ---------------------------------------------------------------------------------


struct AddedToken(Copyable, Movable):
    var id: Int
    var content: String
    var content_bytes: List[UInt8]
    var single_word: Bool
    var lstrip: Bool
    var rstrip: Bool
    var normalized: Bool
    var special: Bool

    def __init__(out self):
        self.id = -1
        self.content = String()
        self.content_bytes = List[UInt8]()
        self.single_word = False
        self.lstrip = False
        self.rstrip = False
        self.normalized = False
        self.special = False


def _same_added_token(a: AddedToken, b: AddedToken) -> Bool:
    return (
        a.id == b.id
        and a.content == b.content
        and a.single_word == b.single_word
        and a.lstrip == b.lstrip
        and a.rstrip == b.rstrip
        and a.normalized == b.normalized
        and a.special == b.special
    )


struct EncodeOptions(Copyable, Movable):
    var parse_added_tokens: Bool
    var max_tokens: Int

    def __init__(out self, parse_added_tokens: Bool = True, max_tokens: Int = NO_LIMIT):
        self.parse_added_tokens = parse_added_tokens
        self.max_tokens = max_tokens


struct DecodeOptions(Copyable, Movable):
    var skip_special_tokens: Bool
    var stop_token_ids: List[Int]

    def __init__(out self, skip_special_tokens: Bool = False):
        self.skip_special_tokens = skip_special_tokens
        self.stop_token_ids = List[Int]()


struct ByteSpan(Copyable, Movable):
    var begin: Int
    var end: Int

    def __init__(out self, begin: Int, end: Int):
        self.begin = begin
        self.end = end


struct TokenBoundaryResult(Copyable, Movable):
    """`exact` is meaningful only when `has_exact`."""

    var has_exact: Bool
    var exact: Int
    var stable: Int

    def __init__(out self):
        self.has_exact = False
        self.exact = 0
        self.stable = 0

    def __init__(out self, exact: Int, stable: Int):
        self.has_exact = True
        self.exact = exact
        self.stable = stable

    @staticmethod
    def inexact(stable: Int) -> Self:
        var r = Self()
        r.stable = stable
        return r^


struct BoundaryEncodedText(Movable):
    var input_ids: List[Int]
    var boundaries: List[TokenBoundaryResult]

    def __init__(out self):
        self.input_ids = List[Int]()
        self.boundaries = List[TokenBoundaryResult]()


struct _IndexedBoundary(Copyable, Movable):
    var offset: Int
    var index: Int

    def __init__(out self, offset: Int, index: Int):
        self.offset = offset
        self.index = index


struct _WordEnd(Copyable, Movable):
    var normalized_offset: Int
    var token_frontier: Int

    def __init__(out self, normalized_offset: Int, token_frontier: Int):
        self.normalized_offset = normalized_offset
        self.token_frontier = token_frontier


def _merge_pair_key(left: Int, right: Int) -> Int:
    return (left << 32) | right


# --- BPE merge over one word ---------------------------------------------------------------


struct _Heap(Movable):
    """Min-heap of merge candidates ordered by (rank, left), as `LaterBpeCandidate` defines."""

    var rank: List[Int]
    var left: List[Int]
    var right: List[Int]
    var result: List[Int]
    var lgen: List[Int]
    var rgen: List[Int]

    def __init__(out self):
        self.rank = List[Int]()
        self.left = List[Int]()
        self.right = List[Int]()
        self.result = List[Int]()
        self.lgen = List[Int]()
        self.rgen = List[Int]()

    def empty(self) -> Bool:
        return len(self.rank) == 0

    def _less(self, a: Int, b: Int) -> Bool:
        if self.rank[a] != self.rank[b]:
            return self.rank[a] < self.rank[b]
        return self.left[a] < self.left[b]

    def _swap(mut self, a: Int, b: Int):
        var t = self.rank[a]
        self.rank[a] = self.rank[b]
        self.rank[b] = t
        t = self.left[a]
        self.left[a] = self.left[b]
        self.left[b] = t
        t = self.right[a]
        self.right[a] = self.right[b]
        self.right[b] = t
        t = self.result[a]
        self.result[a] = self.result[b]
        self.result[b] = t
        t = self.lgen[a]
        self.lgen[a] = self.lgen[b]
        self.lgen[b] = t
        t = self.rgen[a]
        self.rgen[a] = self.rgen[b]
        self.rgen[b] = t

    def push(mut self, rank: Int, left: Int, right: Int, result: Int, lgen: Int, rgen: Int):
        self.rank.append(rank)
        self.left.append(left)
        self.right.append(right)
        self.result.append(result)
        self.lgen.append(lgen)
        self.rgen.append(rgen)
        var i = len(self.rank) - 1
        while i > 0:
            var parent = (i - 1) // 2
            if not self._less(i, parent):
                break
            self._swap(i, parent)
            i = parent

    def pop_into(mut self, mut out: List[Int]):
        """Remove the minimum; `out` receives [rank, left, right, result, lgen, rgen]."""
        out.clear()
        out.append(self.rank[0])
        out.append(self.left[0])
        out.append(self.right[0])
        out.append(self.result[0])
        out.append(self.lgen[0])
        out.append(self.rgen[0])
        var last = len(self.rank) - 1
        self._swap(0, last)
        _ = self.rank.pop()
        _ = self.left.pop()
        _ = self.right.pop()
        _ = self.result.pop()
        _ = self.lgen.pop()
        _ = self.rgen.pop()
        var n = len(self.rank)
        var i = 0
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var m = i
            if l < n and self._less(l, m):
                m = l
            if r < n and self._less(r, m):
                m = r
            if m == i:
                break
            self._swap(i, m)
            i = m


def _bpe_merge(
    mut symbols: List[Int],
    mut ends: List[Int],
    merges: Dict[Int, Int],
    mut out_symbols: List[Int],
    mut out_ends: List[Int],
):
    """Run BPE over `symbols` (with per-symbol `ends`) and append the surviving nodes."""
    var n = len(symbols)
    if n == 0:
        return
    var previous = List[Int]()
    var next = List[Int]()
    var gen = List[Int]()
    var live = List[Bool]()
    for i in range(n):
        previous.append(i - 1)
        next.append(i + 1 if i + 1 < n else -1)
        gen.append(0)
        live.append(True)
    var heap = _Heap()
    # push_candidate(left)
    for left in range(n - 1):
        var key = _merge_pair_key(symbols[left], symbols[left + 1])
        var rule = merges.get(key)
        if rule:
            heap.push(rule.value() >> 32, left, left + 1, rule.value() & 0xFFFFFFFF, gen[left], gen[left + 1])
    var cand = List[Int]()
    while not heap.empty():
        heap.pop_into(cand)
        var l = cand[1]
        var r = cand[2]
        if not live[l] or not live[r] or next[l] != r or gen[l] != cand[4] or gen[r] != cand[5]:
            continue
        symbols[l] = cand[3]
        ends[l] = ends[r]
        gen[l] += 1
        next[l] = next[r]
        live[r] = False
        gen[r] += 1
        if next[l] >= 0:
            previous[next[l]] = l
        # push_candidate(previous[l]) then push_candidate(l)
        for pass_ in range(2):
            var a = previous[l] if pass_ == 0 else l
            if a < 0 or not live[a]:
                continue
            var b = next[a]
            if b < 0:
                continue
            var rule = merges.get(_merge_pair_key(symbols[a], symbols[b]))
            if rule:
                heap.push(rule.value() >> 32, a, b, rule.value() & 0xFFFFFFFF, gen[a], gen[b])
    var node = 0
    while node >= 0:
        out_symbols.append(symbols[node])
        out_ends.append(ends[node])
        node = next[node]


# --- Tokenizer -----------------------------------------------------------------------------


struct Tokenizer(Movable):
    var unicode: Unicode
    var sentencepiece: Bool
    var decoded_token_bytes: List[List[UInt8]]
    var valid_token_ids: List[Bool]
    var special_token_ids: List[Bool]
    var vocab_token_to_id: Dict[String, Int]
    var merges: Dict[Int, Int]  # pair key -> (rank << 32) | result
    var byte_token_ids: List[Int]
    var added_tokens: List[AddedToken]
    var added_token_candidates: List[List[Int]]  # by first byte; indexes into added_tokens
    var default_stop_token_ids: List[Int]

    def __init__(
        out self,
        var tokenizer_json: List[UInt8],
        var tokenizer_config_json: List[UInt8],
        var generation_config_json: List[UInt8],
    ) raises:
        if (
            len(tokenizer_json) == 0
            or len(tokenizer_config_json) == 0
            or len(generation_config_json) == 0
        ):
            raise Error("embedded tokenizer resources are empty")
        self.unicode = Unicode()
        self.sentencepiece = False
        self.decoded_token_bytes = List[List[UInt8]]()
        self.valid_token_ids = List[Bool]()
        self.special_token_ids = List[Bool]()
        self.vocab_token_to_id = Dict[String, Int]()
        self.merges = Dict[Int, Int]()
        self.byte_token_ids = List[Int]()
        self.added_tokens = List[AddedToken]()
        self.added_token_candidates = List[List[Int]]()
        for _ in range(256):
            self.added_token_candidates.append(List[Int]())
        self.default_stop_token_ids = List[Int]()

        var tok = _Asset(tokenizer_json^, "tokenizer.json")
        var cfg = _Asset(tokenizer_config_json^, "tokenizer_config.json")
        var model = _require_object_field(tok.doc, tok.root, "model", "tokenizer.json")
        var generation = generation_config_json.copy()
        if _is_sentencepiece(tok.doc, tok.root):
            self._load_sentencepiece(tok, cfg, model, generation^)
            return
        _validate_pipeline(tok.doc, tok.root, model)

        # Vocabulary.
        var ids = _load_vocab(tok.doc, model, "tokenizer.json")
        var id_to_token = ids.id_to_token.copy()
        self.vocab_token_to_id = ids.token_to_id.copy()
        var occupied = ids.occupied.copy()  # id -> True
        self.valid_token_ids = List[Bool]()
        for _ in range(len(id_to_token)):
            self.valid_token_ids.append(False)
        for id in occupied.keys():
            self.valid_token_ids[id] = True
        self.added_tokens = _load_added_tokens(
            tok.doc, tok.root, "tokenizer.json", id_to_token, occupied, self.vocab_token_to_id
        )
        _merge_added_tokens_decoder(
            cfg.doc, cfg.root, "tokenizer_config.json", id_to_token, occupied,
            self.vocab_token_to_id, self.added_tokens,
        )
        for index in range(len(self.added_tokens)):
            var content = self.added_tokens[index].content_bytes.copy()
            if len(content) > 0:
                self.added_token_candidates[Int(content[0])].append(index)
        while len(self.valid_token_ids) < len(id_to_token):
            self.valid_token_ids.append(False)
        var added_ids = List[Bool]()
        for _ in range(len(id_to_token)):
            added_ids.append(False)
            self.special_token_ids.append(False)
        for token in self.added_tokens:
            self.valid_token_ids[token.id] = True
            added_ids[token.id] = True
            self.special_token_ids[token.id] = token.special
        var decoder = _byte_level_decoder()
        for index in range(len(id_to_token)):
            var text = id_to_token[index]
            if self.valid_token_ids[index] and not added_ids[index]:
                self.decoded_token_bytes.append(
                    _decode_byte_level_token(self.unicode, decoder, text, index)
                )
            else:
                self.decoded_token_bytes.append(_bytes_of(text))
        self.merges = _load_bpe_merge_rules(tok.doc, model, "tokenizer.json", self.vocab_token_to_id, False)
        self.byte_token_ids = _load_byte_token_ids(self.vocab_token_to_id)
        self.default_stop_token_ids = _load_default_stop_token_ids(generation^)

    def _load_sentencepiece(
        mut self, tok: _Asset, cfg: _Asset, model: Int, var generation: List[UInt8]
    ) raises:
        self.sentencepiece = True
        var bf = tok.doc.find(model, "byte_fallback")
        if bf < 0 or tok.doc.kind(bf) != BOOL or tok.doc.nodes[bf].u == 0:
            raise Error("SentencePiece tokenizer.json must use byte fallback")
        var vocab = _load_vocab(tok.doc, model, "tokenizer.json")
        for id in range(len(vocab.id_to_token)):
            self.decoded_token_bytes.append(_decode_sentencepiece_token(vocab.id_to_token[id]))
        self.vocab_token_to_id = vocab.token_to_id.copy()
        for _ in range(len(self.decoded_token_bytes)):
            self.valid_token_ids.append(False)
        for id in vocab.occupied.keys():
            self.valid_token_ids[id] = True
        # Added tokens may repeat a vocabulary entry; their content is matched in the input.
        if tok.doc.contains(tok.root, "added_tokens") and tok.doc.is_array(tok.doc.find(tok.root, "added_tokens")):
            var arr = tok.doc.find(tok.root, "added_tokens")
            for i in range(tok.doc.size(arr)):
                self._add_sp_token(_parse_added_token(tok.doc, tok.doc.at(arr, i), "tokenizer.json"))
        if cfg.doc.contains(cfg.root, "added_tokens_decoder") and cfg.doc.is_object(cfg.doc.find(cfg.root, "added_tokens_decoder")):
            var dec = cfg.doc.find(cfg.root, "added_tokens_decoder")
            for i in range(cfg.doc.size(dec)):
                var id = _parse_added_token_decoder_id(cfg.doc.key(dec, i), "tokenizer_config.json")
                self._add_sp_token(
                    _parse_added_token_decoder_entry(cfg.doc, id, cfg.doc.at(dec, i), "tokenizer_config.json")
                )
        _sort_added_by_id(self.added_tokens)
        # Longest content first among candidates sharing a first byte.
        var order = List[Int]()
        for i in range(len(self.added_tokens)):
            order.append(i)
        # Stable insertion sort by descending content length.
        for i in range(1, len(order)):
            var j = i
            while j > 0 and len(self.added_tokens[order[j]].content_bytes) > len(self.added_tokens[order[j - 1]].content_bytes):
                var t = order[j]
                order[j] = order[j - 1]
                order[j - 1] = t
                j -= 1
        for index in order:
            var content = self.added_tokens[index].content_bytes.copy()
            if len(content) > 0:
                self.added_token_candidates[Int(content[0])].append(index)
        for _ in range(len(self.decoded_token_bytes)):
            self.special_token_ids.append(False)
        for token in self.added_tokens:
            self.special_token_ids[token.id] = token.special
        self.merges = _load_bpe_merge_rules(tok.doc, model, "tokenizer.json", self.vocab_token_to_id, True)
        self.byte_token_ids = _load_byte_fallback_ids(self.vocab_token_to_id)
        self.default_stop_token_ids = _load_default_stop_token_ids(generation^)

    def _add_sp_token(mut self, var token: AddedToken) raises:
        var index = token.id
        while index >= len(self.decoded_token_bytes):
            self.decoded_token_bytes.append(List[UInt8]())
            self.valid_token_ids.append(False)
        for existing in self.added_tokens:
            if existing.id == token.id:
                return
        self.decoded_token_bytes[index] = token.content_bytes.copy()
        self.valid_token_ids[index] = True
        self.added_tokens.append(token^)

    # --- queries -----------------------------------------------------------------------------

    def vocab_size(self) -> Int:
        return len(self.decoded_token_bytes)

    def is_special_token(self, id: Int) -> Bool:
        return id >= 0 and id < len(self.special_token_ids) and self.special_token_ids[id]

    def is_valid_token(self, id: Int) -> Bool:
        return id >= 0 and id < len(self.valid_token_ids) and self.valid_token_ids[id]

    def has_exact_token_domain(self, size: Int) -> Bool:
        if len(self.valid_token_ids) != size:
            return False
        for v in self.valid_token_ids:
            if not v:
                return False
        return True

    # --- decode ------------------------------------------------------------------------------

    def _check_token(self, id: Int) raises:
        if id < 0:
            raise Error("Tokenizer::decode received negative token id " + String(id))
        if id >= len(self.decoded_token_bytes) or id >= len(self.valid_token_ids) or not self.valid_token_ids[id]:
            raise Error("Tokenizer::decode token id " + String(id) + " is outside loaded vocabulary")

    def decode_token_bytes(self, id: Int, skip_special_tokens: Bool = False) raises -> List[UInt8]:
        self._check_token(id)
        if skip_special_tokens and self.special_token_ids[id]:
            return List[UInt8]()
        return self.decoded_token_bytes[id].copy()

    def decode(self, ids: List[Int], options: DecodeOptions) raises -> List[UInt8]:
        var text = List[UInt8]()
        var terminal = len(ids)
        if len(ids) > 0:
            for s in options.stop_token_ids:
                if s == ids[len(ids) - 1]:
                    terminal = len(ids) - 1
        for i in range(len(ids)):
            if i == terminal:
                continue
            text.extend(self.decode_token_bytes(ids[i], options.skip_special_tokens))
        _ = utf8_codepoints(text, "Tokenizer::decode reconstructed output")
        return text^

    # --- encode ------------------------------------------------------------------------------

    def encode(self, text: List[UInt8], options: EncodeOptions = EncodeOptions()) raises -> List[Int]:
        var no_boundaries = List[Int]()
        var no_spans = List[ByteSpan]()
        var r = self.encode_with_boundaries(text, no_boundaries, options, no_spans)
        return r.input_ids.copy()

    def _qwen_word_end(self, text: List[UInt8], begin: Int) raises -> Int:
        var first = utf8_codepoint_at(text, begin, "Tokenizer::encode input")
        var after_first = begin + first.length
        var cp = first.value
        if cp == 0x27:  # '
            var suffixes = List[String]()
            suffixes.append("s")
            suffixes.append("t")
            suffixes.append("re")
            suffixes.append("ve")
            suffixes.append("m")
            suffixes.append("ll")
            suffixes.append("d")
            for suffix in suffixes:
                if _ascii_ci_matches(text, after_first, suffix):
                    return after_first + suffix.byte_length()
        var has_next = after_first < len(text)
        var next = utf8_codepoint_at(text, after_first, "Tokenizer::encode input").value if has_next else 0
        if self._letter_or_mark(cp) or (
            self._non_newline_non_letter_non_number(cp) and has_next and self._letter_or_mark(next)
        ):
            var end = begin if self._letter_or_mark(cp) else after_first
            while end < len(text):
                var v = utf8_codepoint_at(text, end, "Tokenizer::encode input")
                if not self._letter_or_mark(v.value):
                    break
                end += v.length
            return end
        if self.unicode.is_number(cp):
            return after_first
        if (cp == 0x20 and has_next and self._non_space_non_letter_mark_number(next)) or self._non_space_non_letter_mark_number(cp):
            var end = after_first if cp == 0x20 else begin
            while end < len(text):
                var v = utf8_codepoint_at(text, end, "Tokenizer::encode input")
                if not self._non_space_non_letter_mark_number(v.value):
                    break
                end += v.length
            while end < len(text):
                var v = utf8_codepoint_at(text, end, "Tokenizer::encode input")
                if not _is_newline(v.value):
                    break
                end += v.length
            return end
        if self.unicode.is_whitespace(cp):
            var end = begin
            var last_begin = begin
            var last_newline_end = -1
            var count = 0
            while end < len(text):
                var v = utf8_codepoint_at(text, end, "Tokenizer::encode input")
                if not self.unicode.is_whitespace(v.value):
                    break
                last_begin = end
                end += v.length
                count += 1
                if _is_newline(v.value):
                    last_newline_end = end
            if last_newline_end >= 0:
                return last_newline_end
            if end == len(text):
                return end
            return last_begin if count >= 2 else end
        return after_first

    def _letter_or_mark(self, cp: Int) -> Bool:
        return self.unicode.is_letter(cp) or self.unicode.is_mark(cp)

    def _non_newline_non_letter_non_number(self, cp: Int) -> Bool:
        return not _is_newline(cp) and not self.unicode.is_letter(cp) and not self.unicode.is_number(cp)

    def _non_space_non_letter_mark_number(self, cp: Int) -> Bool:
        return (
            not self.unicode.is_whitespace(cp)
            and not self.unicode.is_letter(cp)
            and not self.unicode.is_mark(cp)
            and not self.unicode.is_number(cp)
        )

    def _append_normalized_bpe_ids(
        self,
        mut ids: List[Int],
        normalized: List[UInt8],
        max_tokens: Int,
        mut token_ends: List[Int],
        mut word_ends: List[_WordEnd],
        record: Bool,
    ) raises -> Bool:
        if len(normalized) == 0:
            return True
        if len(ids) == max_tokens:
            return False
        var begin = 0
        while begin < len(normalized):
            var end = self._qwen_word_end(normalized, begin)
            var symbols = List[Int]()
            var ends = List[Int]()
            for index in range(begin, end):
                var symbol = self.byte_token_ids[Int(normalized[index])]
                if symbol < 0:
                    raise Error("Tokenizer::encode produced byte symbol outside vocabulary")
                symbols.append(symbol)
                ends.append(index - begin + 1)
            var out_symbols = List[Int]()
            var out_ends = List[Int]()
            _bpe_merge(symbols, ends, self.merges, out_symbols, out_ends)
            for k in range(len(out_symbols)):
                ids.append(out_symbols[k])
                if record:
                    token_ends.append(begin + out_ends[k])
                if len(ids) == max_tokens:
                    return False
            if record:
                word_ends.append(_WordEnd(end, len(ids)))
            begin = end
        return True

    def _append_sentencepiece_ids(
        self, mut ids: List[Int], text: List[UInt8], max_tokens: Int, mut ends_out: List[Int]
    ) raises -> Bool:
        if len(text) == 0:
            return True
        if len(ids) == max_tokens:
            return False
        var symbols = List[Int]()
        var ends = List[Int]()
        var begin = 0
        while begin < len(text):
            var cp = utf8_codepoint_at(text, begin, "Tokenizer::encode input")
            var end = begin + cp.length
            var piece = List[UInt8]()
            if cp.value == 0x20:
                piece = _bytes_of(_sp_space())
            else:
                for b in range(begin, end):
                    piece.append(text[b])
            var found = self.vocab_token_to_id.get(String(unsafe_from_utf8=piece))
            if found:
                symbols.append(found.value())
                ends.append(end)
            else:
                for b in range(begin, end):
                    var symbol = self.byte_token_ids[Int(text[b])]
                    if symbol < 0:
                        raise Error("Tokenizer::encode found no byte-fallback token")
                    symbols.append(symbol)
                    ends.append(b + 1)
            begin = end
        var out_symbols = List[Int]()
        var out_ends = List[Int]()
        _bpe_merge(symbols, ends, self.merges, out_symbols, out_ends)
        for k in range(len(out_symbols)):
            ids.append(out_symbols[k])
            ends_out.append(out_ends[k])
            if len(ids) == max_tokens:
                return False
        return True

    def _append_sentencepiece_text(
        self,
        mut encoded: BoundaryEncodedText,
        text: List[UInt8],
        text_offset: Int,
        boundaries: List[_IndexedBoundary],
        max_tokens: Int,
    ) raises -> Bool:
        var token_base = len(encoded.input_ids)
        var ends = List[Int]()
        var complete = self._append_sentencepiece_ids(encoded.input_ids, text, max_tokens, ends)
        for boundary in boundaries:
            var local = boundary.offset - text_offset
            if local == 0:
                encoded.boundaries[boundary.index] = TokenBoundaryResult(token_base, token_base)
                continue
            var at = -1
            for k in range(len(ends)):
                if ends[k] == local:
                    at = k
                    break
            if at >= 0:
                var frontier = token_base + at + 1
                encoded.boundaries[boundary.index] = TokenBoundaryResult(frontier, frontier)
            else:
                encoded.boundaries[boundary.index] = TokenBoundaryResult.inexact(token_base)
        return complete

    def _append_ordinary_text(
        self,
        mut encoded: BoundaryEncodedText,
        text: List[UInt8],
        text_offset: Int,
        boundaries: List[_IndexedBoundary],
        max_tokens: Int,
    ) raises -> Bool:
        var token_base = len(encoded.input_ids)
        if len(text) == 0:
            for boundary in boundaries:
                encoded.boundaries[boundary.index] = TokenBoundaryResult(token_base, token_base)
            return True
        var normalized = self.unicode.normalize_nfc(text)
        var has_internal = False
        for boundary in boundaries:
            var local = boundary.offset - text_offset
            if local != 0 and local != len(text):
                has_internal = True
        var token_ends = List[Int]()
        var word_ends = List[_WordEnd]()
        if not self._append_normalized_bpe_ids(
            encoded.input_ids, normalized, max_tokens, token_ends, word_ends, has_internal
        ):
            return False
        var token_end = len(encoded.input_ids)
        if not has_internal:
            for boundary in boundaries:
                var frontier = token_base if boundary.offset == text_offset else token_end
                encoded.boundaries[boundary.index] = TokenBoundaryResult(frontier, frontier)
            return True

        var normalized_boundaries = List[Int]()
        var independently = List[UInt8]()
        var raw_begin = 0
        for request in range(len(boundaries)):
            var raw_end = boundaries[request].offset - text_offset
            if raw_end != raw_begin:
                independently.extend(self.unicode.normalize_nfc(_slice(text, raw_begin, raw_end)))
                raw_begin = raw_end
            normalized_boundaries.append(len(independently))
        if raw_begin != len(text):
            independently.extend(self.unicode.normalize_nfc(_slice(text, raw_begin, len(text))))

        if not _equal_bytes(independently, normalized):
            for boundary in boundaries:
                var local = boundary.offset - text_offset
                if local == 0 or local == len(text):
                    var frontier = token_base if local == 0 else token_end
                    encoded.boundaries[boundary.index] = TokenBoundaryResult(frontier, frontier)
                else:
                    encoded.boundaries[boundary.index] = TokenBoundaryResult.inexact(token_base)
            return True

        var token_cursor = 0
        var word_cursor = 0
        var stable_frontier = token_base
        for request in range(len(boundaries)):
            var nb = normalized_boundaries[request]
            while token_cursor < len(token_ends) and token_ends[token_cursor] <= nb:
                token_cursor += 1
            while word_cursor < len(word_ends) and word_ends[word_cursor].normalized_offset <= nb:
                stable_frontier = word_ends[word_cursor].token_frontier
                word_cursor += 1
            var exact = nb == 0 or (token_cursor != 0 and token_ends[token_cursor - 1] == nb)
            var frontier = token_base + token_cursor
            if exact:
                encoded.boundaries[boundaries[request].index] = TokenBoundaryResult(frontier, stable_frontier)
            else:
                encoded.boundaries[boundaries[request].index] = TokenBoundaryResult.inexact(stable_frontier)
        return True

    def encode_with_boundaries(
        self,
        text: List[UInt8],
        byte_boundaries: List[Int],
        options: EncodeOptions,
        literal_spans: List[ByteSpan],
    ) raises -> BoundaryEncodedText:
        var previous_end = 0
        for span in literal_spans:
            if span.begin < previous_end or span.begin >= span.end or span.end > len(text):
                raise Error("Tokenizer literal spans must be ordered disjoint input ranges")
            previous_end = span.end
        var encoded = BoundaryEncodedText()
        for _ in range(len(byte_boundaries)):
            encoded.boundaries.append(TokenBoundaryResult())
        var boundaries = List[_IndexedBoundary]()
        for index in range(len(byte_boundaries)):
            if byte_boundaries[index] > len(text):
                raise Error("Tokenizer byte boundary exceeds input text")
            boundaries.append(_IndexedBoundary(byte_boundaries[index], index))
        # Stable insertion sort by offset.
        for i in range(1, len(boundaries)):
            var j = i
            while j > 0 and boundaries[j].offset < boundaries[j - 1].offset:
                var t = boundaries[j].copy()
                boundaries[j] = boundaries[j - 1].copy()
                boundaries[j - 1] = t^
                j -= 1
        if options.max_tokens == 0:
            return encoded^

        var cursor = 0
        if not options.parse_added_tokens:
            _ = self._append_ordinary(encoded, text, 0, len(text), boundaries, cursor, options.max_tokens)
            return encoded^

        var ordinary_begin = 0
        var pos = 0
        while pos < len(text):
            var match_index = -1
            ref candidates = self.added_token_candidates[Int(text[pos])]
            for index in candidates:
                ref token = self.added_tokens[index]
                var n = len(token.content_bytes)
                if n <= len(text) - pos and _bytes_equal_at(text, pos, token.content_bytes) and not _overlaps(literal_spans, pos, pos + n):
                    match_index = index
                    break
            if match_index < 0:
                pos += 1
                continue
            if not self._append_ordinary(encoded, text, ordinary_begin, pos, boundaries, cursor, options.max_tokens):
                return encoded^
            var token_end = pos + len(self.added_tokens[match_index].content_bytes)
            var stable = len(encoded.input_ids)
            while cursor < len(boundaries) and boundaries[cursor].offset < token_end:
                encoded.boundaries[boundaries[cursor].index] = TokenBoundaryResult.inexact(stable)
                cursor += 1
            encoded.input_ids.append(self.added_tokens[match_index].id)
            if len(encoded.input_ids) == options.max_tokens:
                return encoded^
            while cursor < len(boundaries) and boundaries[cursor].offset == token_end:
                var n = len(encoded.input_ids)
                encoded.boundaries[boundaries[cursor].index] = TokenBoundaryResult(n, n)
                cursor += 1
            pos = token_end
            ordinary_begin = pos
        _ = self._append_ordinary(encoded, text, ordinary_begin, len(text), boundaries, cursor, options.max_tokens)
        return encoded^

    def _append_ordinary(
        self,
        mut encoded: BoundaryEncodedText,
        text: List[UInt8],
        begin: Int,
        end: Int,
        boundaries: List[_IndexedBoundary],
        mut cursor: Int,
        max_tokens: Int,
    ) raises -> Bool:
        var request_end = cursor
        while request_end < len(boundaries) and boundaries[request_end].offset <= end:
            request_end += 1
        var requests = List[_IndexedBoundary]()
        for i in range(cursor, request_end):
            requests.append(boundaries[i].copy())
        var piece = _slice(text, begin, end)
        var complete: Bool
        if self.sentencepiece:
            complete = self._append_sentencepiece_text(encoded, piece, begin, requests, max_tokens)
        else:
            complete = self._append_ordinary_text(encoded, piece, begin, requests, max_tokens)
        cursor = request_end
        return complete


# --- helpers -------------------------------------------------------------------------------


def _slice(text: List[UInt8], begin: Int, end: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=end - begin)
    for i in range(begin, end):
        out.append(text[i])
    return out^


def _equal_bytes(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _bytes_equal_at(text: List[UInt8], pos: Int, needle: List[UInt8]) -> Bool:
    for i in range(len(needle)):
        if text[pos + i] != needle[i]:
            return False
    return True


def _is_newline(cp: Int) -> Bool:
    return cp == 0x0D or cp == 0x0A


def _ascii_ci_matches(text: List[UInt8], offset: Int, suffix: String) -> Bool:
    var s = suffix.as_bytes()
    if offset + len(s) > len(text):
        return False
    for i in range(len(s)):
        var a = Int(text[offset + i])
        var b = Int(s[i])
        if a >= 0x41 and a <= 0x5A:
            a += 32
        if b >= 0x41 and b <= 0x5A:
            b += 32
        if a != b:
            return False
    return True


def _overlaps(spans: List[ByteSpan], begin: Int, end: Int) -> Bool:
    """`text::overlaps`: lower_bound for the first span whose end is past `begin`."""
    if begin >= end:
        return False
    var lo = 0
    var hi = len(spans)
    while lo < hi:
        var mid = (lo + hi) // 2
        if spans[mid].end <= begin:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(spans) and spans[lo].begin < end


# --- loading helpers ------------------------------------------------------------------------


struct _Vocab(Movable):
    var id_to_token: List[String]
    var token_to_id: Dict[String, Int]
    var occupied: Dict[Int, Bool]

    def __init__(out self):
        self.id_to_token = List[String]()
        self.token_to_id = Dict[String, Int]()
        self.occupied = Dict[Int, Bool]()


def _is_sentencepiece(doc: Json, root: Int) -> Bool:
    if not doc.contains(root, "normalizer") or not doc.is_object(doc.find(root, "normalizer")):
        return False
    var n = doc.find(root, "normalizer")
    var pattern = doc.find(n, "pattern")
    return (
        _string_member(doc, n, "type") == "Replace"
        and pattern >= 0
        and doc.is_object(pattern)
        and _string_member(doc, pattern, "String") == " "
        and _string_member(doc, n, "content") == _sp_space()
    )


def require(valid: Bool, field: String) raises:
    if not valid:
        raise Error("tokenizer.json " + field + " is not supported by the Qwen tokenizer")


def _validate_pipeline(doc: Json, root: Int, model: Int) raises:
    if doc.contains(root, "normalizer"):
        require(_type_is(doc, doc.find(root, "normalizer"), "NFC"), "normalizer")
    if doc.contains(root, "pre_tokenizer"):
        var pre = doc.find(root, "pre_tokenizer")
        var list = doc.find(pre, "pretokenizers")
        require(
            _type_is(doc, pre, "Sequence")
            and list >= 0
            and doc.is_array(list)
            and doc.size(list) == 2,
            "pre_tokenizer",
        )
        var split = doc.at(list, 0)
        var bytes = doc.at(list, 1)
        var pattern = doc.find(split, "pattern")
        require(
            _type_is(doc, split, "Split")
            and pattern >= 0
            and doc.is_object(pattern)
            and _string_member(doc, pattern, "Regex") == QWEN_SPLIT_PATTERN
            and _string_member(doc, split, "behavior") == "Isolated"
            and _bool_is(doc, split, "invert", False),
            "pre_tokenizer.Split",
        )
        require(
            _type_is(doc, bytes, "ByteLevel")
            and _bool_is(doc, bytes, "add_prefix_space", False)
            and _bool_is(doc, bytes, "use_regex", False),
            "pre_tokenizer.ByteLevel",
        )
    if doc.contains(root, "decoder"):
        require(_type_is(doc, doc.find(root, "decoder"), "ByteLevel"), "decoder")
    if doc.contains(root, "post_processor") and not doc.is_null(doc.find(root, "post_processor")):
        require(_type_is(doc, doc.find(root, "post_processor"), "ByteLevel"), "post_processor")
    if doc.contains(model, "dropout") and not doc.is_null(doc.find(model, "dropout")):
        var d = doc.find(model, "dropout")
        var zero = (doc.is_integer(d) and doc.nodes[d].u == 0) or (doc.kind(d) == FLOAT and doc.nodes[d].f == 0.0)
        require(zero, "model.dropout")
    for field in ["continuing_subword_prefix", "end_of_word_suffix"]:
        if doc.contains(model, field):
            var m = doc.find(model, field)
            require(doc.is_null(m) or (doc.is_string(m) and doc.string(m) == ""), field)
    require(_bool_is(doc, model, "ignore_merges", False), "model.ignore_merges")


def _load_vocab(doc: Json, model: Int, label: String) raises -> _Vocab:
    if not doc.contains(model, "type") or not doc.is_string(doc.find(model, "type")) or doc.string(doc.find(model, "type")) != "BPE":
        raise Error("field model.type must be BPE in " + label)
    var vocab = _require_object_field(doc, model, "vocab", label)
    if doc.size(vocab) == 0:
        raise Error("field model.vocab must not be empty in " + label)
    var out = _Vocab()
    var max_id = -1
    for i in range(doc.size(vocab)):
        var id = _parse_token_id(doc, doc.at(vocab, i), "model.vocab", label)
        if id in out.occupied:
            raise Error("field model.vocab has duplicate id in " + label)
        out.occupied[id] = True
        if id > max_id:
            max_id = id
    for _ in range(max_id + 1):
        out.id_to_token.append(String())
    for i in range(doc.size(vocab)):
        var id = _parse_token_id(doc, doc.at(vocab, i), "model.vocab", label)
        var key = doc.key(vocab, i)
        out.id_to_token[id] = key
        if key not in out.token_to_id:
            out.token_to_id[key] = id
    return out^


def _parse_added_token(doc: Json, item: Int, label: String) raises -> AddedToken:
    if not doc.is_object(item):
        raise Error("field added_tokens item must be object in " + label)
    var token = AddedToken()
    if not doc.contains(item, "id"):
        raise Error("missing field added_tokens.id in " + label)
    token.id = _parse_token_id(doc, doc.find(item, "id"), "added_tokens.id", label)
    token.content = _require_string_field(doc, item, "content", label)
    token.single_word = _require_bool_field(doc, item, "single_word", label)
    token.lstrip = _require_bool_field(doc, item, "lstrip", label)
    token.rstrip = _require_bool_field(doc, item, "rstrip", label)
    token.normalized = _require_bool_field(doc, item, "normalized", label)
    token.special = _require_bool_field(doc, item, "special", label)
    token.content_bytes = _bytes_of(token.content)
    return token^


def _parse_added_token_decoder_entry(
    doc: Json, id: Int, item: Int, label: String
) raises -> AddedToken:
    if not doc.is_object(item):
        raise Error("field added_tokens_decoder item must be object in " + label)
    var token = AddedToken()
    token.id = id
    token.content = _require_string_field(doc, item, "content", label)
    token.single_word = _require_bool_field(doc, item, "single_word", label)
    token.lstrip = _require_bool_field(doc, item, "lstrip", label)
    token.rstrip = _require_bool_field(doc, item, "rstrip", label)
    token.normalized = _require_bool_field(doc, item, "normalized", label)
    token.special = _require_bool_field(doc, item, "special", label)
    token.content_bytes = _bytes_of(token.content)
    return token^


def _validate_supported_added_token(token: AddedToken, label: String) raises:
    if len(token.content_bytes) == 0:
        raise Error("added token content must not be empty in " + label)
    if token.single_word or token.lstrip or token.rstrip or token.normalized:
        raise Error(
            "Tokenizer only supports added tokens with single_word=false, "
            "lstrip=false, rstrip=false, and normalized=false in " + label
        )


def _parse_added_token_decoder_id(key: String, label: String) raises -> Int:
    var bytes = key.as_bytes()
    var bad = "added_tokens_decoder key must be a nonnegative token id in " + label
    if len(bytes) == 0 or len(bytes) > 7:
        raise Error(bad)
    var value = 0
    for b in bytes:
        if b < 0x30 or b > 0x39:
            raise Error(bad)
        value = value * 10 + Int(b - 0x30)
    if len(bytes) > 1 and bytes[0] == 0x30:  # canonical decimal only
        raise Error(bad)
    if value > MAX_TOKEN_ID:
        raise Error(bad)
    return value


def _sort_added_by_id(mut tokens: List[AddedToken]):
    for i in range(1, len(tokens)):
        var j = i
        while j > 0 and tokens[j].id < tokens[j - 1].id:
            var t = tokens[j].copy()
            tokens[j] = tokens[j - 1].copy()
            tokens[j - 1] = t^
            j -= 1


def _load_added_tokens(
    doc: Json,
    root: Int,
    label: String,
    mut id_to_token: List[String],
    occupied: Dict[Int, Bool],
    occupied_tokens: Dict[String, Int],
) raises -> List[AddedToken]:
    var added = _require_array_field(doc, root, "added_tokens", label)
    var tokens = List[AddedToken]()
    var seen_ids = Dict[Int, Bool]()
    var seen_contents = Dict[String, Int]()
    for i in range(doc.size(added)):
        var token = _parse_added_token(doc, doc.at(added, i), label)
        _validate_supported_added_token(token, label)
        if token.id in occupied:
            raise Error("field added_tokens overlaps existing id in " + label)
        if token.id in seen_ids:
            raise Error("field added_tokens has duplicate id in " + label)
        seen_ids[token.id] = True
        if token.content in occupied_tokens or token.content in seen_contents:
            raise Error("field added_tokens has duplicate content mapping in " + label)
        seen_contents[token.content] = token.id
        while token.id >= len(id_to_token):
            id_to_token.append(String())
        id_to_token[token.id] = token.content
        tokens.append(token^)
    return tokens^


def _merge_added_tokens_decoder(
    doc: Json,
    root: Int,
    label: String,
    mut id_to_token: List[String],
    occupied: Dict[Int, Bool],
    occupied_tokens: Dict[String, Int],
    mut tokens: List[AddedToken],
) raises:
    var decoder = _require_object_field(doc, root, "added_tokens_decoder", label)
    var by_id = Dict[Int, Int]()
    var by_content = Dict[String, Int]()
    for i in range(len(tokens)):
        if tokens[i].id not in by_id:
            by_id[tokens[i].id] = i
        if tokens[i].content not in by_content:
            by_content[tokens[i].content] = tokens[i].id
    var decoder_ids = Dict[Int, Bool]()
    for i in range(doc.size(decoder)):
        var id = _parse_added_token_decoder_id(doc.key(decoder, i), label)
        if id in decoder_ids:
            raise Error("added_tokens_decoder has duplicate id mapping in " + label)
        decoder_ids[id] = True
        var token = _parse_added_token_decoder_entry(doc, id, doc.at(decoder, i), label)
        _validate_supported_added_token(token, label)
        var existing = by_id.get(id)
        if existing:
            if not _same_added_token(tokens[existing.value()], token):
                raise Error(
                    "conflicting added-token definition for id " + String(id)
                    + " between tokenizer.json and tokenizer_config.json"
                )
            continue
        if id in occupied:
            raise Error("added_tokens_decoder overlaps vocabulary id " + String(id))
        if token.content in by_content or token.content in occupied_tokens:
            raise Error("conflicting added-token content mapping for " + token.content)
        while id >= len(id_to_token):
            id_to_token.append(String())
        if id_to_token[id].byte_length() != 0:
            raise Error("duplicate tokenizer mapping for id " + String(id))
        id_to_token[id] = token.content
        by_id[id] = len(tokens)
        by_content[token.content] = id
        tokens.append(token^)
    _sort_added_by_id(tokens)


def _load_default_stop_token_ids(var contents: List[UInt8]) raises -> List[Int]:
    var label = "generation_config.json"
    var asset = _Asset(contents^, label)
    if not asset.doc.is_object(asset.root) or not asset.doc.contains(asset.root, "eos_token_id"):
        raise Error("missing field eos_token_id in generation_config.json")
    var eos = asset.doc.find(asset.root, "eos_token_id")
    var out = List[Int]()
    if asset.doc.is_integer(eos):
        out.append(_parse_token_id(asset.doc, eos, "eos_token_id", label))
        return out^
    if asset.doc.is_array(eos):
        if asset.doc.size(eos) == 0:
            raise Error("field eos_token_id must not be empty in generation_config.json")
        for i in range(asset.doc.size(eos)):
            out.append(_parse_token_id(asset.doc, asset.doc.at(eos, i), "eos_token_id", label))
        return out^
    raise Error("field eos_token_id must be integer or array in generation_config.json")


def _load_bpe_merge_rules(
    doc: Json, model: Int, label: String, token_to_id: Dict[String, Int], lenient: Bool
) raises -> Dict[Int, Int]:
    var merges = _require_array_field(doc, model, "merges", label)
    var rules = Dict[Int, Int]()
    var rank = 0
    for i in range(doc.size(merges)):
        var item = doc.at(merges, i)
        var left = String()
        var right = String()
        if doc.is_array(item) and doc.size(item) == 2 and doc.is_string(doc.at(item, 0)) and doc.is_string(doc.at(item, 1)):
            left = doc.string(doc.at(item, 0))
            right = doc.string(doc.at(item, 1))
        elif doc.is_string(item):
            var pair = doc.string(item)
            var b = pair.as_bytes()
            var space = -1
            for k in range(len(b)):
                if b[k] == 0x20:
                    space = k
                    break
            var second_space = False
            if space >= 0:
                for k in range(space + 1, len(b)):
                    if b[k] == 0x20:
                        second_space = True
            if space < 0 or space == 0 or space + 1 >= len(b) or second_space:
                raise Error("malformed model.merges entry in " + label)
            left = String(pair[byte=0:space])
            right = String(pair[byte = space + 1 :])
        else:
            raise Error("field model.merges must contain symbol pairs in " + label)
        var l = token_to_id.get(left)
        var r = token_to_id.get(right)
        var res = token_to_id.get(left + right)
        if not l or not r or not res:
            if lenient:
                rank += 1
                continue
            raise Error("model.merges references a symbol outside model.vocab in " + label)
        var key = _merge_pair_key(l.value(), r.value())
        if key in rules:
            raise Error("duplicate merge pair in model.merges")
        rules[key] = (rank << 32) | res.value()
        rank += 1
    return rules^


def _is_visible_byte(byte: Int) -> Bool:
    return (byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or (byte >= 174 and byte <= 255)


def _byte_level_decoder() -> Dict[Int, Int]:
    var decoder = Dict[Int, Int]()
    var next = 256
    for byte in range(256):
        if _is_visible_byte(byte):
            decoder[byte] = byte
        else:
            decoder[next] = byte
            next += 1
    return decoder^


def _byte_level_encoder() raises -> List[String]:
    var out = List[String]()
    var next = 256
    for byte in range(256):
        var cp = byte
        if not _is_visible_byte(byte):
            cp = next
            next += 1
        var b = List[UInt8]()
        append_utf8(b, cp)
        out.append(String(unsafe_from_utf8=b))
    return out^


def _decode_byte_level_token(
    unicode: Unicode, decoder: Dict[Int, Int], token: String, id: Int
) raises -> List[UInt8]:
    var spans = utf8_codepoints(_bytes_of(token), "Tokenizer::decode token id " + String(id))
    var out = List[UInt8](capacity=len(spans))
    for span in spans:
        var found = decoder.get(span.value)
        if not found:
            raise Error(
                "Tokenizer::decode token id " + String(id)
                + " contains a character outside the byte-level alphabet"
            )
        out.append(UInt8(found.value()))
    return out^


def _load_byte_token_ids(token_to_id: Dict[String, Int]) raises -> List[Int]:
    var encoder = _byte_level_encoder()
    var ids = List[Int]()
    for byte in range(256):
        var found = token_to_id.get(encoder[byte])
        ids.append(found.value() if found else -1)
    return ids^


def _load_byte_fallback_ids(token_to_id: Dict[String, Int]) -> List[Int]:
    var ids = List[Int]()
    var digits = "0123456789ABCDEF"
    for byte in range(256):
        var name = "<0x" + digits[byte = byte >> 4 : (byte >> 4) + 1] + digits[byte = byte & 15 : (byte & 15) + 1] + ">"
        var found = token_to_id.get(name)
        ids.append(found.value() if found else -1)
    return ids^


def _hex_value(c: UInt8) -> Int:
    if c >= 0x30 and c <= 0x39:
        return Int(c - 0x30)
    if c >= 0x61 and c <= 0x66:
        return Int(c - 0x61) + 10
    if c >= 0x41 and c <= 0x46:
        return Int(c - 0x41) + 10
    return -1


def _decode_sentencepiece_token(token: String) -> List[UInt8]:
    var b = token.as_bytes()
    if len(b) == 6 and b[0] == 0x3C and b[1] == 0x30 and b[2] == 0x78 and b[5] == 0x3E:
        var hi = _hex_value(b[3])
        var lo = _hex_value(b[4])
        if hi >= 0 and lo >= 0:
            var one = List[UInt8]()
            one.append(UInt8(hi * 16 + lo))
            return one^
    var space = _bytes_of(_sp_space())
    var out = List[UInt8]()
    var i = 0
    while i < len(b):
        var is_space = i + 3 <= len(b) and b[i] == space[0] and b[i + 1] == space[1] and b[i + 2] == space[2]
        if is_space:
            out.append(0x20)
            i += 3
        else:
            out.append(b[i])
            i += 1
    return out^
