"""The prompt corpus languages: en (legacy), zh, and the default 1:1 zh/en mix.

What must hold under every language, as under the English corpus before it: the exact
realised token count, determinism per seed key, and a per-request unique head. What the
mix adds: the Chinese share of the prompt, counted in tokens of the model's own
tokenizer, is ``zh_ratio`` up to a few tokens. And ``en`` must be byte-identical to the
builder that predates the other languages - every past run was made with it.

Most tests use :class:`CjkStubTokenizer` (one token per CJK character, one per ASCII
word) so they stay hermetic; the last group re-checks the properties against the real
fleet tokenizers when they are on local disk, and skips otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import pytest

from tre_replayer.engine import corpus, model_tokenizer, prompt_store, prompts
from tre_replayer.engine.http_sender import StreamingHttpSender, StreamResult
from tre_replayer.engine.schedule import ScheduledRequest

_CJK_CLASS = "　-〿㐀-䶿一-鿿＀-￯"
_TOKEN = re.compile(rf"\s*(?:[{_CJK_CLASS}]|[^\s{_CJK_CLASS}]+)")


class CjkStubTokenizer:
    """One token per CJK character and per ASCII word; each token keeps its leading
    whitespace, so decode is an exact inverse of encode. One leading special token."""

    overhead = 1
    filler = " the"
    path = "<cjk-stub>"

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_piece: dict[int, str] = {}

    def _id(self, piece: str) -> int:
        if piece not in self._to_id:
            index = len(self._to_id) + 1
            self._to_id[piece] = index
            self._to_piece[index] = piece
        return self._to_id[piece]

    def encode_plain(self, text: str) -> list[int]:
        return [self._id(piece) for piece in _TOKEN.findall(text)]

    def decode_plain(self, ids) -> str:
        return "".join(self._to_piece[i] for i in ids)

    def count(self, text: str) -> int:
        return len(self.encode_plain(text)) + self.overhead

    def cjk_token_count(self, text: str) -> int:
        return sum(1 for piece in _TOKEN.findall(text) if any(corpus.is_cjk(c) for c in piece))


class WordStubTokenizer:
    """The stub the English tests have always used: one token per whitespace word."""

    overhead = 1
    filler = " the"
    path = "<stub>"

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_word: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._to_id:
            index = len(self._to_id) + 1
            self._to_id[word] = index
            self._to_word[index] = word
        return self._to_id[word]

    def encode_plain(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]

    def decode_plain(self, ids) -> str:
        return " ".join(self._to_word[i] for i in ids)

    def count(self, text: str) -> int:
        return len(self.encode_plain(text)) + self.overhead


def _zh_share(tok, text: str) -> float:
    plain = tok.count(text) - tok.overhead
    return tok.cjk_token_count(text) / plain


# ------------------------------------------------------------------ defaults and options


def test_the_default_corpus_is_the_one_to_one_mix() -> None:
    assert corpus.DEFAULT_CORPUS_LANG == corpus.LANG_MIX == prompts.DEFAULT_CORPUS_LANG
    assert corpus.DEFAULT_ZH_RATIO == prompts.DEFAULT_ZH_RATIO == 0.5
    assert set(corpus.CORPUS_LANGS) == {"en", "zh", "mix"}


def test_an_unknown_language_or_ratio_is_refused_before_any_work() -> None:
    tok = CjkStubTokenizer()
    with pytest.raises(ValueError, match="unknown corpus language"):
        prompts.build_natural_prompt(64, "k", tokenizer=tok, corpus_lang="fr")
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="zh_ratio"):
            prompts.build_natural_prompt(64, "k", tokenizer=tok, zh_ratio=bad)
    with pytest.raises(ValueError, match="unknown corpus language"):
        StreamingHttpSender("http://gw", stream_call=lambda *a: None, corpus_lang="fr")


# ------------------------------------------------------------------ exactness and share


@pytest.mark.parametrize("lang", ["mix", "zh"])
@pytest.mark.parametrize("target", [2, 8, 16, 32, 128, 513, 1024, 1600, 4096])
def test_every_language_hits_the_exact_token_count(lang: str, target: int) -> None:
    tok = CjkStubTokenizer()
    text = prompts.build_natural_prompt(target, f"exact|{lang}|{target}", tokenizer=tok, corpus_lang=lang)
    assert tok.count(text) == target


@pytest.mark.parametrize("target,tolerance", [(128, 0.08), (256, 0.05), (1024, 0.02), (4096, 0.01)])
def test_the_mix_is_half_chinese_by_token(target: int, tolerance: float) -> None:
    """Counted in tokens - the unit the prompt is cut in - not characters or sentences."""
    tok = CjkStubTokenizer()
    shares = [
        _zh_share(tok, prompts.build_natural_prompt(target, f"share|{target}|{i}", tokenizer=tok))
        for i in range(40)
    ]
    assert all(abs(s - 0.5) <= tolerance for s in shares), (min(shares), max(shares))
    assert abs(sum(shares) / len(shares) - 0.5) <= tolerance / 2


@pytest.mark.parametrize("ratio", [0.25, 0.75])
def test_zh_ratio_sets_the_chinese_share(ratio: float) -> None:
    tok = CjkStubTokenizer()
    for i in range(10):
        text = prompts.build_natural_prompt(1024, f"ratio|{i}", tokenizer=tok, zh_ratio=ratio)
        assert tok.count(text) == 1024
        assert abs(_zh_share(tok, text) - ratio) <= 0.02


def test_the_mix_interleaves_sentences_rather_than_concatenating_two_blocks() -> None:
    tok = CjkStubTokenizer()
    built = corpus.budgeted_text(
        prompts.prompt_seed("interleave"), lang="mix", zh_ratio=0.5, budget=1024,
        encode=tok.encode_plain, decode=tok.decode_plain,
    )
    langs = [lang for lang, _ in built.parts]
    switches = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
    assert switches >= 10
    assert abs(built.zh_tokens - 512) <= 2 and abs(built.en_tokens - 512) <= 2


def test_zh_mode_is_chinese_apart_from_the_reference_id() -> None:
    tok = CjkStubTokenizer()
    text = prompts.build_natural_prompt(512, "zh|only", tokenizer=tok, corpus_lang="zh")
    ascii_runs = re.findall(r"[A-Za-z0-9]+", text)
    assert ascii_runs[0] == corpus.reference_id(prompts.prompt_seed("zh|only"))
    assert _zh_share(tok, text) > 0.95


def test_the_chinese_bank_is_prose_not_a_repeated_character() -> None:
    """The corpus exists so prompts are language, not "token token token"."""
    stream = corpus.zh_sentence_stream(7, with_reference=True)
    sentences = [next(stream) for _ in range(300)]
    for sentence in sentences:
        # No character three times running (the opening's id is digits, not prose).
        assert not re.search(rf"([{_CJK_CLASS}])\1\1", sentence), sentence
        assert sentence.endswith("。")
    assert len(set(sentences)) > 250
    # Every slot of every template is filled: no braces reach the wire.
    assert not any("{" in s or "}" in s for s in sentences)
    # No ASCII (not even a space) outside the opening, whose id is the only ASCII, so a
    # token of a Chinese sentence is a token of Chinese text.
    assert all(not ch.isascii() for s in sentences[1:] for ch in s)


# --------------------------------------------------------------- determinism, uniqueness


@pytest.mark.parametrize("lang", ["mix", "zh"])
def test_the_same_seed_key_gives_the_same_bytes(lang: str) -> None:
    first = prompts.build_natural_prompt(700, "run|cell|7", tokenizer=CjkStubTokenizer(), corpus_lang=lang)
    second = prompts.build_natural_prompt(700, "run|cell|7", tokenizer=CjkStubTokenizer(), corpus_lang=lang)
    assert first == second


@pytest.mark.parametrize("lang", ["mix", "zh"])
def test_every_prompt_carries_its_own_reference_id_in_its_head(lang: str) -> None:
    tok = CjkStubTokenizer()
    keys = [f"run|cell|{i}" for i in range(200)]
    texts = [prompts.build_natural_prompt(128, key, tokenizer=tok, corpus_lang=lang) for key in keys]
    assert len(set(texts)) == 200
    for key, text in zip(keys, texts):
        ref = corpus.reference_id(prompts.prompt_seed(key))
        assert ref in text[:40]
    heads = {tuple(tok.encode_plain(t)[:8]) for t in texts}
    assert len(heads) == 200


# ------------------------------------------------------------------ en regression (bytes)

#: sha256 of json.dumps(prompts) built by the English builder on main before the corpus
#: languages existed (tre/replayer at 19781f50), for the keys and targets below.
GOLDEN_EN_STUB = "71f8149dd5be1209feb770eabf48127d677582b3f1b4fe9cdc6b3bf1f61cd61e"
GOLDEN_EN_FLEET = {
    "dsqwen-7b": "ed9dc73b9933deaf2e194c8a019006fe980c186bde17b06355113cb07ecf6d6d",
    "dsllama-8b": "a84829f06b5c5396e8747e7ee5b08d31803b7b34bd70909259a81cdf826631da",
    "dsqwen-14b": "ed9dc73b9933deaf2e194c8a019006fe980c186bde17b06355113cb07ecf6d6d",
}


def _digest(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def test_en_is_byte_identical_to_the_builder_before_the_corpus_languages() -> None:
    texts = [
        prompts.build_natural_prompt(t, f"golden|{t}|{i}", tokenizer=WordStubTokenizer(), corpus_lang="en")
        for t in (2, 8, 64, 128, 512, 1600, 4096)
        for i in range(10)
    ]
    assert _digest(texts) == GOLDEN_EN_STUB


# ------------------------------------------------------------------ materialiser / sender


def _requests(count: int, tokens: int) -> list[ScheduledRequest]:
    return [
        ScheduledRequest(request_id=f"m-{i:06d}", model="m", scheduled_offset_s=i * 0.1,
                         prompt_tokens=tokens + i, max_output_tokens=16)
        for i in range(count)
    ]


def test_materialised_mix_prompts_are_the_inline_ones_and_never_miss(tmp_path) -> None:
    tok = CjkStubTokenizer()
    requests = _requests(40, 200)
    store = prompt_store.materialize_prompts(
        requests, path=tmp_path / "c.prompts.jsonl", tokenizer=tok, corpus_lang="mix", zh_ratio=0.5,
    )
    for request in requests:
        text = store.get(request.request_id)
        assert text == prompts.build_prompt(
            request.prompt_tokens, prompt_store.sender_seed_key("m", request.request_id),
            model="m", tokenizer=tok, corpus_lang="mix", zh_ratio=0.5,
        )
        assert tok.count(text) == request.prompt_tokens
    assert store.misses == 0
    rows = [json.loads(line) for line in (tmp_path / "c.prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40 and all(r["prompt"] == store.get(r["request_id"]) for r in rows)


def test_the_senders_inline_fallback_uses_the_senders_corpus(monkeypatch) -> None:
    """A miss must still send what the materialiser would have: same corpus, same bytes."""
    seen = {}

    def fake_build(tokens, key, **kwargs):
        seen.update(kwargs)
        return "x"

    monkeypatch.setattr("tre_replayer.engine.http_sender.build_prompt", fake_build)
    sender = StreamingHttpSender(
        "http://gw", stream_call=lambda *a: StreamResult(200, 1.0, 2.0, 1, 1),
        corpus_lang="zh", zh_ratio=0.5,
    )
    try:
        sender._prompt_for(_requests(1, 64)[0], 64)
    finally:
        sender.close()
    assert seen["corpus_lang"] == "zh" and seen["zh_ratio"] == 0.5


# ------------------------------------------------------------------ the real tokenizers

FLEET = tuple(model_tokenizer.FLEET_TOKENIZER_PATHS)


def _fleet_tokenizer(model: str):
    path = model_tokenizer.FLEET_TOKENIZER_PATHS[model]
    if not os.path.isdir(path):
        pytest.skip(f"{model} tokenizer not on this host ({path})")
    pytest.importorskip("transformers")
    return model_tokenizer.load_tokenizer(model, tokenizer_path=path)


@pytest.mark.parametrize("model", FLEET)
def test_fleet_tokenizers_exact_count_and_half_chinese(model: str) -> None:
    tok = _fleet_tokenizer(model)
    for target in (128, 300, 1024, 3000):
        for i in range(3):
            text = prompts.build_natural_prompt(target, f"fleet|{target}|{i}", tokenizer=tok)
            assert tok.count(text) == target
            assert abs(_zh_share(tok, text) - 0.5) <= (0.03 if target < 256 else 0.015)


@pytest.mark.parametrize("model", FLEET)
def test_fleet_tokenizers_en_is_byte_identical_to_before(model: str) -> None:
    tok = _fleet_tokenizer(model)
    texts = [
        prompts.build_natural_prompt(t, f"golden|{t}|{i}", tokenizer=tok, corpus_lang="en")
        for t in (8, 128, 512, 1600)
        for i in range(5)
    ]
    assert _digest(texts) == GOLDEN_EN_FLEET[model]


# ------------------------------------------------------------ byte-level splits (U+FFFD)


class ByteSplitStubTokenizer(CjkStubTokenizer):
    """Like :class:`CjkStubTokenizer`, but every CJK character is two tokens (as a
    byte-level vocabulary splits a rare character), and decoding half of one yields
    U+FFFD - the case the corpus cut and the fit must strip rather than send."""

    path = "<byte-split-stub>"

    def encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _TOKEN.findall(text):
            if corpus.is_cjk(piece[-1]):
                ids += [self._id(("a", piece)), self._id(("b", piece))]
            else:
                ids.append(self._id(("w", piece)))
        return ids

    def decode_plain(self, ids) -> str:
        out, keys, i = [], [self._to_piece[j] for j in ids], 0
        while i < len(keys):
            kind, piece = keys[i]
            if kind == "w":
                out.append(piece)
                i += 1
            elif kind == "a" and i + 1 < len(keys) and keys[i + 1] == ("b", piece):
                out.append(piece)
                i += 2
            else:
                out.append(piece[:-1] + "\ufffd")
                i += 1
        return "".join(out)

    def cjk_token_count(self, text: str) -> int:
        return sum(1 for i in self.encode_plain(text) if self._to_piece[i][0] != "w")


@pytest.mark.parametrize("lang", ["mix", "zh"])
@pytest.mark.parametrize("target", [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 377, 1000])
def test_a_cut_through_a_split_character_is_stripped_and_the_count_stays_exact(lang, target) -> None:
    tok = ByteSplitStubTokenizer()
    for i in range(5):
        text = prompts.build_natural_prompt(target, f"split|{lang}|{target}|{i}", tokenizer=tok,
                                            corpus_lang=lang)
        assert tok.count(text) == target
        assert "\ufffd" not in text


def test_a_high_ratio_mix_opens_in_chinese_and_holds_its_share() -> None:
    """The reference line is charged to the language it is written in; opening a 0.75
    mix in English would leave its short prompts well under their Chinese share."""
    tok = CjkStubTokenizer()
    key = "high|ratio"
    text = prompts.build_natural_prompt(256, key, tokenizer=tok, zh_ratio=0.75)
    assert corpus.is_cjk(text[0])
    assert corpus.reference_id(prompts.prompt_seed(key)) in text[:20]
    assert abs(_zh_share(tok, text) - 0.75) <= 0.03


def test_a_mix_at_ratio_one_is_the_zh_corpus() -> None:
    tok = CjkStubTokenizer()
    for target in (16, 128, 700):
        assert prompts.build_natural_prompt(target, "one", tokenizer=tok, zh_ratio=1.0) == \
            prompts.build_natural_prompt(target, "one", tokenizer=tok, corpus_lang="zh")


def test_run_trace_rejects_a_ratio_outside_the_unit_interval_as_a_usage_error() -> None:
    from tre_replayer import run_trace

    with pytest.raises(SystemExit) as exc:
        run_trace.main(["--trace", "t.json", "--dry-run", "--zh-ratio", "1.5"])
    assert exc.value.code == 2
    assert run_trace.effective_zh_ratio("en", 0.5) == 0.0
    assert run_trace.effective_zh_ratio("zh", 0.5) == 1.0


@pytest.mark.parametrize("model", FLEET)
def test_fleet_pool_materialisation_carries_the_corpus_to_the_workers(model: str, tmp_path) -> None:
    """The process-pool branch (initargs), not the injected-tokenizer shortcut."""
    tok = _fleet_tokenizer(model)
    requests = [
        ScheduledRequest(request_id=f"{model}-{i:06d}", model=model, scheduled_offset_s=i * 0.1,
                         prompt_tokens=64 + 37 * i, max_output_tokens=16)
        for i in range(24)
    ]
    for lang in ("mix", "zh"):
        store = prompt_store.materialize_prompts(
            requests, path=tmp_path / f"{lang}.prompts.jsonl", processes=2, chunk_size=4,
            tokenizer_path=model_tokenizer.FLEET_TOKENIZER_PATHS[model], corpus_lang=lang,
        )
        for request in requests:
            text = store.get(request.request_id)
            assert tok.count(text) == request.prompt_tokens
            assert text == prompts.build_prompt(
                request.prompt_tokens, prompt_store.sender_seed_key(model, request.request_id),
                model=model, tokenizer=tok, corpus_lang=lang)
        assert store.misses == 0


@pytest.mark.parametrize("model", FLEET)
def test_fleet_tokenizers_tiny_targets_are_exact(model: str) -> None:
    tok = _fleet_tokenizer(model)
    for lang in ("mix", "zh"):
        for target in range(2, 41):
            text = prompts.build_natural_prompt(target, f"tiny|{target}", tokenizer=tok, corpus_lang=lang)
            assert tok.count(text) == target
            assert "\ufffd" not in text
