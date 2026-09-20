from __future__ import annotations

import pytest

from tre_replayer.engine import prompts


def test_token_id_prompt_has_exact_length_and_legal_ids() -> None:
    for target in (1, 4, 128, 512, 1024):
        ids = prompts.build_token_id_prompt(target, f"run|cell|{target}")
        assert len(ids) == target
        assert all(prompts.TOKEN_ID_MIN <= i < prompts.TOKEN_ID_MAX for i in ids)


def test_token_id_prompt_is_deterministic_in_the_seed_key() -> None:
    a = prompts.build_token_id_prompt(64, "run|i128_o128_c4|7")
    b = prompts.build_token_id_prompt(64, "run|i128_o128_c4|7")
    assert a == b


def test_token_id_prompts_differ_inside_the_preamble() -> None:
    """No two requests may share a usable prefix, or prefill is served from cache."""
    seen = {}
    for seq in range(500):
        ids = prompts.build_token_id_prompt(256, f"run|cell|{seq}")
        head = tuple(ids[: prompts.PREAMBLE_TOKENS])
        assert head not in seen, f"prefix collision between {seq} and {seen[head]}"
        seen[head] = seq
    assert len(seen) == 500


def test_text_prompt_word_count_and_uniqueness() -> None:
    first_words = set()
    for seq in range(200):
        text = prompts.build_text_prompt(128, f"run|cell|{seq}")
        assert len(text.split()) == 128
        head = tuple(text.split()[: prompts.TEXT_PREAMBLE_WORDS])
        assert head not in first_words
        first_words.add(head)


def test_build_prompt_dispatches_and_rejects_unknown_mode() -> None:
    assert isinstance(prompts.build_prompt(16, "k", mode=prompts.MODE_TOKEN_IDS), list)
    assert isinstance(prompts.build_prompt(16, "k", mode=prompts.MODE_TEXT), str)
    with pytest.raises(ValueError):
        prompts.build_prompt(16, "k", mode="constant")


def test_seed_is_process_stable() -> None:
    # blake2b, not hash(): PYTHONHASHSEED must not change a replay's prompts. The golden
    # value pins the seed policy, so changing it is a deliberate, visible act.
    assert prompts.prompt_seed("run|cell|0") == 0x2EA4C25EE249B697
    assert prompts.prompt_seed("a") == prompts.prompt_seed("a")
    assert prompts.prompt_seed("a") != prompts.prompt_seed("b")
