"""Deterministic natural-language source text for synthesised prompts.

Why a committed bank and not a dataset
--------------------------------------
The cluster is network-restricted and no text corpus is on disk next to the model
weights, so there is nothing to sample from. What this module does instead is
*synthesise* English prose from a committed bank of slot-filled sentence templates.
The goal is a realistic token distribution and a realistic attention pattern - ordinary
English words, ordinary sentence lengths, ordinary punctuation - not literary quality.
The text reads like release notes and incident write-ups because that is the register
the templates were written in; it is not sampled from any real document.

Determinism
-----------
Every choice is drawn from a :class:`random.Random` seeded with the caller's 64-bit
prompt seed, so the same seed yields a byte-identical stream of sentences. Nothing here
reads the clock, the process id, or global RNG state.

Per-request uniqueness
----------------------
:func:`reference_line` opens every generated text with a document-reference sentence
carrying the seed rendered in base 36. Two different seeds therefore differ inside the
first handful of tokens *by construction*, which is the property the whole prompt
machinery exists for: identical prompts plus prefix caching make prefill free and the
measured capacity then rises with prompt length (see :mod:`tre_replayer.engine.prompts`).
Relying on the sentence draws alone would make collisions merely unlikely, not
impossible, and a calibration harness should not depend on "unlikely".

The reference id costs ~8 tokens of the prompt's budget. Real serving traffic carries
ids, timestamps and hashes too, so this is a small and honest distortion rather than a
synthetic artefact; it is documented here so nobody has to rediscover it.
"""
from __future__ import annotations

import random
from typing import Iterator

#: Alphabet for the document reference. Base 36 renders a 64-bit seed in 13 characters.
_REF_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

_OPENINGS = (
    "Support ticket {ref} was filed by the {actor} covering the {system}.",
    "Incident report {ref} describes a regression in the {system}.",
    "Design note {ref} was circulated to the {actor} before the review of the {system}.",
    "Change request {ref} asks the {actor} to revisit the {system}.",
    "Field report {ref} reached the {actor} after a week of trouble with the {system}.",
    "Review thread {ref} collects what the {actor} learned while rebuilding the {system}.",
    "Postmortem {ref} was written by the {actor} the morning after the {system} failed.",
    "Working draft {ref} summarises what the {actor} expects from the {system}.",
)

_ACTORS = (
    "platform team", "on-call engineer", "capacity planning group", "release manager",
    "site reliability lead", "data engineering team", "security reviewer",
    "product analyst", "infrastructure architect", "support desk", "staff engineer",
    "operations manager", "quality lead", "integration team", "network engineer",
    "storage administrator",
)

_SYSTEMS = (
    "scheduling service", "billing pipeline", "inventory database", "search index",
    "message broker", "authentication gateway", "reporting warehouse", "cache tier",
    "deployment pipeline", "monitoring stack", "configuration store", "ingest queue",
    "routing layer", "backup system", "notification service", "audit log",
)

_SENTENCES = (
    "The {actor} noticed that the {system} began to {verb} once the {noun} grew past the "
    "level the original design assumed.",
    "Every request that arrives during a busy minute has to wait behind the {noun}, so the "
    "{adjective} path is the one that decides how the whole {system} behaves.",
    "A {adjective} change to the {system} would remove the worst of the waiting, but it "
    "would also force the {actor} to rewrite the parts that {verb} today.",
    "There is no single number that captures the problem: the {noun} looks healthy in "
    "aggregate while one part of the {system} is clearly struggling.",
    "The {actor} measured the {noun} across a full week and found that the {adjective} "
    "hours account for most of the pain.",
    "Nothing in the current {system} stops two requests from competing for the same {noun}, "
    "which is why the {adjective} case is so hard to reproduce.",
    "When the {noun} is small the {system} responds quickly, and when it is large the "
    "response time grows faster than anyone expects.",
    "The team agreed to {verb} the {adjective} parts first and to leave the {system} alone "
    "until the {noun} has been measured properly.",
    "Documentation for the {system} was written before the {noun} mattered, so it says very "
    "little about what a {adjective} operator should do.",
    "It took three attempts to {verb} the {system} without disturbing the {noun} that the "
    "downstream consumers depend on.",
    "The {adjective} behaviour only shows up under load, which makes a small test "
    "environment almost useless for studying the {system}.",
    "A report from the {actor} suggests that the {noun} has doubled since the last review "
    "of the {system}.",
    "Most of the time the {system} does exactly what the {adjective} design intended, and "
    "the remaining cases are the ones worth writing down.",
    "Anyone who has to {verb} the {system} at night would rather have one clear signal than "
    "a dashboard full of {adjective} charts.",
    "The {noun} is easy to observe from the outside but hard to attribute, because the "
    "{system} reports it only as a total.",
    "After the {actor} reduced the {noun}, the {adjective} complaints stopped, although the "
    "underlying cause in the {system} was never removed.",
    "Two engineers looked at the same {noun} and reached opposite conclusions about whether "
    "the {system} was healthy.",
    "The plan is to {verb} the {system} gradually, checking the {noun} after each step "
    "rather than trusting a {adjective} estimate.",
    "Capacity is not a single quantity here: the {system} can absorb a {adjective} burst and "
    "still fail on a steady stream of the same {noun}.",
    "Every workaround the {actor} has tried so far trades some of the {noun} for a {adjective} "
    "amount of extra work elsewhere in the {system}.",
    "The oldest part of the {system} predates the {noun} entirely and was never meant to "
    "{verb} under these conditions.",
    "A {adjective} morning of profiling showed where the time actually goes, and it was not "
    "where the {actor} had been looking.",
    "The {system} keeps enough history to answer the question, but nobody had written the "
    "query that turns the {noun} into something readable.",
    "Once the {actor} could see the {noun} per component, the {adjective} explanation fell "
    "apart within an hour.",
)

_NOUNS = (
    "queue", "backlog", "latency", "error rate", "working set", "request volume",
    "retry storm", "connection pool", "batch size", "memory footprint", "fan-out",
    "cache miss rate", "lock contention", "tail latency", "write amplification",
    "clock skew", "token budget", "shard count", "replication lag", "thread pool",
    "warm-up cost", "eviction rate", "arrival rate", "service time",
)

_VERBS = (
    "stall", "recover", "degrade", "retry", "spill", "throttle", "restart", "rebalance",
    "drain", "reconnect", "saturate", "back off", "shed load", "fail over", "queue up",
    "settle",
)

_ADJECTIVES = (
    "slow", "unusual", "expensive", "quiet", "fragile", "sudden", "steady", "narrow",
    "obvious", "hidden", "patient", "brief", "familiar", "awkward", "careful", "noisy",
    "modest", "stubborn", "routine", "unexpected",
)

_SLOT_BANKS = {
    "actor": _ACTORS,
    "system": _SYSTEMS,
    "noun": _NOUNS,
    "verb": _VERBS,
    "adjective": _ADJECTIVES,
}


def reference_id(seed: int) -> str:
    """The 64-bit ``seed`` rendered in base 36 (13 characters, zero-padded).

    Fixed width so the opening sentence has a fixed token cost and two seeds cannot
    produce ids where one is a prefix of the other.
    """
    value = int(seed) & ((1 << 64) - 1)
    digits = []
    for _ in range(13):
        value, rest = divmod(value, 36)
        digits.append(_REF_ALPHABET[rest])
    return "".join(reversed(digits))


def reference_line(seed: int, rng: random.Random) -> str:
    """Opening sentence carrying ``seed`` verbatim - the uniqueness guarantee."""
    template = _OPENINGS[seed % len(_OPENINGS)]
    return template.format(
        ref=reference_id(seed),
        actor=rng.choice(_ACTORS),
        system=rng.choice(_SYSTEMS),
    )


def sentence_stream(seed: int) -> Iterator[str]:
    """Endless stream of sentences for ``seed``; the first carries the reference id."""
    rng = random.Random(seed)
    yield reference_line(seed, rng)
    while True:
        template = rng.choice(_SENTENCES)
        yield template.format(**{slot: rng.choice(bank) for slot, bank in _SLOT_BANKS.items()})


class TextBuilder:
    """Grow-only prose for one seed, extended a whole sentence at a time.

    The fitting loop in :mod:`tre_replayer.engine.prompts` asks for more words when the
    text is short of its token target, so the stream has to be resumable: re-seeding and
    regenerating would be both wasteful and, once a truncation had already happened,
    wrong.
    """

    __slots__ = ("_stream", "_parts", "_words")

    def __init__(self, seed: int) -> None:
        self._stream = sentence_stream(seed)
        self._parts: list[str] = []
        self._words = 0

    @property
    def words(self) -> int:
        return self._words

    def ensure_words(self, count: int) -> str:
        """Extend until at least ``count`` whitespace-separated words exist; return the text."""
        while self._words < count:
            sentence = next(self._stream)
            self._parts.append(sentence)
            self._words += sentence.count(" ") + 1
        return self.text()

    def text(self) -> str:
        return " ".join(self._parts)
