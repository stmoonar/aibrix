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

Languages (``corpus_lang``)
---------------------------
* :data:`LANG_EN` - the English bank above, via :class:`TextBuilder`. Byte-identical to
  what this module produced before the Chinese bank existed (a regression test pins it).
* :data:`LANG_ZH` - a Chinese bank written in the same register (slot-filled templates,
  several topics: operations, architecture, product, business analysis), opened by a
  Chinese reference sentence that carries the same base-36 id.
* :data:`LANG_MIX` (the default) - English and Chinese **sentences interleaved**, with
  the Chinese share fixed **in tokens of the model's own tokenizer**: of the prompt's
  body (the target minus the tokenizer's special tokens) ``round(body * zh_ratio)``
  tokens are budgeted to Chinese sentences and the rest to English ones.

Why tokens and not characters or sentences: the prompt is cut to an exact token count
with the model's tokenizer, and one Chinese character costs a different number of tokens
under each fleet vocabulary (Qwen ~1, Llama-3 ~1-2), while an English word costs ~1.3.
A character or sentence ratio would therefore give each model a different - and
length-dependent - share of what the engine actually processes; a token budget gives
every model the same share of its own prefill work.

Why deterministic interleaving and not a coin per sentence: a 128-token prompt holds
four or five sentences, so a fair coin would leave a large fraction of short prompts
far from 1:1 (or entirely monolingual). :func:`budgeted_text` instead always appends
the language whose budget is *proportionally* least spent, and cuts the last sentence of
a language at the token where its budget runs out, so the share is exact up to a few
tokens of tokenizer boundary effects once the prompt is long enough to hold more than
its opening. The mix opens with the reference sentence (so the uniqueness argument above
is unchanged) in the majority language - English at ``zh_ratio <= 0.5`` - and its ~20
tokens count against that language's budget. Consequences: below ~50 tokens a 0.5 mix
is mostly the English opening (measured on the fleet tokenizers: within 0.01 of 0.5 from
64 tokens, within 0.002 from 256); and ``mix`` at ratio 0 / 1 is not byte-identical to
``en`` / ``zh`` (``en`` keeps the legacy builder).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

LANG_EN = "en"
LANG_ZH = "zh"
LANG_MIX = "mix"
CORPUS_LANGS = (LANG_EN, LANG_ZH, LANG_MIX)
#: The corpus a natural-language prompt is written in unless told otherwise.
DEFAULT_CORPUS_LANG = LANG_MIX
#: Share of the prompt body (in tokens of the model's own tokenizer) given to Chinese
#: sentences under :data:`LANG_MIX`.
DEFAULT_ZH_RATIO = 0.5

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


# ------------------------------------------------------------------ the Chinese bank
#
# Same register and slot names as the English bank, several topics, fullwidth
# punctuation and no ASCII at all except the base-36 id of the opening - so a token that
# overlaps a CJK character is exactly a token of Chinese text, which is how the share is
# measured (see :func:`is_cjk`).

_ZH_OPENINGS = (
    "工单 {ref} 由{actor}提交，内容涉及{system}。",
    "事故报告 {ref} 记录了{system}出现的一次性能回退。",
    "设计说明 {ref} 在评审{system}之前已经发给了{actor}。",
    "变更申请 {ref} 请求{actor}重新审视{system}。",
    "现场报告 {ref} 在{system}连续出问题一周之后送到了{actor}手里。",
    "讨论帖 {ref} 汇总了{actor}重建{system}过程中学到的经验。",
    "复盘文档 {ref} 是{actor}在{system}故障后的第二天早上写成的。",
    "工作草稿 {ref} 概述了{actor}对{system}的期望。",
)

_ZH_ACTORS = (
    "平台团队", "值班工程师", "容量规划小组", "发布经理", "站点可靠性负责人",
    "数据工程团队", "安全审查员", "产品分析师", "基础设施架构师", "客服中心",
    "资深工程师", "运营经理", "质量负责人", "集成团队", "网络工程师", "存储管理员",
)

_ZH_SYSTEMS = (
    "调度服务", "计费流水线", "库存数据库", "搜索索引", "消息队列", "认证网关",
    "报表数仓", "缓存层", "部署流水线", "监控平台", "配置中心", "数据接入队列",
    "路由层", "备份系统", "通知服务", "审计日志", "推荐引擎", "风控模型",
    "客服机器人", "订单系统",
)

_ZH_SENTENCES = (
    "{actor}注意到，一旦{noun}超过最初设计时假设的水平，{system}就会开始{verb}。",
    "繁忙时段到达的每个请求都得排在{noun}后面，所以那条{adjective}路径决定了整个{system}的表现。",
    "对{system}做一次{adjective}改动可以消除最严重的等待，但也会迫使{actor}重写那些至今仍会{verb}的部分。",
    "问题无法用单个数字概括：从总体看{noun}很健康，可{system}里有一部分明显很吃力。",
    "{actor}连续一周测量了{noun}，发现那些{adjective}时段造成了大部分麻烦。",
    "当前的{system}没有任何机制阻止两个请求争抢同一份{noun}，这正是{adjective}情形难以复现的原因。",
    "{noun}较小时{system}响应很快，一旦变大，响应时间的增长速度远超所有人的预期。",
    "团队商定先处理{adjective}部分，在{noun}被认真测量之前暂不改动{system}。",
    "{system}的文档写于{noun}还无关紧要的年代，因此几乎没有提到{adjective}操作员该怎么做。",
    "前后尝试了三次，才让{system}顺利{verb}，同时没有扰动下游消费者依赖的{noun}。",
    "这种{adjective}行为只在高负载下出现，所以小规模测试环境几乎无法用来研究{system}。",
    "{actor}的一份报告指出，自上次评审{system}以来，{noun}已经翻了一倍。",
    "大多数时候{system}完全按照{adjective}设计意图运行，剩下的例外才值得记录下来。",
    "任何需要在深夜处理{system}的人，都宁愿要一个清晰的信号，也不想面对满屏{adjective}图表。",
    "{noun}从外部很容易观察，却很难归因，因为{system}只汇报一个总数。",
    "{actor}压低{noun}之后，{adjective}投诉随即消失，不过{system}里的根本原因始终没有排除。",
    "两位工程师看着同一份{noun}，却对{system}是否健康得出了相反的结论。",
    "计划是逐步调整{system}，每走一步都重新检查{noun}，而不是依赖某个{adjective}估算。",
    "容量在这里并不是单一的量：{system}能扛住一次{adjective}突发，却可能在持续涌入的{noun}面前失守。",
    "{actor}至今尝试过的每种变通办法，都是拿{noun}上的收益，换来{system}别处一份{adjective}额外负担。",
    "{system}最古老的部分比{noun}这个概念出现得还早，从来没打算在这种条件下{verb}。",
    "一个{adjective}上午的性能剖析显示了时间真正花在哪里，而那并不是{actor}一直盯着的地方。",
    "{system}保存了足够的历史数据来回答这个问题，只是还没有人写出把{noun}变成可读报表的查询。",
    "当{actor}能够按组件查看{noun}之后，那个{adjective}解释不到一小时就站不住脚了。",
    "季度复盘会上，{actor}把{system}的{noun}和去年同期放在一起比较，差距主要来自几个{adjective}环节。",
    "为了让新用户更快上手，产品组计划简化{system}的配置流程，并在界面上直接展示{noun}。",
    "财务部门关心的是{noun}每上升一点会带来多少额外成本，而不是{system}内部的技术细节。",
    "{actor}在周会上提议把{system}拆成几个独立模块，这样即使某处开始{verb}，也不会拖垮整体。",
    "从用户反馈来看，{adjective}体验往往比功能缺失更容易引发流失，这一点在{system}上尤其突出。",
    "数据分析显示，{noun}与客户满意度之间存在明显关联，但因果方向仍需{actor}进一步验证。",
    "新版本上线前，{actor}编写了一份检查清单，逐项核对{system}的回滚步骤和{noun}告警阈值。",
    "培训材料里专门加入了一节案例，讲述{system}如何在一次{adjective}故障中逐步{verb}。",
)

_ZH_NOUNS = (
    "排队长度", "积压任务", "响应延迟", "错误率", "工作集", "请求量", "重试风暴",
    "连接池", "批大小", "内存占用", "扇出规模", "缓存未命中率", "锁竞争", "尾部延迟",
    "写放大", "时钟偏差", "令牌预算", "分片数量", "复制延迟", "线程池", "预热开销",
    "淘汰速率", "到达速率", "服务时间", "转化率", "退货率", "库存周转", "用户留存",
)

_ZH_VERBS = (
    "停顿", "恢复", "变慢", "反复重试", "溢出到磁盘", "限流", "重启", "重新均衡",
    "排空", "重新连接", "饱和", "主动退避", "丢弃负载", "故障转移", "排起长队", "趋于稳定",
)

_ZH_ADJECTIVES = (
    "缓慢的", "反常的", "昂贵的", "安静的", "脆弱的", "突发的", "平稳的", "狭窄的",
    "明显的", "隐蔽的", "短暂的", "熟悉的", "别扭的", "谨慎的", "嘈杂的", "温和的",
    "顽固的", "例行的", "意外的",
)

_ZH_SLOT_BANKS = {
    "actor": _ZH_ACTORS,
    "system": _ZH_SYSTEMS,
    "noun": _ZH_NOUNS,
    "verb": _ZH_VERBS,
    "adjective": _ZH_ADJECTIVES,
}

#: Salt for the Chinese stream's RNG, so it is independent of the English stream of the
#: same seed (the mix interleaves both and must not correlate their draws).
_ZH_STREAM_SALT = 0x7A68_5F63_6F72_7075  # "zh_corpu"


def is_cjk(ch: str) -> bool:
    """A CJK ideograph or CJK / fullwidth punctuation - what "Chinese text" means here."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # Extension A
        or 0x3000 <= code <= 0x303F  # CJK symbols and punctuation (、。「」 ...)
        or 0xFF00 <= code <= 0xFFEF  # fullwidth forms (，：！？ ...)
    )


def zh_reference_line(seed: int, rng: random.Random) -> str:
    """Chinese opening sentence carrying ``seed`` verbatim (same id as :func:`reference_line`)."""
    template = _ZH_OPENINGS[seed % len(_ZH_OPENINGS)]
    return template.format(
        ref=reference_id(seed),
        actor=rng.choice(_ZH_ACTORS),
        system=rng.choice(_ZH_SYSTEMS),
    )


def zh_sentence_stream(seed: int, *, with_reference: bool) -> Iterator[str]:
    """Endless stream of Chinese sentences for ``seed``; optionally opened by the id line."""
    rng = random.Random((int(seed) & ((1 << 64) - 1)) ^ _ZH_STREAM_SALT)
    if with_reference:
        yield zh_reference_line(seed, rng)
    while True:
        template = rng.choice(_ZH_SENTENCES)
        yield template.format(**{slot: rng.choice(bank) for slot, bank in _ZH_SLOT_BANKS.items()})


@dataclass(frozen=True)
class BudgetedText:
    """What :func:`budgeted_text` built: the text and its token accounting."""

    text: str
    zh_tokens: int
    en_tokens: int
    #: ``(lang, sentence)`` in order - the evidence of how the text was interleaved.
    parts: tuple[tuple[str, str], ...]


def _separator(lang: str) -> str:
    """What goes in front of a sentence of ``lang`` that follows another sentence.

    Nothing in front of Chinese: a space between an English full stop and a CJK
    character is not how mixed text is written, and every fleet vocabulary spends extra
    (often byte-fallback) tokens on the space-plus-ideograph pair. One space in front of
    English, whatever precedes it.
    """
    return "" if lang == LANG_ZH else " "


def budgeted_text(
    seed: int,
    *,
    lang: str,
    zh_ratio: float,
    budget: int,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> BudgetedText:
    """Chinese / English / mixed prose of about ``budget`` tokens under ``encode``.

    ``budget`` is split into ``zh = round(budget * zh_ratio)`` and ``en = budget - zh``
    (``lang`` zh forces ratio 1, en forces 0). Sentences are appended one at a time, each
    from the language whose budget is proportionally least spent (ties go to the
    language with more tokens left, then to Chinese), and a sentence that would overrun
    its language's budget is cut at the token where the budget ends - after which that
    language is closed.

    A sentence is costed *in context*: ``len(encode(previous + separator + sentence)) -
    len(encode(previous))``, so whatever the tokenizer does at the join (a full stop
    merging into the next word, a byte-level split) is charged to the sentence that
    caused it. The joined text therefore lands within a token or two of ``budget``; the
    exact-length fit in :func:`tre_replayer.engine.prompts.build_natural_prompt` closes
    the rest.

    The first sentence is always the reference line - Chinese for ``zh`` and for a ``mix``
    with ``zh_ratio > 0.5``, English otherwise - and it is never cut unless it alone
    exceeds ``budget``: that is what keeps every seed's prompt distinct within its first
    handful of tokens.

    A byte-level vocabulary can split one CJK character over several tokens; a cut
    between them decodes to U+FFFD, which is stripped rather than sent (it would
    re-encode to a different id sequence and is not text anyway).
    """
    if lang not in CORPUS_LANGS:
        raise ValueError(f"unknown corpus language {lang!r} (expected one of {CORPUS_LANGS})")
    ratio = 1.0 if lang == LANG_ZH else 0.0 if lang == LANG_EN else float(zh_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"zh_ratio must be within [0, 1], got {zh_ratio!r}")
    budget = max(0, int(budget))
    total = {LANG_ZH: int(round(budget * ratio)), LANG_EN: 0}
    total[LANG_EN] = budget - total[LANG_ZH]
    left = dict(total)
    used = {LANG_ZH: 0, LANG_EN: 0}
    # The reference line opens in the majority language (see the module docstring).
    opening = LANG_ZH if (lang == LANG_ZH or (lang == LANG_MIX and ratio > 0.5)) else LANG_EN
    other = LANG_EN if opening == LANG_ZH else LANG_ZH
    streams = {
        LANG_EN: sentence_stream(seed),
        LANG_ZH: zh_sentence_stream(seed, with_reference=(opening == LANG_ZH)),
    }
    if opening == LANG_ZH:
        next(streams[LANG_EN])  # the English stream's own reference line: never sent twice
    parts: list[tuple[str, str]] = []
    previous = {"text": "", "tokens": 0}

    def cost(which: str, piece: str) -> int:
        if not parts:
            return len(encode(piece))
        joined = previous["text"] + _separator(which) + piece
        return len(encode(joined)) - previous["tokens"]

    def take(which: str, sentence: str, room: int) -> None:
        piece, spent, was_cut = sentence, cost(which, sentence), False
        if spent > room:
            was_cut = True
            # The overrun is in-context, the cut is on the sentence's own ids; the two can
            # differ by a boundary token, which the exact-length fit closes afterwards.
            ids = list(encode(sentence))
            keep = max(0, len(ids) - (spent - room))
            piece = decode(ids[:keep]).rstrip("\ufffd").rstrip()
            spent = cost(which, piece) if piece else 0
        if piece:
            parts.append((which, piece))
            previous["text"] = piece
            previous["tokens"] = len(encode(piece))
        used[which] += spent
        # At least one token per sentence comes off the budget, whatever the in-context
        # cost says: a tokenizer that folds a whole sentence into the previous word
        # (cost 0) must not stall the loop. The fit closes any shortfall this leaves.
        left[which] -= max(1, spent)
        if was_cut or not piece:
            left[which] = 0  # the language's last sentence was cut: it is closed

    # The reference line first, charged to its own language. If it overruns that
    # language's share the overrun comes out of the other language's share, so the total
    # stays ``budget``; only a prompt too short for the line itself gets it cut.
    take(opening, next(streams[opening]), budget)
    remaining = budget - used[opening]
    left[opening] = max(0, min(left[opening], remaining))
    left[other] = max(0, remaining - left[opening]) if total[other] > 0 else 0

    while left[LANG_ZH] > 0 or left[LANG_EN] > 0:
        open_langs = [lg for lg in (LANG_ZH, LANG_EN) if left[lg] > 0]
        which = min(
            open_langs,
            key=lambda lg: (used[lg] / total[lg] if total[lg] else 1.0, -left[lg], lg != LANG_ZH),
        )
        take(which, next(streams[which]), left[which])

    return BudgetedText(
        text=join_parts(parts),
        zh_tokens=used[LANG_ZH],
        en_tokens=used[LANG_EN],
        parts=tuple(parts),
    )


def join_parts(parts: Sequence[tuple[str, str]]) -> str:
    """Join sentences with :func:`_separator` (nothing before Chinese, a space before English)."""
    out: list[str] = []
    for which, sentence in parts:
        if out:
            out.append(_separator(which))
        out.append(sentence)
    return "".join(out)
