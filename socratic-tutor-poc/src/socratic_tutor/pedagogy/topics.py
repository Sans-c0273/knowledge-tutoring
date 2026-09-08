"""Pipeline step 2, first half: map a student message onto a syllabus topic.

Tech Spec §1 step 2 and §2.2 ("map student question → syllabus topic → look up
stored level"). The syllabus is the teacher-provided topic list; it is also what
the onboarding questionnaire is generated from, so topic strings are identical
across the Student Model, this mapper, and the KL Map (`id` is shared with the
KL node id).

Matching here is **deterministic** — normalised string comparison and a bounded
similarity fallback, no LLM. Intent Detection (Call A) is a separate step and
must not be duplicated inside topic resolution; if semantic matching is needed
later it goes behind `resolve_topic`, which is why this returns a scored
`TopicMatch` rather than a bare topic.

Ambiguity is surfaced, never guessed away. A bare "linear equations" matches
both One-Step and Two-Step Linear Equations; `resolve_topic` returns
`ambiguous=True` with both candidates so the caller can ask, and prefers the
topic already under discussion when the conversation context names one — the
pedagogically correct tie-break, and what keeps multi-turn threads on one topic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.domain.klmap.lookup import normalise_text

#: Confidence when a whole label equals the whole message.
EXACT_CONFIDENCE = 1.0
#: Confidence when a topic label appears inside a longer message.
MENTION_CONFIDENCE = 0.9
#: Ceiling for the similarity fallback (typos); scaled by the actual ratio.
FUZZY_MAX_CONFIDENCE = 0.75
#: Confidence when the message is only a fragment of the label ("linear equations").
FRAGMENT_CONFIDENCE = 0.6
#: Confidence when nothing in the message matched and the topic came from context.
CONTEXT_ONLY_CONFIDENCE = 0.4
#: Below this, a match is not worth reporting at all.
MIN_CONFIDENCE = 0.35
#: Applied when the candidates could not be narrowed to one; the flag matters
#: more than the number, but the caller should see a low score in the trace.
AMBIGUITY_PENALTY = 0.5
#: Similarity floor for the typo fallback.
FUZZY_CUTOFF = 0.82
#: Shorter fragments ("of", "สมการ" is fine, "eq" is not) match too much.
MIN_FRAGMENT_CHARS = 4


class SyllabusError(Exception):
    """A syllabus file is unreadable, malformed, or has duplicate topic ids."""


class MatchKind(str, Enum):
    """How a topic was resolved — recorded in the turn trace, not just for debugging."""

    EXACT = "exact"
    MENTION = "mention"
    FUZZY = "fuzzy"
    FRAGMENT = "fragment"
    CONVERSATION_CONTEXT = "conversation_context"


class SyllabusTopic(BaseModel):
    """One teachable topic. `id` matches the KL Map node id for the same concept."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str
    name: str
    name_th: str | None = None
    aliases: list[str] = Field(default_factory=list)

    def labels(self) -> list[str]:
        """Every string a student might use for this topic, longest first.

        Longest first so a mention of "two-step linear equations" is credited to
        the full name rather than to a shorter alias inside it.
        """
        labels = [self.name, *(self.aliases or [])]
        if self.name_th:
            labels.append(self.name_th)
        return sorted({label for label in labels if label}, key=len, reverse=True)


#: `(light-independent reactions)` — a gloss a student names on its own.
_PARENTHETICAL = re.compile(r"\(([^()]*)\)")
#: Conjunctions that join two nameable things in one label.
_CONJUNCTION = re.compile(r"\s+(?:and|vs\.?)\s+|\s*/\s*|\s*&\s*", re.IGNORECASE)
#: "Limiting factors of photosynthesis": the head is what the student asks about.
_OF = re.compile(r"\s+of\s+", re.IGNORECASE)
#: Below this many normalised characters an alias matches too much ("of", "vs").
MIN_ALIAS_CHARS = 3


def derive_aliases(name: str) -> list[str]:
    """Deterministic aliases for a topic label that was never given any by hand.

    Extracted concept maps carry long compound labels — "RuBisCO and carbon
    fixation", "Calvin cycle (light-independent reactions)" — and students name
    the part they are stuck on. Without aliases the mention tier needs the whole
    label verbatim, so no real message ever resolves. The rules are a fixed,
    readable set rather than a model call: parenthetical glosses; the label with
    its glosses removed; the head before " of "; and each conjunct after splitting
    on " and ", " vs ", " / " and "&". Aliases are kept in the label's own
    spelling — `resolve_topic` normalises both sides — and compared through
    `normalise_text` for length and duplication so "Calvin-cycle" and "Calvin
    cycle" count once. Anything equal to the full label, or shorter than
    `MIN_ALIAS_CHARS` once normalised, is dropped.

    The " of " head is kept only when it is more than one word. A conjunct is a
    concept in its own right ("Cells", "RuBisCO"), but a bare head is a
    grammatical slot: "Order of Operations" → "Order", "Law of Sines" → "Law",
    "Area of a triangle" → "Area". Each of those matches at `MENTION_CONFIDENCE`,
    which outranks the conversation-context carry-over, so "in order to use it…"
    mid-thread on the quadratic formula was re-targeted to Order of Operations
    at 0.9 (cycle-2 re-review, M3). Multi-word heads — "Limiting factors",
    "Newton's laws" — are the phrases students actually type. The same bar
    applies to every fragment of a label that contains " of ", including its
    conjuncts: "Ecological and agricultural importance of photosynthesis" keeps
    "agricultural importance", not "Ecological", and "Order" cannot re-enter by
    another route.

    Pure: same label in, same list out, in a stable order. Does not touch
    `name_th`, which the extractor already supplies as its own label.
    """
    full = normalise_text(name)
    seen: set[str] = {full}
    aliases: list[str] = []

    def keep(candidate: str, *, multiword: bool = False) -> None:
        text = candidate.strip(" ,;:-")
        key = normalise_text(text)
        if len(key) < MIN_ALIAS_CHARS or key in seen:
            return
        if multiword and " " not in key:
            return
        seen.add(key)
        aliases.append(text)

    glosses = _PARENTHETICAL.findall(name)
    bare = _PARENTHETICAL.sub(" ", name)
    for phrase in (*glosses, bare):
        keep(phrase)
        head, *tail = _OF.split(phrase, maxsplit=1)
        for conjunct in _CONJUNCTION.split(phrase):
            keep(conjunct, multiword=bool(tail))
        if not tail:
            continue
        keep(head, multiword=True)
        for conjunct in _CONJUNCTION.split(head):
            keep(conjunct, multiword=True)
    return aliases


class Syllabus(BaseModel):
    """A course's topic list in teaching order (Tech Spec §2.2)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    course_id: str
    course_name: str = ""
    topics: list[SyllabusTopic] = Field(default_factory=list)

    def topic(self, key: str) -> SyllabusTopic | None:
        """Look up by id, name, Thai name, or alias. Case- and punctuation-insensitive."""
        if not key or not key.strip():
            return None
        needle = normalise_text(key)
        for topic in self.topics:
            if normalise_text(topic.id) == needle:
                return topic
        for topic in self.topics:
            if any(normalise_text(label) == needle for label in topic.labels()):
                return topic
        return None

    def position(self, topic_id: str) -> int:
        """Teaching-order index, or -1. Used only as a deterministic tie-break."""
        for index, topic in enumerate(self.topics):
            if topic.id == topic_id:
                return index
        return -1


class ConversationContext(BaseModel):
    """What the conversation was already about, most recent topic first.

    Entries may be topic ids or topic names; both resolve through `Syllabus.topic`.
    """

    recent_topics: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, *topics: str) -> ConversationContext:
        """`ConversationContext.of("Two-Step Linear Equations")` — most recent first."""
        return cls(recent_topics=[topic for topic in topics if topic])

    @property
    def current_topic(self) -> str | None:
        """The topic of the previous turn, if there was one."""
        return self.recent_topics[0] if self.recent_topics else None


class TopicMatch(BaseModel):
    """A resolved topic plus how confident, and how ambiguous, the resolution was.

    `ambiguous` is the caller's signal to ask a clarifying question: it is True
    only when the message fits several topics *and* nothing in the conversation
    context broke the tie. When context did break it, `ambiguous` is False and
    `resolved_by_context` is True — `candidates` is populated either way so the
    trace shows what was considered.
    """

    topic: SyllabusTopic
    confidence: float
    matched_on: MatchKind
    matched_label: str = ""
    ambiguous: bool = False
    resolved_by_context: bool = False
    candidates: list[SyllabusTopic] = Field(default_factory=list)

    @property
    def topic_id(self) -> str:
        return self.topic.id

    @property
    def topic_name(self) -> str:
        """English name — the key the Student Model and the KL Map are stored under."""
        return self.topic.name


def load_syllabus(path: str | Path) -> Syllabus:
    """Read a syllabus YAML file, rejecting duplicate or empty topic ids.

    Raises `SyllabusError` rather than returning a report: unlike the KL Map,
    the syllabus has no human-review UI in the POC, so an unreadable file is a
    deploy-time fault.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SyllabusError(f"Cannot read syllabus {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SyllabusError(f"Syllabus {path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise SyllabusError(f"Syllabus {path} must contain a mapping at the top level.")

    try:
        syllabus = Syllabus.model_validate(data)
    except Exception as exc:  # pydantic ValidationError, plus anything malformed
        raise SyllabusError(f"Syllabus {path} is malformed: {exc}") from exc

    if not syllabus.topics:
        raise SyllabusError(f"Syllabus {path} lists no topics; the questionnaire would be empty.")

    seen: set[str] = set()
    for topic in syllabus.topics:
        if not topic.id:
            raise SyllabusError(f"Syllabus {path} has a topic with an empty id ({topic.name!r}).")
        if topic.id in seen:
            raise SyllabusError(f"Syllabus {path} declares topic id {topic.id!r} twice.")
        seen.add(topic.id)

    return syllabus


def resolve_topic(
    message: str,
    syllabus: Syllabus,
    conversation_context: ConversationContext | str | Sequence[str] | None = None,
) -> TopicMatch | None:
    """Map a student message to a syllabus topic, or None if nothing fits.

    Tiers, highest first: the message *is* a topic label (exact), the message
    *contains* one (mention), the message is a near-miss of one (fuzzy, for
    typos), the message is a fragment of one ("linear equations"). Ties within a
    tier are resolved by the conversation context when possible and flagged as
    ambiguous when not.

    When the message names no topic at all — "I still don't get it" — the topic
    under discussion is carried forward at `CONTEXT_ONLY_CONFIDENCE`, because
    every turn needs a topic for the Student Model and KL Map lookups and the
    previous one is the only defensible answer.
    """
    context = _as_context(conversation_context)
    scored = _score_topics(message, syllabus)

    if not scored:
        return _context_only_match(syllabus, context)

    best = max(score for score, _, _, _ in scored)
    if best < MIN_CONFIDENCE:
        return _context_only_match(syllabus, context)

    candidates = _narrow([entry for entry in scored if entry[0] == best])
    kind, matched_label = candidates[0][2], candidates[0][3]
    topics = [topic for _, topic, _, _ in candidates]

    if len(topics) == 1:
        return TopicMatch(
            topic=topics[0],
            confidence=best,
            matched_on=kind,
            matched_label=matched_label,
            candidates=topics,
        )

    preferred = _preferred_by_context(topics, syllabus, context)
    if preferred is not None:
        return TopicMatch(
            topic=preferred,
            confidence=best,
            matched_on=kind,
            matched_label=matched_label,
            resolved_by_context=True,
            candidates=topics,
        )

    # No context to lean on: report the first candidate in teaching order but say
    # plainly that it was a guess, so the caller asks instead of assuming.
    return TopicMatch(
        topic=topics[0],
        confidence=round(best * AMBIGUITY_PENALTY, 4),
        matched_on=kind,
        matched_label=matched_label,
        ambiguous=True,
        candidates=topics,
    )


def _score_topics(
    message: str, syllabus: Syllabus
) -> list[tuple[float, SyllabusTopic, MatchKind, str]]:
    """Best (score, topic, kind, label) per topic, sorted by score then teaching order."""
    needle = normalise_text(message)
    if not needle:
        return []

    scored: list[tuple[float, SyllabusTopic, MatchKind, str]] = []
    for topic in syllabus.topics:
        best: tuple[float, MatchKind, str] | None = None
        for label in [*topic.labels(), topic.id]:
            candidate = _score_label(needle, label)
            if candidate is None:
                continue
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            scored.append((best[0], topic, best[1], best[2]))

    scored.sort(key=lambda entry: (-entry[0], syllabus.position(entry[1].id)))
    return scored


def _score_label(needle: str, label: str) -> tuple[float, MatchKind, str] | None:
    """Score one normalised message against one label.

    Containment is plain substring, not word-boundary: Thai is written without
    spaces, so a boundary rule would silently stop matching every Thai label.
    """
    normalised = normalise_text(label)
    if not normalised:
        return None

    if needle == normalised:
        return (EXACT_CONFIDENCE, MatchKind.EXACT, label)

    if len(normalised) >= MIN_FRAGMENT_CHARS and normalised in needle:
        return (MENTION_CONFIDENCE, MatchKind.MENTION, label)

    ratio = SequenceMatcher(None, needle, normalised).ratio()
    if ratio >= FUZZY_CUTOFF:
        return (round(ratio * FUZZY_MAX_CONFIDENCE, 4), MatchKind.FUZZY, label)

    if len(needle) >= MIN_FRAGMENT_CHARS and needle in normalised:
        return (FRAGMENT_CONFIDENCE, MatchKind.FRAGMENT, label)

    return None


def _narrow(
    candidates: list[tuple[float, SyllabusTopic, MatchKind, str]],
) -> list[tuple[float, SyllabusTopic, MatchKind, str]]:
    """Drop a candidate whose matched label sits inside another candidate's label.

    A message naming "one-step linear equations" also contains the shorter label
    of any topic called "linear equations"; the longer label is the topic the
    student actually named, so this is over-matching rather than ambiguity.
    Genuine siblings — "One-Step" and "Two-Step Linear Equations" — share no
    containment and both survive.
    """
    labels = [normalise_text(entry[3]) for entry in candidates]
    keep: list[tuple[float, SyllabusTopic, MatchKind, str]] = []
    for index, entry in enumerate(candidates):
        subsumed = any(
            other != labels[index] and labels[index] in other
            for position, other in enumerate(labels)
            if position != index
        )
        if not subsumed:
            keep.append(entry)
    return keep or candidates


def _preferred_by_context(
    topics: list[SyllabusTopic], syllabus: Syllabus, context: ConversationContext
) -> SyllabusTopic | None:
    """The most recently discussed topic that is among `topics`, if any."""
    by_id = {topic.id: topic for topic in topics}
    for recent in context.recent_topics:
        resolved = syllabus.topic(recent)
        if resolved is not None and resolved.id in by_id:
            return by_id[resolved.id]
    return None


def _context_only_match(syllabus: Syllabus, context: ConversationContext) -> TopicMatch | None:
    """Carry the previous topic forward for a message that names none."""
    for recent in context.recent_topics:
        resolved = syllabus.topic(recent)
        if resolved is not None:
            return TopicMatch(
                topic=resolved,
                confidence=CONTEXT_ONLY_CONFIDENCE,
                matched_on=MatchKind.CONVERSATION_CONTEXT,
                matched_label=resolved.name,
                resolved_by_context=True,
                candidates=[resolved],
            )
    return None


def _as_context(
    value: ConversationContext | str | Sequence[str] | None,
) -> ConversationContext:
    """Accept a context object, a single topic string, or a most-recent-first list."""
    if value is None:
        return ConversationContext()
    if isinstance(value, ConversationContext):
        return value
    if isinstance(value, str):
        return ConversationContext.of(value)
    if isinstance(value, Iterable):
        return ConversationContext(recent_topics=[str(item) for item in value])
    raise TypeError(f"unsupported conversation context: {type(value).__name__}")


__all__ = [
    "AMBIGUITY_PENALTY",
    "CONTEXT_ONLY_CONFIDENCE",
    "EXACT_CONFIDENCE",
    "FRAGMENT_CONFIDENCE",
    "MENTION_CONFIDENCE",
    "MIN_ALIAS_CHARS",
    "MIN_CONFIDENCE",
    "ConversationContext",
    "MatchKind",
    "Syllabus",
    "SyllabusError",
    "SyllabusTopic",
    "TopicMatch",
    "derive_aliases",
    "load_syllabus",
    "resolve_topic",
]
