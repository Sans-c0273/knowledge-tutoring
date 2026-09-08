#!/usr/bin/env python3
"""E1 — Intent-detection accuracy (Tech Spec §9).

Gate: **≥85% intent accuracy, ≥95% special-handling detection.** The second
number is the strict one because a missed `homework` or `assessment` label is a
policy breach, not a classification error.

    uv run python eval/run_e1_intent.py
    uv run python eval/run_e1_intent.py --limit 5 --out data/e1-report.json

Needs live Claude credentials (`ant auth login`). Runs sequentially on purpose:
the subscription's rate limits make a slow, ordered run the reliable one
(Addendum §1, "known trade-off").

The seed set is 40 items. §9 wants ≥300 for the real gate — Phase C (R13–R16)
extends it. A run against 40 items reports a measurement, not a passed gate, and
the summary says so.

**Keep the classifier's few-shot examples and this dataset disjoint.** None of the
worked examples in `pedagogy.intent.SYSTEM_PROMPT` is taken from
`e1-intent-labeled.jsonl`, and none should be when either side grows. Reusing an
eval item as a few-shot turns the gate into a memorisation check: the score goes
up, the measurement stops meaning anything, and nothing fails to warn you.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from socratic_tutor.config import PROJECT_ROOT, Settings
from socratic_tutor.pedagogy.intent import detect_intent
from socratic_tutor.providers import LLMProvider, get_provider
from socratic_tutor.providers.base import ProviderError

DEFAULT_DATASET = PROJECT_ROOT / "content" / "seed" / "eval" / "e1-intent-labeled.jsonl"

INTENT_GATE = 0.85
SPECIAL_HANDLING_GATE = 0.95
GATE_SAMPLE_SIZE = 300


@dataclass
class Tally:
    """Running counts for one E1 run."""

    total: int = 0
    intent_hits: int = 0
    learner_state_hits: int = 0
    special_handling_hits: int = 0
    errors: int = 0
    fallbacks: int = 0
    per_intent: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def rate(self, hits: int) -> float:
        return hits / self.total if self.total else 0.0


# ------------------------------------------------- provenance for the result


class UsageRecorder(LLMProvider):
    """Wraps a provider and records what each call actually cost on the wire.

    Exists so a result file can state the *observed* prompt size rather than an
    estimate. That is the number that makes gate results comparable across
    providers: the same run costs ~600 input tokens on a direct HTTP path and
    ~1,500 through the Agent SDK, which wraps our prompt in its own scaffolding
    and runs a two-turn agent loop. A gate measured on one prompt cannot be cited
    to justify a deployment on the other, so the file has to say which it was.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.name = inner.name
        self.input_tokens: list[int] = []
        self.output_tokens: list[int] = []
        self.models: set[str] = set()

    async def complete_structured(self, **kwargs: Any) -> Any:
        result = await self.inner.complete_structured(**kwargs)
        self.input_tokens.append(result.usage.input_tokens)
        self.output_tokens.append(result.usage.output_tokens)
        self.models.add(result.model)
        return result

    def stream_text(self, **kwargs: Any) -> Any:
        return self.inner.stream_text(**kwargs)

    async def aclose(self) -> None:
        await self.inner.aclose()

    def provenance(self, settings: Settings, role: str) -> dict[str, Any]:
        """Everything needed to read this result without guessing how it was produced."""
        return {
            "provider": settings.provider_for(role),
            "model_requested": settings.model_for(role),
            "models_served": sorted(self.models),
            "calls": len(self.input_tokens),
            "prompt_tokens": _token_stats(self.input_tokens),
            "output_tokens": _token_stats(self.output_tokens),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": (
                "Gate results do not transfer between providers: each path sends a "
                "different prompt. Re-run on the path a pilot would actually use "
                "before citing these numbers for a deployment decision."
            ),
        }


def _token_stats(values: list[int]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 1),
        "min": min(values),
        "max": max(values),
    }


def load_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                items.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path.name}:{line_number}: malformed JSON: {exc}") from exc
    return items[:limit] if limit else items


async def run(
    items: list[dict[str, Any]], settings: Settings, verbose: bool
) -> tuple[Tally, UsageRecorder]:
    provider = UsageRecorder(get_provider("intent", settings))
    tally = Tally()
    try:
        for item in items:
            tally.total += 1
            expected_intent = item["intent"]
            try:
                result = await detect_intent(
                    item["message"], item.get("context", []), provider, settings=settings
                )
            except ProviderError as exc:
                tally.errors += 1
                tally.failures.append({"id": item["id"], "error": str(exc)})
                if verbose:
                    print(f"  {item['id']}  ERROR  {exc}")
                continue

            got_intent = result.intent.value
            got_state = result.learner_state.value
            got_special = result.special_handling.type.value

            seen, hit = tally.per_intent[expected_intent]
            intent_ok = got_intent == expected_intent
            tally.per_intent[expected_intent] = [seen + 1, hit + int(intent_ok)]
            tally.confusion[(expected_intent, got_intent)] += 1
            tally.intent_hits += int(intent_ok)
            tally.learner_state_hits += int(got_state == item["learner_state"])
            tally.special_handling_hits += int(got_special == item["special_handling"])
            tally.fallbacks += int(result.low_confidence_fallback)

            if not intent_ok or got_special != item["special_handling"]:
                tally.failures.append(
                    {
                        "id": item["id"],
                        "lang": item.get("lang"),
                        "message": item["message"],
                        "expected": {
                            "intent": expected_intent,
                            "learner_state": item["learner_state"],
                            "special_handling": item["special_handling"],
                        },
                        "got": {
                            "intent": got_intent,
                            "learner_state": got_state,
                            "special_handling": got_special,
                            "confidence": result.confidence,
                            "low_confidence_fallback": result.low_confidence_fallback,
                        },
                    }
                )
            if verbose:
                mark = "ok  " if intent_ok else "MISS"
                print(
                    f"  {item['id']}  {mark}  expected={expected_intent:<17}"
                    f"got={got_intent:<17}conf={result.confidence:.2f}"
                )
    finally:
        await provider.aclose()
    return tally, provider


def report(tally: Tally, dataset: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    graded = tally.total - tally.errors
    summary = {
        "dataset": str(dataset),
        "provenance": provenance,
        "items": tally.total,
        "graded": graded,
        "errors": tally.errors,
        "intent_accuracy": tally.rate(tally.intent_hits),
        "learner_state_accuracy": tally.rate(tally.learner_state_hits),
        "special_handling_accuracy": tally.rate(tally.special_handling_hits),
        "low_confidence_fallbacks": tally.fallbacks,
        "per_intent": {
            name: {"n": seen, "correct": hit, "accuracy": hit / seen if seen else 0.0}
            for name, (seen, hit) in sorted(tally.per_intent.items())
        },
        "confusion_matrix": {f"{a} -> {b}": n for (a, b), n in sorted(tally.confusion.items())},
        "gates": {
            "intent": {
                "target": INTENT_GATE,
                "passed": tally.rate(tally.intent_hits) >= INTENT_GATE,
            },
            "special_handling": {
                "target": SPECIAL_HANDLING_GATE,
                "passed": tally.rate(tally.special_handling_hits) >= SPECIAL_HANDLING_GATE,
            },
            "sample_size_sufficient": tally.total >= GATE_SAMPLE_SIZE,
        },
        "failures": tally.failures,
    }
    return summary


def print_report(summary: dict[str, Any]) -> None:
    print("\nE1 — Intent detection")
    print("=" * 62)
    print_provenance(summary["provenance"])
    print(f"items graded          {summary['graded']} / {summary['items']}")
    if summary["errors"]:
        print(f"provider errors       {summary['errors']}")
    print(
        f"intent accuracy       {summary['intent_accuracy']:.1%}   "
        f"(gate ≥{INTENT_GATE:.0%})  "
        f"{'PASS' if summary['gates']['intent']['passed'] else 'FAIL'}"
    )
    print(
        f"special handling      {summary['special_handling_accuracy']:.1%}   "
        f"(gate ≥{SPECIAL_HANDLING_GATE:.0%})  "
        f"{'PASS' if summary['gates']['special_handling']['passed'] else 'FAIL'}"
    )
    print(f"learner state         {summary['learner_state_accuracy']:.1%}   (no gate)")
    print(f"low-confidence falls  {summary['low_confidence_fallbacks']}")

    print("\nper intent")
    for name, row in summary["per_intent"].items():
        print(f"  {name:<18} {row['correct']:>3}/{row['n']:<3} {row['accuracy']:>6.1%}")

    confusions = {k: v for k, v in summary["confusion_matrix"].items() if not _is_diagonal(k)}
    if confusions:
        print("\nconfusions (expected -> got)")
        for key, count in sorted(confusions.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<44} {count}")

    if not summary["gates"]["sample_size_sufficient"]:
        print(
            f"\nNOTE: {summary['items']} items. Tech Spec §9 requires ≥{GATE_SAMPLE_SIZE} for the"
            f" E1 gate,\n      so this is a measurement, not a passed gate. Phase C extends the set."
        )


def print_provenance(prov: dict[str, Any]) -> None:
    """Print how this run was produced, above the numbers it produced.

    Above, not below, because the numbers are what get copied into a document
    and the provider is what decides whether copying them is legitimate.
    """
    prompt = prov.get("prompt_tokens")
    print(f"provider              {prov['provider']}")
    print(f"model                 {prov['model_requested']}")
    if prov.get("models_served") and prov["models_served"] != [prov["model_requested"]]:
        print(f"served                {', '.join(prov['models_served'])}")
    if prompt:
        print(
            f"prompt tokens/call    mean {prompt['mean']:.0f} "
            f"(min {prompt['min']}, max {prompt['max']}) over {prov['calls']} calls"
        )
    print(f"run at                {prov['run_at']}")
    print()


def _is_diagonal(key: str) -> bool:
    expected, _, got = key.partition(" -> ")
    return expected == got


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="grade only the first N items")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--model", default=None, help="override the intent model ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="print each item as it runs")
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    settings = Settings(intent_model=args.model) if args.model else Settings()
    items = load_dataset(args.dataset, args.limit)
    if not items:
        print("dataset is empty", file=sys.stderr)
        return 2

    print(f"E1: grading {len(items)} items with {settings.model_for('intent')} (sequential)…")
    try:
        tally, recorder = asyncio.run(run(items, settings, args.verbose))
    except ProviderError as exc:
        print(f"\nprovider unavailable: {exc}", file=sys.stderr)
        print(
            "E1 needs live Claude credentials. Run `ant auth login` once on this machine,\n"
            "or set POC_INTENT_PROVIDER=openai_compat with POC_OPENAI_COMPAT_API_KEY.",
            file=sys.stderr,
        )
        return 3

    summary = report(tally, args.dataset, recorder.provenance(settings, "intent"))
    print_report(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.out}")

    gates = summary["gates"]
    return 0 if gates["intent"]["passed"] and gates["special_handling"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
