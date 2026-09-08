"""Demo CLI: `python -m kg_reasoner.cli --index path/to/_index.json --topic "..." --intent explain ...`"""

from __future__ import annotations

import argparse
import json
import sys

from kg_reasoner.graph_index import load_index
from kg_reasoner.models import StudentContext
from kg_reasoner.pipeline import reason


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the knowledge-graph reasoner against one student turn.")
    p.add_argument("--index", required=True, help="Path to data/graph/_index.json")
    p.add_argument("--topic", required=True, help="Topic text (from RAG / topic mapper)")
    p.add_argument("--intent", required=True, choices=["explain", "clarify", "solve", "hint", "check_answer", "practice_quiz", "summarize_review"])
    p.add_argument("--learner-state", default="normal", choices=["normal", "confused"])
    p.add_argument("--student-level", default="beginner", choices=["beginner", "intermediate", "advanced"])
    p.add_argument("--special-handling", default="none", choices=["none", "homework", "assessment"])
    p.add_argument("--exclude", nargs="*", default=[], help="Node ids to exclude (already used this turn)")
    args = p.parse_args(argv)

    index = load_index(args.index)
    ctx = StudentContext(
        topic_query=args.topic,
        intent=args.intent,
        learner_state=args.learner_state,
        student_level=args.student_level,
        special_handling=args.special_handling,
        exclude_node_ids=frozenset(args.exclude),
    )
    guidance = reason(ctx, index)
    print(json.dumps(guidance.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
