"""Step 2: relationship-filtered, depth-limited traversal from the anchor.

This is graph traversal, not reasoning (Part 1 PDF, p.36: "Important: Don't
confuse traversal with reasoning"). It only answers "what's connected, by
which relation, how far away" — `rules.py` is what decides what that means
pedagogically.
"""

from __future__ import annotations

from dataclasses import dataclass

from kg_reasoner.graph_index import GraphIndex, Neighbor


@dataclass(frozen=True)
class Reached:
    node_id: str
    distance: int
    relation: str
    direction: str  # "out" | "in" | "sym" — see Neighbor.direction
    relevance: int | None


def bfs(
    index: GraphIndex,
    start_id: str,
    *,
    allowed_types: set[str],
    max_depth: int = 2,
) -> dict[str, Reached]:
    """Breadth-first search from `start_id`, following only `allowed_types` edges.

    Matches the PDF's recommended default (BFS, depth 1-2, relation-filtered)
    rather than following every edge blindly. Returns the nearest path found
    to each reachable node (first-seen-wins, since BFS visits in distance order).
    """
    seen: dict[str, Reached] = {}
    frontier: list[str] = [start_id]
    distance = 0
    while frontier and distance < max_depth:
        distance += 1
        next_frontier: list[str] = []
        for node_id in frontier:
            for nb in index.neighbors(node_id, types=allowed_types):
                if nb.node_id in seen or nb.node_id == start_id:
                    continue
                seen[nb.node_id] = Reached(nb.node_id, distance, nb.type, nb.direction, nb.relevance)
                next_frontier.append(nb.node_id)
        frontier = next_frontier
    return seen


def direct_neighbors(index: GraphIndex, node_id: str, *, types: set[str]) -> list[Neighbor]:
    """Distance-1 neighbors only, filtered by type — a thin wrapper for single-hop rules."""
    return index.neighbors(node_id, types=types)


__all__ = ["Reached", "bfs", "direct_neighbors"]
