"""kg_reasoner: identifies which knowledge-graph nodes support teaching feedback for one student turn.

Consumes a kg-mapper-poc `_index.json`. See DESIGN.md for the full write-up.
"""

from kg_reasoner.models import CandidateNode, KnowledgeGuidance, StudentContext
from kg_reasoner.pipeline import reason

__all__ = ["CandidateNode", "KnowledgeGuidance", "StudentContext", "reason"]
