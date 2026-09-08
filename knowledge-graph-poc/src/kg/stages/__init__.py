"""Model-backed and deterministic pipeline stages (DESIGN §6): atomize (S3+S3b),
consolidate (S4), edges (S5), dedup (S6). Every model call goes through the injected
``call_stage`` (``kg.llm.call_stage`` in production, a scripted double in tests)."""
