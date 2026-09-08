"""Provider adapters (DESIGN §7.3–§7.5).

Only modules in this package may import a provider SDK, and each imports its
own lazily enough that `kg.llm.adapters.base` stays SDK-free. Nothing is
imported here so that importing the package never pulls in an SDK.
"""
