"""Shared Azure AI Foundry / Azure OpenAI client code for the three sibling
projects in `socratic_tutor/` — kg_reasoner, knowledge-graph-poc, and
socratic-tutor-poc. See the top-level README.md for what to use where."""

from azure_foundry_adapter.client import AzureFoundrySettings, complete_async, complete_sync

__all__ = ["AzureFoundrySettings", "complete_async", "complete_sync"]
