"""Transport-level errors shared by every adapter in this package.

Deliberately provider-agnostic (no dependency on any of the 3 sibling
projects' own exception hierarchies) — each per-project adapter module is
responsible for catching these and re-raising as whatever exception type its
host project expects (see the `_raise_as_*` helpers at the bottom of each
adapter file).
"""

from __future__ import annotations


class AzureFoundryError(Exception):
    """Base class for every error raised from this package."""


class AzureFoundryConfigError(AzureFoundryError):
    """Required settings are missing (endpoint, deployment, or a credential)."""


class AzureFoundryConnectionError(AzureFoundryError):
    """The request never reached Azure — DNS, TLS, timeout."""


class AzureFoundryRateLimited(AzureFoundryError):
    """HTTP 429. Azure's `Retry-After` header value, in seconds, is on `retry_after` when present."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AzureFoundryModelNotFound(AzureFoundryError):
    """HTTP 404 — usually a wrong deployment name, not a wrong model name (Azure addresses
    deployments, not base model ids)."""


class AzureFoundryResponseError(AzureFoundryError):
    """Azure answered, but the payload was not what the caller's contract requires
    (malformed JSON, missing choices, schema-validation failure downstream)."""


class AzureFoundryStatusError(AzureFoundryError):
    """Any other non-2xx HTTP status."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


__all__ = [
    "AzureFoundryConfigError",
    "AzureFoundryConnectionError",
    "AzureFoundryError",
    "AzureFoundryModelNotFound",
    "AzureFoundryRateLimited",
    "AzureFoundryResponseError",
    "AzureFoundryStatusError",
]
