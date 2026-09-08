"""WU2 — adapter protocol and wire types (DESIGN §7.1, §4.3, §20 `kg.llm.adapters.base`).

R21: one provider boundary, three adapters behind it. R6: images travel to every
adapter as `ImageInput{media_type, data_b64}` (DESIGN §4.3).
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest

from kg.llm.adapters.base import Adapter, ImageInput, Msg, RawCompletion, Usage

from wu2_fakes import PNG_B64, PNG_BYTES

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- ImageInput


def test_R6_image_input_carries_media_type_and_base64_payload():
    img = ImageInput(media_type="image/png", data_b64=PNG_B64)
    assert img.media_type == "image/png"
    assert img.data_b64 == PNG_B64
    assert isinstance(img.data_b64, str)


def test_R6_image_input_from_bytes_base64_encodes_exactly():
    img = ImageInput.from_bytes(PNG_BYTES, "image/png")
    assert img.media_type == "image/png"
    assert base64.b64decode(img.data_b64) == PNG_BYTES


@pytest.mark.parametrize("bad", ["text/plain", "application/pdf", "", "png"])
def test_R6_image_input_rejects_non_image_media_type(bad):
    with pytest.raises(ValueError):
        ImageInput.from_bytes(PNG_BYTES, bad)


def test_R6_image_input_from_empty_bytes_is_refused():
    with pytest.raises(ValueError):
        ImageInput.from_bytes(b"", "image/png")


# ---------------------------------------------------------------------- Msg


def test_R21_msg_defaults_to_no_images():
    m = Msg(role="user", text="hello")
    assert m.role == "user"
    assert m.text == "hello"
    assert list(m.images) == []


def test_R21_msg_carries_images_in_order():
    a = ImageInput(media_type="image/png", data_b64=PNG_B64)
    b = ImageInput(media_type="image/jpeg", data_b64=PNG_B64)
    m = Msg(role="user", text="two", images=[a, b])
    assert list(m.images) == [a, b]


def test_R21_msg_role_is_user_or_assistant_only():
    with pytest.raises(ValueError):
        Msg(role="system", text="not allowed here; system is a separate argument")


# -------------------------------------------------------------------- Usage


def test_R15_usage_defaults_cached_and_reasoning_to_zero():
    u = Usage(input_tokens=10, output_tokens=5)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens, u.reasoning_tokens) == (10, 5, 0, 0)


def test_R15_usage_rejects_negative_counts():
    with pytest.raises(ValueError):
        Usage(input_tokens=-1, output_tokens=0)


# ------------------------------------------------------------ RawCompletion


def test_R21_raw_completion_fields_and_defaults():
    rc = RawCompletion(text_or_obj='{"a": 1}', usage=Usage(1, 2), model_requested="m")
    assert rc.text_or_obj == '{"a": 1}'
    assert rc.model_requested == "m"
    assert rc.model_served is None
    assert rc.cost_usd is None  # "unknown", never 0.0 by default (D28)
    assert rc.stop_reason is None
    assert dict(rc.provider_meta) == {}
    assert list(rc.events) == []


def test_R21_raw_completion_accepts_dict_output_for_native_structured_output():
    rc = RawCompletion(text_or_obj={"a": 1}, usage=Usage(1, 2), model_requested="m", model_served="m-2", cost_usd=None)
    assert rc.text_or_obj == {"a": 1}
    assert rc.model_served == "m-2"


# ------------------------------------------------------------ Adapter proto


def test_R21_adapter_protocol_is_runtime_checkable_by_duck_typing():
    class Duck:
        provider = "duck"
        max_repairs = None

        def complete(self, system, messages, schema_json, model, max_tokens):
            return None

    class NotADuck:
        provider = "x"

    assert isinstance(Duck(), Adapter)
    assert not isinstance(NotADuck(), Adapter)


def test_R21_conftest_fake_adapter_satisfies_protocol(fake_adapter):
    assert isinstance(fake_adapter, Adapter)


# ---------------------------------------------------- SDK import isolation


def test_R21_boundary_modules_do_not_import_provider_sdks():
    """DESIGN §7.1: only files under src/kg/llm/adapters/ import a provider SDK.

    Run in a fresh interpreter because conftest's guard imports the SDKs into this one.
    """
    code = (
        "import sys, kg.llm, kg.llm.ledger, kg.llm.schema_profile, kg.llm.adapters.base, kg.config;"
        "bad=[m for m in ('claude_agent_sdk','openai','anthropic') if m in sys.modules];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, f"provider SDK imported outside adapters: {proc.stdout} {proc.stderr}"
