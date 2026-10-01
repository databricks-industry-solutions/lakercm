"""The per-tier model-service endpoints read "none" as unset.

The bundle cannot pass an empty env var (the Apps deployment API rejects an
entry with no value: "Must specify environment variable source using either
`value` or `valueFrom`"), so an unconfigured tier arrives as "none".
"""

from __future__ import annotations

import os
import sys

_AGENT_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_APP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_APP_DIR)

_TIER_ENVS = ("LLM_ENDPOINT_LOW", "LLM_ENDPOINT_MED", "LLM_ENDPOINT_HIGH")


def test_none_leaves_every_tier_on_the_blend(monkeypatch):
    from config import Settings

    for name in _TIER_ENVS:
        monkeypatch.setenv(name, "none")
    settings = Settings()
    assert (
        settings.llm_endpoint_low,
        settings.llm_endpoint_med,
        settings.llm_endpoint_high,
    ) == ("", "", "")


def test_the_sentinel_is_case_and_space_insensitive(monkeypatch):
    from config import Settings

    monkeypatch.setenv("LLM_ENDPOINT_HIGH", " None ")
    assert Settings().llm_endpoint_high == ""


def test_a_configured_service_is_kept(monkeypatch):
    from config import Settings

    for name in _TIER_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_ENDPOINT_MED", "cat.sch.lakercm-agent-llm-med")
    settings = Settings()
    assert settings.llm_endpoint_med == "cat.sch.lakercm-agent-llm-med"
    assert settings.llm_endpoint_low == ""
