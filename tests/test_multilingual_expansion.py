"""
Phase 2 Tests — Multilingual Query Expansion
=============================================
Tests expand_query_multilingual() in advanced_agent.py.

Run:
    python -m pytest tests/test_multilingual_expansion.py -v
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _make_agent():
    from advanced_agent import AdvancedAgent
    return AdvancedAgent()


def test_multilingual_two_languages_returns_both_keys():
    """2 languages → dict with both keys and non-empty values."""
    agent = _make_agent()
    result = agent.expand_query_multilingual(
        "Was ist Verhaltenstherapie?",
        ["en", "de"],
        intent="SEARCH"
    )
    print("Result:", result)
    assert isinstance(result, dict), "Did not return a dict"
    assert "en" in result, "Key 'en' missing"
    assert "de" in result, "Key 'de' missing"
    assert result["en"].strip(), "EN query is empty"
    assert result["de"].strip(), "DE query is empty"
    print("✅ Both languages present")


def test_multilingual_single_language_no_extra_llm_call():
    """1 language → fast path, optimize_query() is called, no extra LLM call."""
    from advanced_agent import AdvancedAgent

    agent = AdvancedAgent()

    call_count = {"n": 0}
    original_optimize = agent.optimize_query

    def counting_optimize(q, intent="FACTS"):
        call_count["n"] += 1
        return original_optimize(q, intent)

    agent.optimize_query = counting_optimize

    result = agent.expand_query_multilingual("What is CBT?", ["en"])
    print("Result:", result)
    assert len(result) == 1, f"Expected 1 key, got: {list(result.keys())}"
    assert "en" in result, "Key 'en' missing"
    assert result["en"].strip(), "EN query is empty"
    assert call_count["n"] == 1, f"optimize_query should be called once, was called {call_count['n']}x"
    print("✅ Single-language fast path correct")


def test_multilingual_empty_languages_fallback():
    """Empty language list → fallback to {'en': question}."""
    agent = _make_agent()
    result = agent.expand_query_multilingual("What is CBT?", [])
    assert result == {"en": "What is CBT?"}, f"Expected fallback, got: {result}"
    print("✅ Empty language list → fallback correct")


def test_multilingual_error_fallback():
    """LLM error → fallback to {'en': question}, no crash."""
    from advanced_agent import AdvancedAgent
    agent = AdvancedAgent()

    # Simulate LLM error
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = Exception("API error")

    result = agent.expand_query_multilingual("Was ist CBT?", ["en", "de"])
    assert "en" in result, "Fallback should contain 'en'"
    print("✅ LLM error → clean fallback")
