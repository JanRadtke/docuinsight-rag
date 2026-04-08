"""
Phase 5 Tests — Writer Output Language Enforcement
====================================================
Verifies that draft_answer() responds in the correct language based on
the target_language parameter.

Run:
    python -m pytest tests/test_writer_language.py -v -s
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ENGLISH_FACTS = {
    "facts": [
        {"source_id": "1", "fact": "Cognitive Behavioural Therapy (CBT) is a structured, time-limited therapy for depression."},
        {"source_id": "2", "fact": "CBT uses cognitive restructuring to challenge negative thought patterns."},
        {"source_id": "3", "fact": "Behavioural activation is a core technique of CBT."},
    ]
}


def _make_agent():
    from advanced_agent import AdvancedAgent
    return AdvancedAgent()


def test_writer_german_target_language():
    """draft_answer with target_language='de' and English facts must produce German output."""
    from langdetect import detect
    agent = _make_agent()
    answer = agent.draft_answer(
        "Was ist kognitive Verhaltenstherapie?",
        ENGLISH_FACTS,
        target_language="de"
    )
    print(f"\nDE answer (first 300 chars):\n{answer[:300]}")
    detected = detect(answer)
    assert detected == "de", f"Expected German ('de'), langdetect returned '{detected}'. Answer: {answer[:200]}"
    print("✅ target_language='de' → German output confirmed")


def test_writer_english_target_language():
    """draft_answer with target_language='en' must produce English output."""
    from langdetect import detect
    agent = _make_agent()
    answer = agent.draft_answer(
        "What is CBT?",
        ENGLISH_FACTS,
        target_language="en"
    )
    print(f"\nEN answer (first 300 chars):\n{answer[:300]}")
    detected = detect(answer)
    assert detected == "en", f"Expected English ('en'), langdetect returned '{detected}'. Answer: {answer[:200]}"
    print("✅ target_language='en' → English output confirmed")


def test_writer_no_target_language_backward_compatible():
    """draft_answer without target_language must still work (backward compatibility)."""
    agent = _make_agent()
    answer = agent.draft_answer(
        "What is CBT?",
        ENGLISH_FACTS
        # no target_language — should use old default behaviour
    )
    print(f"\nNo target_language answer (first 200 chars):\n{answer[:200]}")
    assert len(answer) > 10, "Expected non-empty answer with no target_language"
    print("✅ No target_language → backward compatible (returns answer)")
