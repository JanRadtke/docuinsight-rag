"""
Phase 4 End-to-End Tests — Cross-Lingual LangGraph Pipeline
============================================================
Tests the full pipeline: planner detects language → multilingual retrieval →
reader extracts facts → writer responds in user language.

Prerequisites: python src/ingest.py must have been run (ChromaDB populated).

Run:
    python -m pytest tests/test_e2e_crosslingual.py -v -s
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


def _setup():
    from retriever import Retriever, get_chroma_collection
    from llm_provider import get_llm_client
    from advanced_agent import AdvancedAgent
    from agent_graph import run_agent
    retriever = Retriever(get_chroma_collection(), get_llm_client())
    agent = AdvancedAgent()
    return retriever, agent, run_agent


def test_german_question_returns_facts():
    """DE query against English corpus must yield > 0 facts (core bug fix)."""
    retriever, agent, run_agent = _setup()
    result = run_agent(
        "Was ist Cognitive Behavioural Therapy (CBT)?",
        retriever=retriever,
        agent=agent,
        intent="SEARCH"
    )
    facts = result.get("facts", [])
    print(f"\nDE question → {len(facts)} facts")
    if facts:
        print(f"  First fact: {facts[0].get('fact', '')[:100]}")
    print(f"  Answer (first 200 chars): {result.get('final_answer', '')[:200]}")
    assert len(facts) > 0, f"FAIL: German question returned 0 facts. Answer: {result.get('final_answer', '')[:300]}"
    print("✅ German question returns > 0 facts")


def test_english_question_no_regression():
    """EN query must still yield >= 10 facts (regression guard)."""
    retriever, agent, run_agent = _setup()
    result = run_agent(
        "What is CBT and which techniques does it use?",
        retriever=retriever,
        agent=agent,
        intent="SEARCH"
    )
    facts = result.get("facts", [])
    print(f"\nEN question → {len(facts)} facts")
    assert len(facts) >= 10, f"REGRESSION: English question returned only {len(facts)} facts (expected >= 10)"
    print("✅ English question still returns >= 10 facts (no regression)")
