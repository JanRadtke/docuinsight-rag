"""
Phase 3 Tests — Cross-Lingual RRF Retrieval
============================================
Tests _multilingual_hybrid_search() and retrieve_knowledge(multilingual_queries=...).

Prerequisite: python src/ingest.py has been executed (ChromaDB populated).

Run:
    python -m pytest tests/test_crosslingual_retrieval.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ROOT = os.path.join(os.path.dirname(__file__), '..')
CHROMA_DIR = os.path.join(ROOT, 'data', 'chroma_db')


def _make_retriever():
    from retriever import Retriever, get_chroma_collection
    from llm_provider import get_llm_client
    return Retriever(get_chroma_collection(), get_llm_client())


def test_multilingual_search_returns_results():
    """_multilingual_hybrid_search returns a non-empty list."""
    r = _make_retriever()
    queries = {
        "en": "cognitive behavioural therapy CBT depression treatment",
        "de": "kognitive Verhaltenstherapie KVT Depression Behandlung"
    }
    results = r._multilingual_hybrid_search(queries, top_k=10)
    print(f"Got {len(results)} results")
    for doc_id, content, meta in results[:3]:
        print(f"  [{meta.get('language','?')}] {meta.get('source_file','?')[:35]} p.{meta.get('page_number','?')}")
    assert len(results) > 0, "No results for multilingual search"
    print("✅ _multilingual_hybrid_search returns results")


def test_multilingual_finds_at_least_as_many_as_mono():
    """Multilingual search finds at least as many docs as monolingual search."""
    r = _make_retriever()
    mono = r._hybrid_search("cognitive behavioural therapy CBT", top_k=10)
    multi = r._multilingual_hybrid_search(
        {"en": "CBT therapy depression", "de": "KVT Therapie Depression"},
        top_k=10
    )
    mono_files = set(m["source_file"] for _, _, m in mono)
    multi_files = set(m["source_file"] for _, _, m in multi)
    print(f"Mono: {len(mono_files)} files | Multi: {len(multi_files)} files")
    print(f"New via multi: {multi_files - mono_files}")
    assert len(multi_files) >= len(mono_files), (
        f"Multi ({len(multi_files)}) should be >= Mono ({len(mono_files)})"
    )
    print("✅ Multilingual finds at least as many docs")


def test_retrieve_knowledge_multilingual_queries_none_uses_mono():
    """retrieve_knowledge with multilingual_queries=None uses _hybrid_search."""
    r = _make_retriever()
    called = {"hybrid": False, "multi": False}
    original_hybrid = r._hybrid_search
    original_multi = r._multilingual_hybrid_search

    def mock_hybrid(q, top_k=10):
        called["hybrid"] = True
        return original_hybrid(q, top_k=top_k)

    def mock_multi(queries, top_k=10):
        called["multi"] = True
        return original_multi(queries, top_k=top_k)

    r._hybrid_search = mock_hybrid
    r._multilingual_hybrid_search = mock_multi

    r.retrieve_knowledge("What is CBT?", multilingual_queries=None)
    assert called["hybrid"], "_hybrid_search should have been called"
    assert not called["multi"], "_multilingual_hybrid_search should NOT have been called"
    print("✅ multilingual_queries=None → _hybrid_search (backward compatibility)")


def test_retrieve_knowledge_multilingual_queries_routes_correctly():
    """retrieve_knowledge with >1 language in multilingual_queries uses _multilingual_hybrid_search."""
    r = _make_retriever()
    called = {"multi": False}
    original_multi = r._multilingual_hybrid_search

    def mock_multi(queries, top_k=10):
        called["multi"] = True
        return original_multi(queries, top_k=top_k)

    r._multilingual_hybrid_search = mock_multi

    r.retrieve_knowledge(
        "Was ist CBT?",
        multilingual_queries={"en": "CBT therapy", "de": "KVT Therapie"}
    )
    assert called["multi"], "_multilingual_hybrid_search should have been called"
    print("✅ multilingual_queries with 2 languages → _multilingual_hybrid_search")
