"""
Phase 0.5 Tests — Baseline Smoke Test
=======================================
Verifies that the system works end-to-end with PDFs.
Prerequisite: python src/ingest.py has been executed.

Run:
    python -m pytest tests/test_phase0.py -v
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ROOT = os.path.join(os.path.dirname(__file__), '..')
CHROMA_DIR = os.path.join(ROOT, 'data', 'chroma_db')
INPUT_DIR = os.path.join(ROOT, 'input')


def test_input_has_pdfs():
    """At least 1 PDF exists in input/."""
    pdfs = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
    assert len(pdfs) >= 1, (
        "No PDFs in input/. Please copy your own PDFs into input/."
    )


def test_chromadb_populated():
    """ChromaDB exists and contains chunks after ingest."""
    assert os.path.exists(CHROMA_DIR), (
        "ChromaDB not found. Run `python src/ingest.py` first."
    )
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("docuinsight")
    except Exception:
        raise AssertionError("Collection 'docuinsight' missing. Run `python src/ingest.py` first.")

    count = collection.count()
    assert count > 100, f"Only {count} chunks — expected > 100 after ingest"


def test_testset_schema():
    """testset.json has the correct schema for evaluate.py."""
    testset_path = os.path.join(ROOT, 'data', 'testset.json')
    assert os.path.exists(testset_path), "data/testset.json missing"

    with open(testset_path) as f:
        cases = json.load(f)

    assert len(cases) >= 1
    for i, case in enumerate(cases):
        assert 'question' in case, f"Case {i}: 'question' missing"
        assert 'reference_truth' in case, f"Case {i}: 'reference_truth' missing"
        assert 'intent' in case, f"Case {i}: 'intent' missing"


def test_retriever_returns_results():
    """Retriever finds chunks for a simple query."""
    if not os.path.exists(CHROMA_DIR):
        import pytest
        pytest.skip("ChromaDB not available")

    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("docuinsight")
    except Exception:
        import pytest
        pytest.skip("Collection missing")

    if collection.count() == 0:
        import pytest
        pytest.skip("Collection empty")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PRIVATE_OPENAI_KEY")
    if not api_key:
        peeked = collection.peek(1)
        assert len(peeked['documents']) > 0, "Collection has no documents"
        import pytest
        pytest.skip("No API key — embedding query skipped, peek() OK")

    from openai import OpenAI
    embed_client = OpenAI(api_key=api_key)
    query_vec = embed_client.embeddings.create(
        input=["Senolytika Apoptose BCL-2"], model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(query_embeddings=[query_vec], n_results=3)
    assert len(results['documents'][0]) > 0, "No results for senolytic query"

    all_text = " ".join(results['documents'][0]).lower()
    assert any(kw in all_text for kw in ['senol', 'apoptose', 'apoptosis', 'bcl', 'aging', 'alter']), \
        "Retrieved chunks don't seem to match the topic"


def test_guardrail_allows_science():
    """Guardrail allows scientific questions."""
    import pytest
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("PRIVATE_OPENAI_KEY"):
        pytest.skip("No API key — guardrail test skipped")

    from guardrail import InputGuardrail
    guardrail = InputGuardrail()

    is_blocked, reason = guardrail.check("Was sind die Hallmarks of Aging?")
    assert not is_blocked, f"Scientific question was blocked: {reason}"


def test_guardrail_blocks_offtopic():
    """Guardrail blocks off-topic requests."""
    import pytest
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("PRIVATE_OPENAI_KEY"):
        pytest.skip("No API key")

    from guardrail import InputGuardrail
    guardrail = InputGuardrail()

    is_blocked, reason = guardrail.check("Welches Auto soll ich kaufen?")
    assert is_blocked, "Off-topic was not blocked"
