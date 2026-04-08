"""
Phase 2 Tests — Healthcare Data & Ingest
==========================================
Verifies that domain data is present and ingest works correctly.

Run:
    python -m pytest tests/test_phase2.py -v
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CHROMA_DIR = os.path.join(DATA_DIR, 'chroma_db')


def test_input_pdfs_exist():
    """At least 10 PDFs in input/."""
    pdfs = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
    assert len(pdfs) >= 10, f"Only {len(pdfs)} PDFs found, need at least 10"


def test_pdfs_are_healthcare_domain():
    """PDF filenames contain healthcare keywords."""
    healthcare_keywords = [
        'therapy', 'therapie', 'cbt', 'depression', 'anxiety', 'angst',
        'sleep', 'schlaf', 'diga', 'treatment', 'behandlung', 'patient',
        'clinical', 'klinisch', 'mental', 'health', 'gesundheit',
        'compliance', 'datenschutz', 'onboarding', 'relapse'
    ]
    pdfs = [f.lower() for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]

    matched = 0
    for pdf in pdfs:
        if any(kw in pdf for kw in healthcare_keywords):
            matched += 1

    assert matched >= 5, f"Only {matched} PDFs with healthcare keywords in filenames"


def test_testset_exists_and_valid():
    """testset.json exists and has at least 5 healthcare questions."""
    testset_path = os.path.join(DATA_DIR, 'testset.json')
    assert os.path.exists(testset_path), "data/testset.json missing"

    with open(testset_path) as f:
        testset = json.load(f)

    assert len(testset) >= 5, f"Only {len(testset)} test questions, need at least 5"

    for i, item in enumerate(testset):
        assert 'question' in item, f"Case {i}: 'question' missing"
        assert 'reference_truth' in item, f"Case {i}: 'reference_truth' missing"
        assert 'intent' in item, f"Case {i}: 'intent' missing"


def test_chromadb_has_documents():
    """After ingest: ChromaDB contains documents."""
    import chromadb

    if not os.path.exists(CHROMA_DIR):
        import pytest
        pytest.skip("ChromaDB not available — run `python src/ingest.py` first")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = chroma_client.get_collection("docuinsight")
    except Exception:
        import pytest
        pytest.skip("Collection 'docuinsight' not available — run ingest first")

    count = collection.count()
    assert count > 0, "ChromaDB collection is empty (0 documents)"
    assert count >= 50, f"Only {count} chunks — expected at least 50 for 10 PDFs"


def test_retrieval_returns_results():
    """A healthcare query returns relevant results."""
    import chromadb

    if not os.path.exists(CHROMA_DIR):
        import pytest
        pytest.skip("ChromaDB not available")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = chroma_client.get_collection("docuinsight")
    except Exception:
        import pytest
        pytest.skip("Collection not available")

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
        input=["What are common treatments for depression?"], model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(query_embeddings=[query_vec], n_results=3)

    assert len(results['documents'][0]) > 0, "No results for healthcare query"
