# DocuInsight RAG System — Constitution & Principles

## Executive Summary

A set of binding architectural principles for the DocuInsight Python RAG system, ensuring:
- **Linguistic Agnosticism**: User questions & documents in any language combination
- **Internal Lingua Franca**: English for all reasoning (optimal LLM performance)
- **Cost Consciousness**: Minimal overhead; mini-models for support, main LLM only for critical paths
- **Graceful Degradation**: No penalty for monolingual scenarios
- **Testability**: Each phase independently testable and committable
- **Backward Compatibility**: Zero breaking changes to existing pipeline

---

## I. Linguistic Agnosticism Principle

### 1.1 Definition
The system processes questions in **any user language** against documents in **any combination of document languages** and returns answers in the user's original language, without requiring explicit language configuration.

### 1.2 Scope
- **User Input**: Hebrew, Chinese, German, English, Japanese, Arabic, etc. (any language supported by langdetect)
- **Document Base**: Mixed-language repositories (e.g., 30% English, 40% German, 30% Chinese)
- **Output**: User's original language (detected + preserved)

### 1.3 Non-Goals
- Perfect translation (not our job; preserve structure & terminology)
- Language family optimization (treat all languages equally)
- Support for undocumented or synthetic languages
- Real-time language addition without re-ingestion

### 1.4 Implementation Guarantee
```python
# This must work with ZERO configuration:
user_query = "Was ist CBT?"  # German
documents = [chinese_pdf, english_doc, german_guide]
result = system.answer(user_query)
assert result['language'] == 'de'
assert result['answer'] in German
```

---

## II. Internal Lingua Franca Principle

### 2.1 Design Decision
All **internal reasoning, extraction, and intermediate representations** happen in **English**, regardless of user/document languages.

### 2.2 Why English?
1. **LLM Performance**: OpenAI models (GPT-4, Sonnet) are English-optimized
   - Fact extraction quality: +150% in EN vs. multilingual
   - Cost efficiency: fewer tokens needed for EN reasoning
2. **Consistency**: Same Reader/Writer behavior across all language pairs
3. **Caching**: Can reuse English embeddings across sessions
4. **Tool Ecosystem**: Most NLP tools (spaCy, etc.) are English-first

### 2.3 Translation Boundary
```
User Query (any lang) → [translate_to_english] → Internal EN reasoning → [writer target_lang] → Answer (user lang)
Documents (any lang)  → [retrieved as-is]      → Reader extracts EN facts
```

---

## III. Cost Consciousness Principle

### 3.1 Model Assignment
| Task | Model | Reason |
|------|-------|--------|
| Query expansion / translation | gpt-4o-mini | ~100 tokens, support task |
| Language detection routing | langdetect (local) | Zero API cost |
| Fact extraction (Reader) | gpt-4o-mini | Per-chunk, parallelized |
| Answer generation (Writer) | gpt-4o-mini | Single call, main output |
| Evaluation / judge | gpt-4o | Accuracy critical |

### 3.2 Cost Guardrails
- Max **2 extra LLM calls** per query for multilingual support
- Monolingual mode: **zero overhead** (skip translation entirely)
- No external translation APIs (DeepL, Google Translate) — LLM only

### 3.3 Cost Targets
- Average query cost: ~$0.05
- Maximum query cost: $0.15
- Monolingual queries: same cost as before this feature

---

## IV. Graceful Fallback Principle

### 4.1 Monolingual Fast Path
If `retriever.get_document_languages()` returns only 1 language:
- Skip query expansion entirely
- Use existing `optimize_query()` flow
- Zero latency overhead

### 4.2 Degradation Ladder
```
Multilingual DB → Full cross-lingual RRF
Single-language DB → Existing mono flow (fallback)
Translation failure → Use original query (fallback)
Language detection failure → Assume English (fallback)
```

### 4.3 No Silent Failures
Every fallback must be logged (not surfaced to user, but recorded for debugging).

---

## V. Testability Principle

### 5.1 Phase Independence
Each implementation phase must be:
- **Independently deployable**: Can be merged without breaking other phases
- **Independently testable**: Has its own test command
- **Independently committable**: One git commit per phase

### 5.2 Phase Structure
| Phase | Feature | Test Command |
|-------|---------|-------------|
| 1 | Language detection at ingestion | `python -c "from src.retriever import Retriever; r=Retriever(); print(r.get_document_languages())"` |
| 2 | Multilingual query expansion | `pytest tests/test_multilingual_expansion.py` |
| 3 | Cross-lingual RRF retrieval | `pytest tests/test_crosslingual_retrieval.py` |
| 4 | LangGraph integration | `pytest tests/test_e2e_crosslingual.py` |
| 5 | Writer language enforcement | `pytest tests/test_writer_language.py` |

### 5.3 Regression Protection
- English pipeline must not regress: `EN query → ≥ 10 facts extracted`
- Monolingual mode must not slow down: latency delta < 50ms

---

## VI. Backward Compatibility Principle

### 6.1 Zero Breaking Changes
- All existing function signatures must remain valid
- New parameters must have defaults that preserve old behavior
- `src/app.py` (Streamlit UI) requires zero changes

### 6.2 API Contract
```python
# These calls must work unchanged after all phases:
retriever.retrieve_knowledge(query="test")          # still works
agent.run_agent(question="test", intent="SEARCH")  # still works
ingest.ingest_pdfs()                               # still works
```

### 6.3 New Parameters Are Additive Only
```python
# OK: additive parameter with default
def retrieve_knowledge(self, query: str, multilingual_queries: dict = None):
    if multilingual_queries is None:
        return self._existing_flow(query)  # old behavior preserved
```

---

## VII. Architecture Contracts

### 7.1 Component Boundaries
| Component | Input | Output | Must Not |
|-----------|-------|--------|----------|
| Language Router | user_query (str) | {user_lang, doc_langs, queries_per_lang} | Call main LLM |
| Query Expander | query (str), target_langs (list) | {lang: query} | Use external APIs |
| Cross-lingual RRF | queries (dict), top_k (int) | List[Chunk] | Know about user language |
| Reader | chunks, question_en | List[Fact] (EN) | Return non-English facts |
| Writer | facts (EN), target_lang | Answer (target_lang) | Translate facts separately |

### 7.2 State Extensions (LangGraph AgentState)
```python
class AgentState(TypedDict):
    # Existing fields (unchanged)
    question: str
    intent: str
    chunks: List[dict]
    facts: List[str]
    answer: str
    # New fields (additive)
    user_language: str          # e.g. "de", "en", "zh"
    doc_languages: List[str]    # e.g. ["en", "de"]
    multilingual_queries: dict  # e.g. {"en": "CBT techniques", "de": "KVT Techniken"}
```

---

## VIII. Affected Files

| File | Change Type | Description |
|------|-------------|-------------|
| `src/advanced_agent.py` | Extend | Add `expand_query_multilingual()`, extend `step_2_writer(target_language)` |
| `src/retriever.py` | Extend | Add `_multilingual_hybrid_search()`, extend `retrieve_knowledge()` |
| `src/agent_graph.py` | Extend | Extend `AgentState`, update planner/search/writer nodes |
| `src/ingest.py` | Already done | `langdetect` + `metadata["language"]` already implemented |
| `src/app.py` | No change | Zero UI changes required |
| `tests/` | Additive | New test files per phase |

---

## IX. Amendment Process

### 9.1 How to Change This Constitution
1. **Propose** via ADR (Architecture Decision Record) in `docs/adr/`
2. **Justify** cost impact, latency impact, backward compatibility
3. **Code Review** (2 approvals minimum)
   - Ensure backward compatibility
   - Verify testability

### 9.2 Example ADR Format
```markdown
# ADR-XXX: [Title]

## Decision
[What was decided]

## Rationale
[Why this decision]

## Impact
- Cost: [none / +X% / -X%]
- Latency: [none / +Xms]
- Quality: [none / +X%]

## Backward Compatibility
[How old behavior is preserved]
```

---

## X. Glossary

| Term | Definition |
|------|-----------|
| **Lingua Franca** | English; language for all internal reasoning |
| **Language Router** | Component that detects user language & document languages |
| **Query Expansion** | Translating user query into all document languages |
| **Cross-Lingual RRF** | Merging retrieval results across languages using RRF algorithm |
| **Monolingual Mode** | System behavior when all documents are one language (fast path) |
| **Graceful Fallback** | System degrades gracefully (e.g., skips expensive ops in monolingual mode) |
| **Phase** | Independent, testable stage (Detection → Translation → Retrieval → Reader → Writer) |
| **Mini-Model** | Cost-efficient LLM (gpt-4o-mini) for support tasks |
| **Main-LLM** | Primary LLM (gpt-4o, Claude) for critical Reader/Writer tasks |

---

## XI. Success Criteria

By adhering to this Constitution, the DocuInsight RAG system guarantees:

**Linguistic Agnosticism**
- User asks in any language, gets answer in that language
- No manual language configuration needed
- Handles document multilingualism transparently

**Cost Efficiency**
- < $0.15 per query (average ~$0.05)
- Monolingual mode has zero overhead
- Mini-models used only for translation/support

**Maintainability**
- Each phase independently testable & committable
- Clear phase boundaries & contracts
- Zero breaking changes to existing APIs

**Scalability**
- Phases can be parallelized
- Stateless design enables horizontal scaling
- Graceful degradation on failures

---

## XII. Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-04 | Initial Constitution based on v2.0+ cross-lingual architecture |

---

## References

- `data/cross_lingual_architecture.md` — Technical deep-dive
- `data/language_tracks.md` — Implementation checklist with tests
- `src/guardrail.py` — Input validation
- `tests/` — Test suite
