"""
DocuInsight LangGraph Agent (v0.1)
===================================
Graph-based RAG architecture using LangGraph.

Architecture:
- State-based (no manual variable passing)
- Parallel retrieval (Facts + Concepts simultaneously)
- Self-correcting (Quality Check + Retry Logic)
- Modular (nodes are reusable)
"""

import json
import operator
import os
import re
import concurrent.futures
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import logging

logger = logging.getLogger("docuinsight")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO if os.environ.get("DOCUINSIGHT_LOGS", "true").lower() == "true" else logging.WARNING)


from langdetect import detect as _detect
from langgraph.checkpoint.memory import MemorySaver
from retriever import Retriever
from advanced_agent import AdvancedAgent

# --- CONVERSATIONAL MEMORY ---
MAX_HISTORY_TURNS = 5
_memory_saver = None

def _get_memory_saver():
    """Lazy singleton — one MemorySaver shared across all threaded calls.

    Note: Not thread-safe under concurrent initialization. This is fine for
    Streamlit's default single-threaded model. For multi-worker deployments,
    replace with a lock or an external checkpointer (e.g. PostgresSaver).
    """
    global _memory_saver
    if _memory_saver is None:
        _memory_saver = MemorySaver()
    return _memory_saver


# ============================================================================
# STATE DEFINITION - The agent's "memory"
# ============================================================================

class AgentState(TypedDict):
    """
    State flows through the entire graph.
    Every node can read and write.

    NOTE: 'Annotated[List, operator.add]' means:
    - When multiple nodes write to 'documents' concurrently,
      the lists are MERGED (not overwritten).
    - Perfect for parallel retrieval!
    """

    # === INPUT ===
    question: str
    intent: str

    # === RETRIEVAL ===
    documents: Annotated[List[Dict[str, Any]], operator.add]
    context_text: str
    references: List[Dict[str, Any]]

    # === ANALYSIS ===
    facts: List[Dict[str, Any]]
    concepts: List[str]
    quality_score: float

    # === OUTPUT ===
    draft_answer: str
    final_answer: str

    # === CONTROL FLOW ===
    retry_count: int
    needs_recursion: bool
    new_query: Optional[str]
    error_message: Optional[str]
    entity_match: bool  # True if COMPARE matched docs by entity name
    entities: Optional[List[str]]  # Extracted entity names (COMPARE intent)
    english_question: Optional[str]  # English translation for cross-lingual extraction

    # === CROSS-LINGUAL (additive, all Optional) ===
    user_language: Optional[str]          # ISO 639-1 code detected from user query
    doc_languages: Optional[List[str]]    # Unique languages in ChromaDB
    multilingual_queries: Optional[Dict[str, str]]  # lang → optimised search phrase

    # === REFLECTION ===
    critic_feedback: Optional[str]       # JSON string from Critic, None if skipped
    reflection_skipped: Optional[bool]   # True if reflection was skipped, for logs

    # === CONVERSATIONAL MEMORY ===
    chat_history: Optional[List[Dict[str, str]]]

    # === DEVELOPER MODE ===
    logs: Annotated[List[str], operator.add]


# ============================================================================
# NODES - The "workers" in the graph
# ============================================================================


def node_aggregator(state: AgentState) -> Dict[str, Any]:
    """SYNCHRONIZATION POINT - Waits for both tracks and merges results."""
    documents = state.get("documents", [])

    all_sources = []
    for doc in documents:
        refs = doc.get("references", [])
        for ref in refs:
            all_sources.append(ref.get('file', 'Unknown'))

    unique_sources = list(set(all_sources))

    all_context = "\n\n".join([doc.get("content", "") for doc in documents])

    quality = 1.0 if len(all_context) > 100 else 0.3

    # Deduplicated chunk list across both tracks
    seen_ids: set = set()
    all_refs_deduped = []
    for doc in documents:
        for ref in doc.get("references", []):
            rid = ref.get("id")
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_refs_deduped.append(ref)
    all_refs_deduped.sort(key=lambda x: x.get("id", 0))


    log_summary = f"🔄 AGGREGATOR: {len(all_refs_deduped)} unique chunks from {len(unique_sources)} files — {len(all_context)} chars total"
    chunk_detail = "  ".join(
        f"[{r['id']}] {r.get('file','?').replace('.pdf','')[:20]} S.{r.get('page','?')}"
        for r in all_refs_deduped
    )
    log_chunks = f"AGGREGATOR_CHUNKS: {chunk_detail}"
    logger.info(log_summary)

    return {
        "context_text": all_context,
        "quality_score": quality,
        "logs": [log_summary, log_chunks]
    }


def node_quality_check(state: AgentState) -> Dict[str, Any]:
    """
    QUALITY GATE - LLM-based relevance check WITH reasoning!
    """
    context = state.get("context_text", "")
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    max_retries = 1

    # Fast-path: Entity Match already validated document relevance — skip quality gate
    if state.get("entity_match"):
        log = f"✅ QUALITY_CHECK: Entity Match confirmed — skipping LLM quality gate ({len(context)} chars)"
        logger.info(log)
        return {
            "quality_score": 0.8,
            "needs_recursion": False,
            "logs": [log]
        }

    if not context or len(context) < 100:
        log = f"⚠️ QUALITY_CHECK: No context found ({len(context)} chars). Triggering retry."
        logger.info(log)
        return {
            "quality_score": 0.0,
            "needs_recursion": True,
            "new_query": f"{question} detailed",
            "retry_count": retry_count + 1,
            "logs": [log]
        }

    try:
        from llm_provider import get_llm_client, get_model_name
        import json

        client = get_llm_client()
        model_name = get_model_name()

        # Smart sampling: instead of blind truncation, sample from different
        # parts of the context so the judge sees both Facts and Concepts tracks
        if len(context) > 5000:
            chunk_size = 1500
            head = context[:chunk_size]
            # Middle section (where Concepts track results often land)
            mid_start = len(context) // 2
            middle = context[mid_start:mid_start + chunk_size]
            # Tail (last chunks, often from retry results)
            tail = context[-chunk_size:]
            context_sample = f"{head}\n\n[...]\n\n{middle}\n\n[...]\n\n{tail}"
        else:
            context_sample = context

        check_prompt = f"""You are a quality checker for RAG systems.
Assess whether the context can PRECISELY answer the question.

QUESTION: "{question}"

CONTEXT (sampled from {len(context)} chars): "{context_sample}"

EVALUATION CRITERIA:
- 1.0: Perfect answer possible
- 0.8: Very good (main info present, minor details missing)
- 0.6: Sufficient (topic covered, not all aspects)
- 0.4: Weak (only tangentially related)
- 0.2: Very weak (topic barely mentioned)
- 0.1: Topic completely missed

RESPONSE FORMAT (JSON):
{{"score": 0.7, "reason": "Brief explanation (1-2 sentences)"}}"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": check_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        quality = float(result.get("score", 0.5))
        reason = result.get("reason", "No reason provided")

    except Exception as e:
        logger.info("Quality Check LLM Error: %s", e)
        quality = 0.6  # Borderline pass — don't block, but don't endorse either
        reason = f"Fallback heuristic (LLM error: {str(e)[:30]})"

    # Case A: Poor quality and retries remaining -> SMART RETRY
    if quality < 0.6 and retry_count < max_retries:
        try:
            refine_prompt = f"""You are a query optimiser for vector search in documents.
The previous search was NOT successful.

ORIGINAL QUERY: "{question}"
PROBLEM: {reason}

TASK:
Generate a NEW, IMPROVED search query that addresses the problem.

STRATEGIES:
- If "topic missed": Use more specific keywords
- If "too general": Add domain-specific terms
- If "only tangential": Focus on the core aspect of the question

IMPORTANT: Return ONLY the new query (no explanation, no quotes)."""

            refine_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.3
            )
            smart_query = refine_response.choices[0].message.content.strip()
            smart_query = smart_query.replace('"', '').replace("'", "")

        except Exception as e:
            logger.info("Smart Query Refinement Error: %s", e)
            smart_query = f"{question} specific detailed"

        log_score = f"⚠️ QUALITY_CHECK: Score {quality:.2f}/1.0"
        log_reason = f"📋 REASON: {reason}"
        log_action = f"🔄 ACTION: Smart Retry {retry_count + 1}/{max_retries}"
        log_new_query = f"🎯 NEW QUERY: '{smart_query}'"

        logger.info(log_score)
        logger.info(log_reason)
        logger.info(log_action)
        logger.info(log_new_query)

        return {
            "needs_recursion": True,
            "new_query": smart_query,
            "retry_count": retry_count + 1,
            "quality_score": quality,
            "logs": [log_score, log_reason, log_action, log_new_query]
        }

    # Case B: Poor but NO retries left -> BEST EFFORT
    elif quality < 0.6 and retry_count >= max_retries:
        log_score = f"⚠️ QUALITY_CHECK: Score {quality:.2f}/1.0 (after {retry_count} retries)"
        log_reason = f"📋 REASON: {reason}"
        log_action = "❗ ACTION: Max retries reached → 'Best Effort' mode"

        logger.info(log_score)
        logger.info(log_reason)
        logger.info(log_action)

        return {
            "needs_recursion": False,
            "quality_score": quality,
            "error_message": f"Note: Retrieved data may not be specific enough ({reason})",
            "logs": [log_score, log_reason, log_action]
        }

    # Case C: Good -> CONTINUE
    else:
        log_score = f"✅ QUALITY_CHECK: Score {quality:.2f}/1.0"
        log_reason = f"📋 REASON: {reason}"
        log_action = "✅ ACTION: Proceeding to Reader"

        logger.info(log_score)
        logger.info(log_reason)
        logger.info(log_action)

        return {
            "needs_recursion": False,
            "quality_score": quality,
            "logs": [log_score, log_reason, log_action]
        }


def node_error_handler(state: AgentState) -> Dict[str, Any]:
    """ERROR NODE - Fallback for errors or missing information."""
    error_msg = state.get("error_message", "No relevant information found.")

    log = f"❌ ERROR_HANDLER: Fallback activated. Reason: {error_msg}"
    logger.info(log)

    return {
        "final_answer": f"I'm sorry, {error_msg}",
        "logs": [log]
    }


# ============================================================================
# CONDITIONAL EDGES - Dynamic Routing
# ============================================================================

def should_retry(state: AgentState) -> str:
    """Decides after Quality Check: retry or continue?"""
    if state.get("needs_recursion", False):
        return "retry"
    else:
        return "continue"



# ============================================================================
# GRAPH CONSTRUCTION - The core
# ============================================================================

def create_graph(retriever: Retriever, agent: AdvancedAgent, checkpointer=None) -> StateGraph:
    """
    Creates the compiled LangGraph with real functions bound.
    When checkpointer is provided, the graph persists state across invocations.
    """

    # --- NODE FACTORIES: Bind Retriever & Agent ---

    def planner_bound(state: AgentState) -> Dict[str, Any]:
        """ENTRY POINT — strategy + cross-lingual language detection."""
        intent = state.get("intent", "SEARCH")
        question = state.get("question", "")

        strategy_reasons = {
            "SEARCH": "Question requires fact-based search. Dual-track activated: Facts + Concepts in parallel.",
            "SUMMARIZE": "Summarisation request. Single-track: Full-document retrieval.",
            "COMPARE": "Comparison request. Multi-document retrieval + Criteria Scout.",
        }
        reason = strategy_reasons.get(intent, f"Standard strategy for intent '{intent}'.")
        log_strategy = f"🧠 PLANNER: Intent='{intent}' → {reason}"
        log_question = f"📝 QUESTION: '{question[:80]}...'" if len(question) > 80 else f"📝 QUESTION: '{question}'"
        logger.info(log_strategy)
        logger.info(log_question)

        # --- Cross-lingual: detect user language ---
        try:
            user_language = _detect(question) if question else "en"
        except Exception as e:
            logger.debug("Language detection failed, defaulting to 'en': %s", e)
            user_language = "en"

        # --- Cross-lingual: get corpus languages ---
        doc_languages = retriever.get_document_languages()

        log_lang = f"🌐 LANG: user={user_language}, docs={doc_languages}"
        logger.info(log_lang)

        # --- Conversational memory: query rewriting ---
        chat_history = state.get("chat_history") or []
        rewritten_question = None
        log_rewrite = None

        if chat_history and intent in ("SEARCH", "COMPARE", "SUMMARIZE"):
            rewritten = agent.rewrite_follow_up(question, chat_history)
            if rewritten != question:
                log_rewrite = f"🔄 REWRITE: '{question}' → '{rewritten}'"
                logger.info(log_rewrite)
                rewritten_question = rewritten

        result = {
            "retry_count": state.get("retry_count", 0),
            "needs_recursion": False,
            "quality_score": 0.0,
            "english_question": None,
            "user_language": user_language,
            "doc_languages": doc_languages,
            "multilingual_queries": None,
            "logs": [entry for entry in [log_strategy, log_question, log_lang, log_rewrite] if entry]
        }

        if rewritten_question:
            result["question"] = rewritten_question

        return result

    def search_facts_bound(state: AgentState) -> Dict[str, Any]:
        """Facts Search with real Retriever - uses smart query on retry.
        Cross-lingual: uses multilingual query expansion when corpus has >1 language."""

        if state.get("new_query"):
            base_query = state["new_query"]
            is_retry = True
        else:
            base_query = state["question"]
            is_retry = False

        doc_languages = state.get("doc_languages") or ["en"]

        if is_retry:
            log_original = f"🔄 FACTS_RETRY: '{base_query}'"
        else:
            log_original = f"📝 FACTS_ORIGINAL: '{base_query}'"
        logger.info(log_original)

        # --- Multilingual path ---
        if len(doc_languages) > 1:
            multilingual_queries = agent.expand_query_multilingual(base_query, doc_languages, intent="FACTS")
            log_multi = f"🌐 FACTS_MULTILINGUAL: {list(multilingual_queries.keys())} — {multilingual_queries}"
            logger.info(log_multi)
            try:
                context, refs, citation_map = retriever.retrieve_knowledge(
                    base_query, top_k=10, multilingual_queries=multilingual_queries
                )
                unique_files = list(set([r.get('file', 'Unknown') for r in refs]))
                file_list = ", ".join([f.replace('.pdf', '')[:25] for f in unique_files[:5]])
                log_hits = f"📚 FACTS_FOUND: {len(refs)} hits from {len(unique_files)} files: [{file_list}]"
                chunk_detail = "  ".join(f"[{r['id']}] {r.get('file','?').replace('.pdf','')[:20]} S.{r.get('page','?')}" for r in refs)
                log_chunks = f"📚 FACTS_CHUNKS: {chunk_detail}"
                logger.info(log_hits)
                return {
                    "documents": [{"type": "facts", "content": context, "references": refs, "citation_map": citation_map}],
                    "multilingual_queries": multilingual_queries,
                    "logs": [log_original, log_multi, log_hits, log_chunks]
                }
            except Exception as e:
                log = f"❌ SEARCH_FACTS (multilingual): Error - {str(e)}"
                logger.info(log)
                return {"documents": [], "error_message": f"Facts search error: {str(e)}", "logs": [log_original, log_multi, log]}

        # --- Monolingual path (unchanged) ---
        optimized = agent.optimize_query(base_query, intent="FACTS")
        search_query = optimized["query"]
        optimization_reason = optimized["reasoning"]
        log_optimized = f"🔍 FACTS_QUERY: '{search_query}' ({optimization_reason})"
        logger.info(log_optimized)

        try:
            context, refs, citation_map = retriever.retrieve_knowledge(search_query, top_k=10)

            unique_files = list(set([r.get('file', 'Unknown') for r in refs]))
            file_list = ", ".join([f.replace('.pdf', '')[:25] for f in unique_files[:5]])
            log_hits = f"📚 FACTS_FOUND: {len(refs)} hits from {len(unique_files)} files: [{file_list}]"
            chunk_detail = "  ".join(f"[{r['id']}] {r.get('file','?').replace('.pdf','')[:20]} S.{r.get('page','?')}" for r in refs)
            log_chunks = f"📚 FACTS_CHUNKS: {chunk_detail}"
            logger.info(log_hits)

            return {
                "documents": [{
                    "type": "facts",
                    "content": context,
                    "references": refs,
                    "citation_map": citation_map
                }],
                "logs": [log_original, log_optimized, log_hits, log_chunks]
            }
        except Exception as e:
            log = f"❌ SEARCH_FACTS: Error - {str(e)}"
            logger.info(log)
            return {
                "documents": [],
                "error_message": f"Facts search error: {str(e)}",
                "logs": [log_original, log_optimized, log]
            }


    def search_concepts_bound(state: AgentState) -> Dict[str, Any]:
        """Concepts Search — for COMPARE intent, uses document-aware entity matching
        (searches first pages for author names / terms), otherwise standard vector search.
        Cross-lingual: uses multilingual query expansion when corpus has >1 language."""

        intent = state.get("intent", "SEARCH")
        question = state.get("new_query") or state["question"]
        is_retry = bool(state.get("new_query"))
        doc_languages = state.get("doc_languages") or ["en"]

        # === COMPARE: Document-Aware Entity Matching ===
        if intent == "COMPARE":
            logs = []
            try:
                # Step 1: Extract entities (author names, terms) via LLM
                extract_result = agent.extract_entities(question)
                entities = extract_result.get("entities", [])
            except Exception as e:
                logger.debug("Entity extraction failed: %s", e)
                entities = []

            log_entities = f"🔎 COMPARE_ENTITIES: {entities} (from: '{question[:60]}')"
            logger.info(log_entities)
            logs.append(log_entities)

            matched_files = []
            if entities:
                # Step 2: Search first pages of all docs for these entities
                matches = retriever.match_documents_by_entities(entities, intro_pages=2)
                matched_files = [filename for filename, _ in matches]
                match_detail = [(f[:30], ents) for f, ents in matches]
                log_matches = f"📄 COMPARE_MATCHED: {match_detail}"
                logger.info(log_matches)
                logs.append(log_matches)

            if len(matched_files) >= 2:
                # Step 3: Load matched documents via retrieve_multiple_documents
                context, refs, citation_map = retriever.retrieve_multiple_documents(
                    matched_files[:3], total_token_budget=100000, strategy="balanced"
                )
                log_loaded = f"📚 COMPARE_LOADED: {len(refs)} pages from {len(matched_files)} documents"
                logger.info(log_loaded)
                logs.append(log_loaded)

                return {
                    "documents": [{
                        "type": "concepts",
                        "content": context,
                        "references": refs,
                        "citation_map": citation_map
                    }],
                    "entity_match": True,
                    "entities": entities,
                    "logs": logs
                }
            else:
                log_fallback = f"⚠️ COMPARE_FALLBACK: <2 docs matched ({len(matched_files)}), using vector search"
                logger.info(log_fallback)
                logs.append(log_fallback)
                # Fall through to standard vector search below

        # === STANDARD: Vector Search (with optional multilingual path) ===
        base_query = state.get("new_query") or state["question"]

        if is_retry:
            log_original = f"🔄 CONCEPTS_RETRY: '{base_query}'"
        else:
            log_original = f"📝 CONCEPTS_ORIGINAL: '{base_query}'"
        logger.info(log_original)

        extra_logs = logs if intent == "COMPARE" else []

        # --- Multilingual path (skip for COMPARE — entity matching handles doc selection) ---
        if len(doc_languages) > 1 and intent != "COMPARE":
            multilingual_queries = agent.expand_query_multilingual(base_query, doc_languages, intent="CONCEPTS")
            log_multi = f"🌐 CONCEPTS_MULTILINGUAL: {list(multilingual_queries.keys())}"
            logger.info(log_multi)
            try:
                context, refs, citation_map = retriever.retrieve_knowledge(
                    base_query, top_k=10, multilingual_queries=multilingual_queries
                )
                unique_files = list(set([r.get('file', 'Unknown') for r in refs]))
                file_list = ", ".join([f.replace('.pdf', '')[:25] for f in unique_files[:5]])
                log_hits = f"🔬 CONCEPTS_FOUND: {len(refs)} hits from {len(unique_files)} files: [{file_list}]"
                chunk_detail = "  ".join(f"[{r['id']}] {r.get('file','?').replace('.pdf','')[:20]} S.{r.get('page','?')}" for r in refs)
                log_chunks = f"🔬 CONCEPTS_CHUNKS: {chunk_detail}"
                logger.info(log_hits)
                return {
                    "documents": [{"type": "concepts", "content": context, "references": refs, "citation_map": citation_map}],
                    "logs": extra_logs + [log_original, log_multi, log_hits, log_chunks]
                }
            except Exception as e:
                log = f"❌ SEARCH_CONCEPTS (multilingual): Error - {str(e)}"
                logger.info(log)
                return {"documents": [], "error_message": f"Concepts search error: {str(e)}", "logs": extra_logs + [log_original, log_multi, log]}

        # --- Monolingual path (unchanged) ---
        optimized = agent.optimize_query(base_query, intent="CONCEPTS")
        search_query = optimized["query"]
        optimization_reason = optimized["reasoning"]
        log_optimized = f"🔍 CONCEPTS_QUERY: '{search_query}' ({optimization_reason})"
        logger.info(log_optimized)

        try:
            context, refs, citation_map = retriever.retrieve_knowledge(search_query, top_k=10)

            unique_files = list(set([r.get('file', 'Unknown') for r in refs]))
            file_list = ", ".join([f.replace('.pdf', '')[:25] for f in unique_files[:5]])
            log_hits = f"🔬 CONCEPTS_FOUND: {len(refs)} hits from {len(unique_files)} files: [{file_list}]"
            chunk_detail = "  ".join(f"[{r['id']}] {r.get('file','?').replace('.pdf','')[:20]} S.{r.get('page','?')}" for r in refs)
            log_chunks = f"🔬 CONCEPTS_CHUNKS: {chunk_detail}"
            logger.info(log_hits)

            return {
                "documents": [{
                    "type": "concepts",
                    "content": context,
                    "references": refs,
                    "citation_map": citation_map
                }],
                "logs": extra_logs + [log_original, log_optimized, log_hits, log_chunks]
            }
        except Exception as e:
            log = f"❌ SEARCH_CONCEPTS: Error - {str(e)}"
            logger.info(log)
            return {
                "documents": [],
                "error_message": f"Concepts search error: {str(e)}",
                "logs": extra_logs + [log_original, log_optimized, log]
            }

    def reader_bound(state: AgentState) -> Dict[str, Any]:
        """Map-Reduce Reader: extracts facts per chunk in parallel, then merges.
        Intent-aware: passes intent + entities to Reader for targeted extraction.
        Cross-lingual: translates non-English questions to English for extraction."""
        question = state["question"]
        context = state.get("context_text", "")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 1)
        intent = state.get("intent", "SEARCH")

        # Cross-lingual fix: translate question to English for extraction
        # (documents are predominantly English — German questions produce 0 facts)
        english_q = state.get("english_question")
        if not english_q:
            english_q = agent.translate_to_english(question)
        reader_question = english_q  # Use English for extraction from English docs

        # Reuse entities from concepts search (stored in state)
        reader_entities = state.get("entities") if intent == "COMPARE" else None

        if not context:
            log = "❌ READER: No context available"
            logger.info(log)
            return {
                "facts": [],
                "error_message": "No context available for analysis.",
                "logs": [log]
            }

        log_input = f"📥 READER_INPUT: {len(context)} chars of context received"
        if reader_question != question:
            log_lang = f"🌐 READER_LANG: Translated to English for extraction: '{reader_question[:80]}...'"
            logger.info(log_lang)
        else:
            log_lang = None
        logger.info(log_input)

        base_logs = [entry for entry in [log_lang, log_input] if entry]

        # --- HYBRID: Stuffing for small contexts, Map-Reduce for large ones ---
        MAP_REDUCE_THRESHOLD = 25000  # ~6k tokens — safe for all models

        if len(context) <= MAP_REDUCE_THRESHOLD:
            # STUFFING MODE: single LLM call (original behavior)
            log_mode = f"📋 READER_MODE: Stuffing ({len(context)} chars < {MAP_REDUCE_THRESHOLD} threshold)"
            logger.info(log_mode)
            try:
                result = agent.extract_facts(reader_question, context, intent=intent, entities=reader_entities)
                status = result.get("status")
                facts = result.get("facts", [])

                if status == "SUFFICIENT":
                    log_status = f"✅ READER: Status=SUFFICIENT - {len(facts)} facts extracted"
                    logger.info(log_status)
                    facts_preview = []
                    for idx, f in enumerate(facts[:3], 1):
                        facts_preview.append(f"  {idx}. [{f.get('source_id', '?')}] {f.get('fact', '')[:80]}...")
                    log_facts = "📊 FACTS_PREVIEW:\n" + "\n".join(facts_preview)
                    logger.info(log_facts)
                    return {
                        "facts": facts,
                        "quality_score": 1.0,
                        "logs": base_logs + [log_mode, log_status, log_facts]
                    }

                elif status == "INSUFFICIENT" and retry_count < max_retries:
                    missing = result.get("missing_reason", "Unknown")
                    log_status = f"⚠️ READER: Status=INSUFFICIENT - {missing}"
                    log_action = f"🔄 READER: Triggering retry loop (count: {retry_count})"
                    logger.info(log_status)
                    logger.info(log_action)
                    return {
                        "facts": [],
                        "quality_score": 0.3,
                        "needs_recursion": True,
                        "new_query": result.get("new_query", question),
                        "logs": base_logs + [log_mode, log_status, log_action]
                    }

                else:
                    log_status = f"❗ READER: Status=INSUFFICIENT, but max retries ({retry_count}) reached."
                    log_action = "Forcing BEST EFFORT extraction..."
                    logger.info(log_status)
                    logger.info(log_action)
                    if not facts:
                        facts = [{
                            "fact": "The context contained no direct matches for the specific question, but the following text was analysed.",
                            "source_id": "System_Notice",
                            "reasoning": "Best effort fallback: no exact data found."
                        }]
                    return {
                        "facts": facts,
                        "quality_score": 0.5,
                        "needs_recursion": False,
                        "error_message": f"Answer based on incomplete information ({result.get('missing_reason')})",
                        "logs": base_logs + [log_mode, log_status, log_action]
                    }

            except Exception as e:
                log = f"❌ READER: Error - {str(e)}"
                logger.info(log)
                return {"facts": [], "error_message": f"Reader error: {str(e)}", "logs": base_logs + [log]}

        # MAP-REDUCE MODE: split context, extract per chunk in parallel, merge
        chunk_splits = re.split(r'(?=SOURCE_ID \[\d+\])', context)
        chunks = [c.strip() for c in chunk_splits if c.strip() and len(c.strip()) > 50]

        if len(chunks) <= 1:
            max_chars = 8000
            chunks = [context[i:i + max_chars].strip() for i in range(0, len(context), max_chars)]

        log_mode = f"🗺️ READER_MODE: Map-Reduce ({len(context)} chars > {MAP_REDUCE_THRESHOLD} threshold) → {len(chunks)} chunks"
        logger.info(log_mode)

        all_facts = []
        map_logs = []
        insufficient_count = 0

        def extract_from_chunk(chunk_text):
            return agent.extract_facts(reader_question, chunk_text, intent=intent, entities=reader_entities)

        try:
            # Limit parallel chunk calls to avoid TPM rate limits on smaller API tiers
            map_workers = min(len(chunks), int(os.getenv("MAP_WORKERS", "3")))
            with concurrent.futures.ThreadPoolExecutor(max_workers=map_workers) as executor:
                futures = {executor.submit(extract_from_chunk, chunk): i for i, chunk in enumerate(chunks)}
                for future in concurrent.futures.as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        result = future.result()
                        facts = result.get("facts", [])
                        status = result.get("status", "INSUFFICIENT")
                        if facts:
                            all_facts.extend(facts)
                            map_logs.append(f"  ✅ Chunk {chunk_idx + 1}: {len(facts)} facts")
                        else:
                            insufficient_count += 1
                            map_logs.append(f"  ⚠️ Chunk {chunk_idx + 1}: no facts")
                    except Exception as e:
                        map_logs.append(f"  ❌ Chunk {chunk_idx + 1}: {str(e)[:60]}")

            log_reduce = f"📊 READER_REDUCE: {len(all_facts)} facts from {len(chunks)} chunks ({insufficient_count} empty)"
            logger.info(log_reduce)

            if all_facts:
                log_status = f"✅ READER: {len(all_facts)} facts extracted (Map-Reduce)"
                logger.info(log_status)
                facts_preview = []
                for idx, f in enumerate(all_facts[:3], 1):
                    facts_preview.append(f"  {idx}. [{f.get('source_id', '?')}] {f.get('fact', '')[:80]}...")
                log_facts = "📊 FACTS_PREVIEW:\n" + "\n".join(facts_preview)
                logger.info(log_facts)
                return {
                    "facts": all_facts,
                    "quality_score": 1.0,
                    "logs": base_logs + [log_mode] + map_logs + [log_reduce, log_status, log_facts]
                }

            elif retry_count < max_retries:
                log_status = f"⚠️ READER: No facts across {len(chunks)} chunks"
                log_action = f"🔄 READER: Triggering retry (count: {retry_count})"
                logger.info(log_status)
                return {
                    "facts": [], "quality_score": 0.3, "needs_recursion": True,
                    "new_query": question,
                    "logs": base_logs + [log_mode] + map_logs + [log_reduce, log_status, log_action]
                }

            else:
                log_status = f"❗ READER: No facts, max retries ({retry_count}) reached → Best Effort"
                logger.info(log_status)
                all_facts = [{
                    "fact": "The context contained no direct matches for the specific question.",
                    "source_id": "System_Notice",
                    "reasoning": "Best effort fallback."
                }]
                return {
                    "facts": all_facts, "quality_score": 0.5, "needs_recursion": False,
                    "error_message": "Answer based on incomplete information",
                    "logs": base_logs + [log_mode] + map_logs + [log_reduce, log_status]
                }

        except Exception as e:
            log = f"❌ READER: Error - {str(e)}"
            logger.info(log)
            return {"facts": [], "error_message": f"Reader error: {str(e)}", "logs": base_logs + [log]}

    def writer_bound(state: AgentState) -> Dict[str, Any]:
        """Writer with real Agent."""
        question = state["question"]
        facts = state.get("facts", [])
        quality_score = state.get("quality_score", 0.0)
        error_message = state.get("error_message")

        if not facts:
            log = "⚠️ WRITER: No facts available, generating fallback answer"
            logger.info(log)
            return {
                "final_answer": "I'm sorry, I couldn't find any relevant information in the documents.",
                "logs": [log]
            }

        log_input = f"📥 WRITER_INPUT: {len(facts)} facts received (quality: {quality_score:.2f})"
        logger.info(log_input)

        facts_preview = []
        for idx, f in enumerate(facts[:4], 1):
            fact_text = f.get('fact', '')[:100]
            source_id = f.get('source_id', '?')
            facts_preview.append(f"  [{source_id}] {fact_text}...")

        log_facts_input = "📋 WRITER_SEES:\n" + "\n".join(facts_preview)
        if len(facts) > 4:
            log_facts_input += f"\n  ... and {len(facts) - 4} more facts"
        logger.info(log_facts_input)

        intent = state.get("intent", "SEARCH")
        target_language = state.get("user_language")

        try:
            extracted_facts = {"facts": facts}
            draft = agent.draft_answer(question, extracted_facts, target_language=target_language)

            log_output = f"✍️ WRITER_OUTPUT: {len(draft)} chars generated"
            logger.info(log_output)

            # === REFLECTION GATE ===
            should_reflect = False
            if intent in ("SUMMARIZE", "COMPARE"):
                should_reflect = True
            elif intent == "SEARCH" and quality_score < 0.8:
                should_reflect = True
            # Hard-skip conditions (override above)
            if intent == "CHAT" or not facts or quality_score <= 0.6:
                should_reflect = False

            if not should_reflect:
                log_skip = f"🔄 REFLECTION: Skipped (intent={intent}, quality={quality_score:.2f})"
                logger.info(log_skip)

                if error_message and quality_score < 0.6:
                    draft = f"{draft}\n\n---\n⚠️ *{error_message}*"

                return {
                    "draft_answer": draft,
                    "final_answer": draft,
                    "reflection_skipped": True,
                    "logs": [log_input, log_facts_input, log_output, log_skip]
                }

            # === CRITIC ===
            log_critic_start = f"🔍 CRITIC: Fact-checking draft ({len(draft)} chars, {len(facts)} facts)..."
            logger.info(log_critic_start)

            critic_result = agent.critique_draft(question, facts, draft, intent)
            verdict = critic_result.get("verdict", "PASS")
            hallucination_count = len(critic_result.get("hallucinations", []))
            missing_count = len(critic_result.get("missing_facts", []))

            log_critic = f"📋 CRITIC: verdict={verdict}, hallucinations={hallucination_count}, missing={missing_count}"
            logger.info(log_critic)

            critic_feedback_json = json.dumps(critic_result)

            if verdict == "PASS":
                if error_message and quality_score < 0.6:
                    draft = f"{draft}\n\n---\n⚠️ *{error_message}*"

                return {
                    "draft_answer": draft,
                    "final_answer": draft,
                    "critic_feedback": critic_feedback_json,
                    "reflection_skipped": False,
                    "logs": [log_input, log_facts_input, log_output, log_critic_start, log_critic]
                }

            # === REVISER (verdict == "REVISE") ===
            revision_instructions = critic_result.get("revision_instructions", "")
            log_revise_start = "✏️ REVISER: Revising draft based on critic feedback..."
            logger.info(log_revise_start)

            revised = agent.revise_draft(
                question, facts, draft, revision_instructions, target_language=target_language
            )

            log_revise = f"✏️ REVISER: {len(draft)} → {len(revised)} chars"
            logger.info(log_revise)

            if error_message and quality_score < 0.6:
                revised = f"{revised}\n\n---\n⚠️ *{error_message}*"

            return {
                "draft_answer": draft,
                "final_answer": revised,
                "critic_feedback": critic_feedback_json,
                "reflection_skipped": False,
                "logs": [log_input, log_facts_input, log_output, log_critic_start, log_critic, log_revise_start, log_revise]
            }

        except Exception as e:
            log = f"❌ WRITER: Error - {str(e)}"
            logger.info(log)
            return {
                "final_answer": f"Error generating answer: {str(e)}",
                "logs": [log_input, log_facts_input, log]
            }

    def history_update_bound(state: AgentState) -> Dict[str, Any]:
        """Appends current Q&A to chat_history with sliding window."""
        history = list(state.get("chat_history") or [])
        question = state.get("question", "")
        final_answer = state.get("final_answer", "")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": final_answer})

        max_messages = MAX_HISTORY_TURNS * 2
        history = history[-max_messages:]

        log = f"💬 HISTORY: {len(history) // 2} turns stored (max {MAX_HISTORY_TURNS})"
        logger.info(log)

        return {"chat_history": history, "logs": [log]}

    # --- BUILD GRAPH ---

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_bound)
    workflow.add_node("search_facts", search_facts_bound)
    workflow.add_node("search_concepts", search_concepts_bound)
    workflow.add_node("aggregator", node_aggregator)
    workflow.add_node("quality_check", node_quality_check)
    workflow.add_node("reader", reader_bound)
    workflow.add_node("writer", writer_bound)
    workflow.add_node("history_update", history_update_bound)
    workflow.add_node("error_handler", node_error_handler)

    # --- FLOW DEFINITION ---

    workflow.set_entry_point("planner")

    # From Planner to both tracks (PARALLEL!)
    workflow.add_edge("planner", "search_facts")
    workflow.add_edge("planner", "search_concepts")

    # SYNCHRONISATION: Both tracks must finish
    workflow.add_edge("search_facts", "aggregator")
    workflow.add_edge("search_concepts", "aggregator")

    # After aggregation: Quality Check
    workflow.add_edge("aggregator", "quality_check")

    # CONDITIONAL: Retry or continue?
    workflow.add_conditional_edges(
        "quality_check",
        should_retry,
        {
            "retry": "planner",
            "continue": "reader"
        }
    )

    # Reader → Writer → History Update → END
    workflow.add_edge("reader", "writer")
    workflow.add_edge("writer", "history_update")
    workflow.add_edge("history_update", END)

    # Error Handler as fallback
    workflow.add_edge("error_handler", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_cached_graph_stateless = None
_cached_graph_memorized = None
_cached_graph_key = None  # (retriever_id, agent_id) to detect changed instances


def _get_or_create_graph(retriever, agent, thread_id):
    """Returns a cached compiled graph, rebuilding only when retriever/agent change."""
    global _cached_graph_stateless, _cached_graph_memorized, _cached_graph_key
    key = (id(retriever), id(agent))
    if key != _cached_graph_key:
        _cached_graph_stateless = None
        _cached_graph_memorized = None
        _cached_graph_key = key
    if thread_id:
        if _cached_graph_memorized is None:
            _cached_graph_memorized = create_graph(retriever, agent, checkpointer=_get_memory_saver())
        return _cached_graph_memorized
    else:
        if _cached_graph_stateless is None:
            _cached_graph_stateless = create_graph(retriever, agent)
        return _cached_graph_stateless


def run_agent(
    question: str,
    retriever: Retriever,
    agent: AdvancedAgent,
    intent: str = "SEARCH",
    thread_id: str = None
) -> Dict[str, Any]:
    """
    High-level interface: ask a question, get an answer.
    When thread_id is provided, conversational memory is enabled (MemorySaver).
    Without thread_id, behavior is identical to pre-feature (stateless).
    """

    graph = _get_or_create_graph(retriever, agent, thread_id)

    # chat_history is intentionally omitted so the checkpointed value persists
    initial_state: AgentState = {
        "question": question,
        "intent": intent,
        "documents": [],
        "context_text": "",
        "references": [],
        "facts": [],
        "concepts": [],
        "quality_score": 0.0,
        "draft_answer": "",
        "final_answer": "",
        "retry_count": 0,
        "needs_recursion": False,
        "new_query": None,
        "error_message": None,
        "entity_match": False,
        "entities": None,
        "english_question": None,
        "user_language": None,
        "doc_languages": None,
        "multilingual_queries": None,
        "critic_feedback": None,
        "reflection_skipped": None,
        "logs": []
    }

    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    logger.info("\n%s", "=" * 60)
    logger.info("LangGraph Agent started for: '%s'", question)
    if thread_id:
        logger.info("Thread: %s", thread_id)
    logger.info("%s\n", "=" * 60)

    try:
        final_state = graph.invoke(initial_state, config=config)

        return {
            "success": True,
            "final_answer": final_state.get("final_answer", ""),
            "references": _extract_references(final_state),
            "facts": final_state.get("facts", []),
            "concepts": final_state.get("concepts", []),
            "retry_count": final_state.get("retry_count", 0),
            "quality_score": final_state.get("quality_score", 1.0),
            "logs": final_state.get("logs", [])
        }

    except Exception as e:
        logger.info("Graph Execution Error: %s", e)
        return {
            "success": False,
            "final_answer": f"Execution error: {str(e)}",
            "references": [],
            "logs": [f"❌ Graph error: {str(e)}"],
            "error": str(e)
        }


def _extract_references(state: AgentState) -> List[Dict[str, Any]]:
    """Extracts AND DEDUPLICATES references from documents."""
    all_refs = []
    seen_ids = set()

    for doc in state.get("documents", []):
        refs = doc.get("references", [])
        for ref in refs:
            unique_key = (ref.get('id'), ref.get('file'))
            if unique_key not in seen_ids:
                seen_ids.add(unique_key)
                all_refs.append(ref)

    all_refs.sort(key=lambda x: x.get('id', 0))
    return all_refs


# ============================================================================
# STANDALONE SETUP (Factory Pattern for LangGraph Studio)
# ============================================================================

def get_graph():
    """
    Factory function that lazy-initialises the graph.
    Called only by 'langgraph dev'.
    """
    try:
        logger.info("🔧 Initialising standalone graph via factory...")

        from llm_provider import get_llm_client
        from retriever import get_chroma_collection

        load_dotenv()

        _openai_client = get_llm_client()
        _collection = get_chroma_collection()

        _retriever_instance = Retriever(_collection, _openai_client)
        _agent_instance = AdvancedAgent()

        graph = create_graph(_retriever_instance, _agent_instance)

        logger.info("✅ Standalone graph loaded successfully!")
        return graph

    except Exception as e:
        logger.info("Error in get_graph factory: %s", e)
        import traceback
        traceback.print_exc()
        raise e


# ============================================================================
# MAIN - For direct testing
# ============================================================================

if __name__ == "__main__":
    logger.info("🧪 LangGraph Agent Test Mode\n")

    from llm_provider import get_llm_client
    from retriever import get_chroma_collection

    load_dotenv()

    openai_client = get_llm_client()
    collection = get_chroma_collection()

    retriever = Retriever(collection, openai_client)
    agent = AdvancedAgent()

    test_question = "What are the main topics in the documents?"

    result = run_agent(
        question=test_question,
        retriever=retriever,
        agent=agent,
        intent="SEARCH"
    )

    logger.info("\n%s", "=" * 60)
    logger.info("RESULT:")
    logger.info("%s\n", "=" * 60)
    logger.info("Answer: %s\n", result['final_answer'])
    logger.info("Facts: %d found", len(result.get('facts', [])))
    logger.info("References: %d loaded", len(result.get('references', [])))
    logger.info("Retries: %d", result.get('retry_count', 0))
    logger.info("\n%s\n", "=" * 60)
