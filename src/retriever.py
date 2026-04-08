"""
DocuInsight Retriever Module (v0.2)
====================================
ChromaDB-based hybrid retrieval (Vector + BM25) for:
- app.py (Streamlit UI)
- discovery.py (Batch Discovery)
- agent_graph.py (LangGraph Agent)

Architecture:
- Retriever class holds ChromaDB Collection + OpenAI Client
- Hybrid search: ChromaDB vector similarity + BM25 keyword matching
- Results merged via Reciprocal Rank Fusion (RRF)
- Can be initialised with st.session_state OR standalone
"""

import os
import re
import logging
import chromadb
from rank_bm25 import BM25Okapi
from llm_provider import get_embedding

logger = logging.getLogger("docuinsight")

# Constants
INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
COLLECTION_NAME = "docuinsight"

# Retrieval tuning
RRF_K = 60                  # Reciprocal Rank Fusion constant (standard value from Cormack et al.)
HYBRID_FETCH_FACTOR = 2     # fetch N × top_k candidates before fusion/reranking

# Shared stopwords for filename extraction (DE + EN)
_FILENAME_STOPWORDS = {
    "die", "der", "das", "und", "oder", "ist", "sind", "war", "waren",
    "eine", "ein", "einen", "eines", "einer", "einem",
    "den", "dem", "des", "zu", "zur", "zum", "von", "vom", "für", "mit", "ohne",
    "über", "unter", "auf", "in", "an", "aus", "bei",
    "fasse", "zusammen", "zusammenfassung", "studie", "studien",
    "paper", "dokument", "pdf", "file", "datei", "analyse", "analysiere",
    "mir", "mich", "dir", "dich", "uns", "euch", "bitte",
    "the", "a", "an", "and", "or", "is", "are", "of", "to", "for", "with",
    "summarize", "summary", "study", "studies", "document",
    "vergleiche", "zwischen", "unterschied", "unterschiede", "gegenüber",
}


class Retriever:
    """
    Central retrieval class for DocuInsight.
    Hybrid retrieval: ChromaDB vector search + BM25 keyword matching.
    """

    def __init__(self, chroma_collection, openai_client) -> None:
        """
        Args:
            chroma_collection: ChromaDB Collection instance
            openai_client: OpenAI Client instance
        """
        self.collection = chroma_collection
        self.openai_client = openai_client
        self._bm25_index = None
        self._bm25_ids = None
        self._bm25_docs = None
        self._bm25_metas = None
        self._cross_encoder = None
        self._nltk_available = None  # None = not tried, True/False = result
        self._stemmers = {}
        self._stopword_sets = {}

    _LANG_MAP = {
        "de": "german", "en": "english", "fr": "french", "es": "spanish",
        "it": "italian", "pt": "portuguese", "nl": "dutch", "sv": "swedish",
        "no": "norwegian", "da": "danish", "fi": "finnish", "ru": "russian",
    }

    def _tokenize(self, text: str, language: str = "en") -> list[str]:
        """Language-aware tokenization with stemming and stopword removal.
        Falls back to simple whitespace split if NLTK is unavailable."""
        if self._nltk_available is None:
            try:
                import nltk
                nltk.download("stopwords", quiet=True)
                self._nltk_available = True
            except (ImportError, LookupError):
                self._nltk_available = False
                logger.warning("NLTK unavailable — falling back to simple tokenization")

        tokens = text.lower().split()

        if not self._nltk_available:
            return tokens

        lang_name = self._LANG_MAP.get(language, "english")

        if lang_name not in self._stemmers:
            from nltk.stem import SnowballStemmer
            self._stemmers[lang_name] = SnowballStemmer(lang_name)

        if lang_name not in self._stopword_sets:
            from nltk.corpus import stopwords
            self._stopword_sets[lang_name] = set(stopwords.words(lang_name))

        stemmer = self._stemmers[lang_name]
        stops = self._stopword_sets[lang_name]
        return [stemmer.stem(t) for t in tokens if t not in stops]

    def get_document_languages(self) -> list:
        """Returns the unique languages across all documents in ChromaDB.
        Reads from metadata['language'] set during ingestion.
        Falls back to ['en'] if no language metadata exists."""
        try:
            results = self.collection.get(
                where={"type": "parent"},
                include=["metadatas"]
            )
            languages = set()
            for meta in results["metadatas"]:
                lang = meta.get("language")
                if lang:
                    languages.add(lang)
            return sorted(languages) if languages else ["en"]
        except (KeyError, ValueError):
            return ["en"]

    def _build_bm25_index(self) -> None:
        """Builds BM25 index from all chunks in ChromaDB (lazy, built once)."""
        if self._bm25_index is not None:
            return

        results = self.collection.get(
            where={"$or": [{"type": "chunk"}, {"type": "image"}]},
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            self._bm25_index = None
            return

        self._bm25_ids = results["ids"]
        self._bm25_docs = results["documents"]
        self._bm25_metas = results["metadatas"]

        # Tokenize: language-aware stemming + stopword removal (falls back to simple split)
        tokenized = [
            self._tokenize(doc, meta.get("language", "en"))
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        self._bm25_index = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int = 10, query_lang: str = "en") -> list[tuple]:
        """Returns top_k BM25 results as list of (id, doc, metadata, score)."""
        self._build_bm25_index()
        if self._bm25_index is None:
            return []

        tokenized_query = self._tokenize(query, query_lang)
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    self._bm25_ids[idx],
                    self._bm25_docs[idx],
                    self._bm25_metas[idx],
                    scores[idx]
                ))
        return results

    def _get_embedding(self, text: str) -> list[float]:
        """Delegates to the central get_embedding() in llm_provider.py."""
        return get_embedding(text, client=self.openai_client)

    def _hybrid_search(self, question: str, top_k: int = 10) -> list[tuple]:
        """
        Hybrid search: Vector (ChromaDB) + BM25 keyword matching.
        Results merged via Reciprocal Rank Fusion (RRF).

        Returns:
            List of (doc_id, content, metadata) tuples, ranked by RRF score.
        """
        # --- Vector search ---
        vector = self._get_embedding(question)
        vector_results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k * HYBRID_FETCH_FACTOR,  # fetch more for better fusion
            where={"$or": [{"type": "chunk"}, {"type": "image"}]},
            include=["documents", "metadatas"]
        )

        vector_ranks = {}
        vector_data = {}
        if vector_results["ids"][0]:
            for rank, doc_id in enumerate(vector_results["ids"][0]):
                vector_ranks[doc_id] = rank + 1
                vector_data[doc_id] = (
                    vector_results["documents"][0][rank],
                    vector_results["metadatas"][0][rank]
                )

        # --- BM25 search ---
        bm25_hits = self._bm25_search(question, top_k=top_k * HYBRID_FETCH_FACTOR)

        bm25_ranks = {}
        bm25_data = {}
        for rank, (doc_id, doc, meta, score) in enumerate(bm25_hits):
            bm25_ranks[doc_id] = rank + 1
            bm25_data[doc_id] = (doc, meta)

        # --- Reciprocal Rank Fusion ---
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 1.0 / (RRF_K + vector_ranks[doc_id])
            if doc_id in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[doc_id])
            rrf_scores[doc_id] = score

        # Sort by RRF score, take top candidates for reranking
        rerank_k = min(top_k * HYBRID_FETCH_FACTOR, len(rrf_scores))
        ranked_ids = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:rerank_k]

        candidates = []
        for doc_id in ranked_ids:
            if doc_id in vector_data:
                content, meta = vector_data[doc_id]
            else:
                content, meta = bm25_data[doc_id]
            candidates.append((doc_id, content, meta))

        # --- Cross-Encoder Reranking (optional, requires PyTorch >= 2.4) ---
        if candidates and len(candidates) > 1:
            try:
                if self._cross_encoder is None:
                    from sentence_transformers import CrossEncoder
                    self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

                # Score each (question, chunk_preview) pair
                pairs = [(question, c[1][:512]) for c in candidates]
                ce_scores = self._cross_encoder.predict(pairs)

                # Sort by cross-encoder score descending
                scored = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
                candidates = [c for c, s in scored[:top_k]]
            except ImportError:
                logger.warning("sentence-transformers not installed — pip install -e '.[ml]'")
                candidates = candidates[:top_k]
            except Exception as e:
                logger.error("Cross-encoder reranking failed: %s", e)
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        return candidates

    def _multilingual_hybrid_search(self, queries: dict, top_k: int = 10) -> list:
        """
        Cross-lingual hybrid search: runs one Vector+BM25 pass per language in `queries`,
        accumulates RRF scores across all passes, and returns the top_k results.

        Documents found in multiple language passes score higher (cross-lingual RRF property).

        Args:
            queries: ISO 639-1 lang code → search phrase, e.g. {"en": "CBT therapy", "de": "KVT"}
            top_k:   Number of final results to return.

        Returns:
            List of (doc_id, content, metadata) tuples sorted by accumulated RRF score.
            Same structure as _hybrid_search() — callers are interchangeable.
        """
        all_rrf_scores: dict = {}
        all_data: dict = {}

        for lang, query in queries.items():
            # --- Vector search filtered by language ---
            try:
                vector = self._get_embedding(query)
                vec_results = self.collection.query(
                    query_embeddings=[vector],
                    n_results=top_k * HYBRID_FETCH_FACTOR,
                    where={"$and": [
                        {"$or": [{"type": "chunk"}, {"type": "image"}]},
                        {"language": lang}
                    ]},
                    include=["documents", "metadatas"]
                )
                for rank, doc_id in enumerate(vec_results["ids"][0]):
                    all_rrf_scores[doc_id] = all_rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                    if doc_id not in all_data:
                        all_data[doc_id] = (
                            vec_results["documents"][0][rank],
                            vec_results["metadatas"][0][rank]
                        )
            except Exception as e:
                logger.warning("CrossLingualRRF: Vector search failed for lang=%s: %s", lang, e)

            # --- BM25 search (global index, language-aware stemming matches same-language chunks) ---
            bm25_hits = self._bm25_search(query, top_k=top_k * HYBRID_FETCH_FACTOR, query_lang=lang)
            for rank, (doc_id, doc, meta, score) in enumerate(bm25_hits):
                all_rrf_scores[doc_id] = all_rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                if doc_id not in all_data:
                    all_data[doc_id] = (doc, meta)

        # Sort by accumulated RRF score
        rerank_k = min(top_k * HYBRID_FETCH_FACTOR, len(all_rrf_scores))
        ranked_ids = sorted(all_rrf_scores, key=lambda k: all_rrf_scores[k], reverse=True)[:rerank_k]
        candidates = [(doc_id, *all_data[doc_id]) for doc_id in ranked_ids if doc_id in all_data]

        # --- Cross-Encoder Reranking (optional, same path as _hybrid_search) ---
        primary_query = next(iter(queries.values()), "")
        if candidates and len(candidates) > 1:
            try:
                if self._cross_encoder is None:
                    from sentence_transformers import CrossEncoder
                    self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                pairs = [(primary_query, c[1][:512]) for c in candidates]
                ce_scores = self._cross_encoder.predict(pairs)
                scored = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
                candidates = [c for c, s in scored[:top_k]]
            except ImportError:
                logger.warning("sentence-transformers not installed — pip install -e '.[ml]'")
                candidates = candidates[:top_k]
            except Exception as e:
                logger.error("Cross-encoder reranking failed: %s", e)
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        return candidates

    def retrieve_knowledge(self, question: str, top_k: int = 10, multilingual_queries: dict | None = None) -> tuple[str, list[dict], dict]:
        """
        Finds chunks AND images via hybrid search (Vector + BM25).
        Loads parent context (Parent Expansion).

        Args:
            question:             User question (used for mono path + parent expansion context).
            top_k:                Number of chunks to retrieve.
            multilingual_queries: If provided and has > 1 key, activates cross-lingual RRF path.
                                  If None or single-key: uses existing _hybrid_search path.
        """
        # Route: multilingual (>1 language) vs monolingual
        if multilingual_queries and len(multilingual_queries) > 1:
            hybrid_results = self._multilingual_hybrid_search(multilingual_queries, top_k=top_k)
        else:
            hybrid_results = self._hybrid_search(question, top_k=top_k)

        context_for_llm = ""
        references: list[dict] = []
        processed_parents = set()
        citation_counter = 1

        if not hybrid_results:
            return context_for_llm, references, {}

        for doc_id, content, metadata in hybrid_results:
            display_id = citation_counter
            citation_counter += 1

            # Case 1: IMAGE
            if metadata.get("type") == "image":
                snippet = f"SOURCE_ID [{display_id}] (IMAGE-DIAGRAM):\nDocument: {metadata['source_file']} (p. {metadata['page_number']})\nContent (image description): {content}\n\n"
                context_for_llm += snippet

                references.append({
                    "id": display_id,
                    "file": metadata["source_file"],
                    "page": metadata["page_number"],
                    "text": content,
                    "type": "image",
                    "image_path": metadata.get("image_path", "")
                })

            # Case 2: TEXT CHUNK (with Parent Expansion)
            else:
                parent_id = metadata.get("parent_id", "")

                # Skip if parent page was already loaded (avoids redundant child chunks)
                if parent_id and parent_id != "root" and parent_id in processed_parents:
                    citation_counter -= 1  # reclaim unused ID
                    continue

                full_text = content
                if parent_id and parent_id != "root":
                    try:
                        parent_result = self.collection.get(
                            ids=[parent_id],
                            include=["documents"]
                        )
                        if parent_result["documents"]:
                            full_text = parent_result["documents"][0]
                            processed_parents.add(parent_id)
                    except Exception as e:
                        logger.debug("Parent chunk lookup failed for %s: %s", parent_id, e)

                snippet = f"SOURCE_ID [{display_id}]:\nDocument: {metadata['source_file']} (p. {metadata['page_number']})\nContent: {full_text}\n\n"
                context_for_llm += snippet

                references.append({
                    "id": display_id,
                    "file": metadata["source_file"],
                    "page": metadata["page_number"],
                    "text": full_text,
                    "type": "text",
                    "image_path": ""
                })

        citation_map = {i+1: i+1 for i in range(len(references))}  # identity — reserved for future renumbering
        return context_for_llm, references, citation_map

    def get_document_page_count(self, filename: str) -> int:
        """Determines the maximum page number of a document."""
        try:
            results = self.collection.get(
                where={"$and": [
                    {"source_file": filename},
                    {"type": "parent"}
                ]},
                include=["metadatas"]
            )
            if results["metadatas"]:
                pages = [m["page_number"] for m in results["metadatas"]]
                return max(pages) if pages else 0
            return 0
        except Exception as e:
            logger.debug("Page count lookup failed: %s", e)
            return 0

    def retrieve_full_document(self, filename: str | None = None, include_images: bool = False, structured: bool = False,
                               intro_pages: int = 3, discussion_pages: int = 2, middle_pages: int = 0) -> tuple[str, list[dict], dict]:
        """
        Document-level retrieval for SUMMARIZE intents.

        Args:
            filename: Filename or None for all
            include_images: Include image descriptions?
            structured: Load only intro + middle + discussion?
            intro_pages: Number of intro pages
            discussion_pages: Number of discussion pages
            middle_pages: Number of pages from the middle section
        """
        # Build filter
        where_filter = self._build_where_filter(filename, include_images)

        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            return "", [], {}

        # Prepare results as list of dicts
        results_list = []
        for i, doc_id in enumerate(results["ids"]):
            results_list.append({
                "id": doc_id,
                "content": results["documents"][i],
                "source_file": results["metadatas"][i].get("source_file", ""),
                "page_number": results["metadatas"][i].get("page_number", 0),
                "type": results["metadatas"][i].get("type", "parent"),
                "image_path": results["metadatas"][i].get("image_path", "")
            })

        results_list.sort(key=lambda x: (x.get("source_file", ""), x.get("page_number", 0)))

        # STRUCTURED RETRIEVAL
        if structured and filename:
            results_list = self._apply_structured_sampling(
                results_list, intro_pages, middle_pages, discussion_pages
            )

        # Format context
        context_for_llm = ""
        references = []
        citation_counter = 1

        for r in results_list:
            display_id = citation_counter
            citation_counter += 1

            if r["type"] == "image":
                snippet = f"SOURCE_ID [{display_id}] (IMAGE):\n{r['content']}\n\n"
                context_for_llm += snippet
                references.append({
                    "id": display_id,
                    "file": r["source_file"],
                    "page": r["page_number"],
                    "text": r["content"],
                    "type": "image",
                    "image_path": r.get("image_path")
                })
            else:
                snippet = f"SOURCE_ID [{display_id}]:\nDocument: {r['source_file']} (p. {r['page_number']})\nContent: {r['content']}\n\n"
                context_for_llm += snippet
                references.append({
                    "id": display_id,
                    "file": r["source_file"],
                    "page": r["page_number"],
                    "text": r["content"],
                    "type": "text",
                    "image_path": ""
                })

        citation_map = {i+1: i+1 for i in range(len(references))}  # identity — reserved for future renumbering
        return context_for_llm, references, citation_map

    def _build_where_filter(self, filename: str | None = None, include_images: bool = False) -> dict:
        """Builds the ChromaDB where filter."""
        if include_images:
            type_filter = {"$or": [{"type": "parent"}, {"type": "image"}]}
        else:
            type_filter = {"type": "parent"}

        if filename:
            return {"$and": [{"source_file": filename}, type_filter]}
        else:
            return type_filter

    def _apply_structured_sampling(self, results_list: list, intro_pages: int, middle_pages: int, discussion_pages: int) -> list:
        """Applies structured sampling (intro + middle + outro)."""
        pages_by_file = {}
        for r in results_list:
            file = r.get("source_file", "")
            if file not in pages_by_file:
                pages_by_file[file] = []
            pages_by_file[file].append(r)

        filtered_results = []
        for file, pages in pages_by_file.items():
            total_pages = len(pages)

            if total_pages > (intro_pages + discussion_pages + middle_pages):
                # A. Intro
                filtered_results.extend(pages[:intro_pages])

                # B. Middle section (sampling)
                if middle_pages > 0:
                    middle_start = intro_pages
                    middle_end = total_pages - discussion_pages
                    middle_candidates = pages[middle_start:middle_end]

                    if middle_candidates:
                        step = max(1, len(middle_candidates) // middle_pages)
                        sampled = middle_candidates[::step][:middle_pages]
                        filtered_results.extend(sampled)

                # C. Discussion
                filtered_results.extend(pages[-discussion_pages:])
            else:
                filtered_results.extend(pages)

        filtered_results.sort(key=lambda x: (x.get("source_file", ""), x.get("page_number", 0)))
        return filtered_results

    def retrieve_multiple_documents(self, filenames: list[str], total_token_budget: int = 100000, strategy: str = "balanced") -> tuple[str, list[dict], dict]:
        """
        Loads multiple documents with ID renumbering.

        strategy:
            - 'balanced': Intro 25%, middle 50%, discussion 25%
        """
        if not filenames:
            return "", [], {}

        safe_budget_per_file = (total_token_budget / len(filenames)) * 0.8

        all_contexts = []
        all_refs = []
        citation_offset = 0
        citation_map = {}

        for filename in filenames:
            page_count = self.get_document_page_count(filename)

            if page_count == 0:
                context, refs, _ = self.retrieve_full_document(filename=filename, structured=False)
            else:
                est_tokens = page_count * 800
                if est_tokens < safe_budget_per_file:
                    context, refs, _ = self.retrieve_full_document(filename=filename, structured=False)
                else:
                    affordable_pages = int(safe_budget_per_file / 800)
                    intro = max(2, affordable_pages // 4)
                    outro = max(2, affordable_pages // 4)
                    middle = max(2, affordable_pages // 2)

                    context, refs, _ = self.retrieve_full_document(
                        filename=filename,
                        structured=True,
                        intro_pages=intro,
                        middle_pages=middle,
                        discussion_pages=outro
                    )

            # ID renumbering — default parameter binds citation_offset by value
            def replace_id_match(match, offset=citation_offset):
                old_id = int(match.group(1))
                new_id = old_id + offset
                return f"SOURCE_ID [{new_id}]"

            context = re.sub(r'SOURCE_ID \[(\d+)\]', replace_id_match, context)

            # Adjust metadata
            start_id = citation_offset + 1
            end_id = citation_offset + len(refs)

            adjusted_refs = []
            for ref in refs:
                ref_copy = ref.copy()
                ref_copy['id'] += citation_offset
                adjusted_refs.append(ref_copy)
                citation_map[ref_copy['id']] = {
                    'file': ref_copy['file'],
                    'page': ref_copy['page'],
                    'type': ref_copy.get('type', 'text')
                }

            if refs:
                header = f"\n\n=== DOCUMENT: {filename} (IDs [{start_id}] to [{end_id}]) ===\n\n"
            else:
                header = f"\n\n=== DOCUMENT: {filename} ===\n\n"

            all_contexts.append(f"{header}{context}")
            all_refs.extend(adjusted_refs)
            citation_offset += len(refs)

        combined_context = "\n\n".join(all_contexts)
        return combined_context, all_refs, citation_map

    def retrieve_intro_only(self, filenames: list[str], intro_pages: int = 3) -> tuple[str, list[dict], dict]:
        """Loads only intro pages for the Criteria Scout."""
        intro_contexts = {}
        for filename in filenames:
            context, _, _ = self.retrieve_full_document(
                filename=filename,
                structured=True,
                intro_pages=intro_pages,
                middle_pages=0,
                discussion_pages=0
            )
            intro_contexts[filename] = context
        return intro_contexts

    def match_documents_by_entities(self, entities: list[str], intro_pages: int = 2) -> tuple[str, list[dict], dict, list[str]]:
        """
        Searches the first N pages of ALL documents for entity names (authors, terms).
        Returns list of filenames that contain at least one entity.

        Args:
            entities: List of search terms, e.g. ["Li", "Chen"]
            intro_pages: How many pages from the start to search (default: 2)

        Returns:
            List of (filename, matched_entities) tuples, sorted by match count desc
        """
        if not entities:
            return []

        # Get all unique source files
        all_results = self.collection.get(
            where={"type": "parent"},
            include=["documents", "metadatas"]
        )

        if not all_results["ids"]:
            return []

        # Group pages by file
        pages_by_file = {}
        for i, doc_id in enumerate(all_results["ids"]):
            meta = all_results["metadatas"][i]
            content = all_results["documents"][i]
            filename = meta.get("source_file", "")
            page_num = meta.get("page_number", 0)

            if filename not in pages_by_file:
                pages_by_file[filename] = []
            pages_by_file[filename].append((page_num, content))

        # Search first N pages of each file for entities
        matches = []
        for filename, pages in pages_by_file.items():
            pages.sort(key=lambda x: x[0])
            intro_text = " ".join(content for _, content in pages[:intro_pages]).lower()

            matched = []
            for e in entities:
                # Short entities (<=3 chars): word boundary match to avoid false positives
                # e.g. "Li" should not match "quality" or "clinical"
                if len(e) <= 3:
                    pattern = r'\b' + re.escape(e.lower()) + r'\b'
                    if re.search(pattern, intro_text):
                        matched.append(e)
                else:
                    if e.lower() in intro_text:
                        matched.append(e)
            if matched:
                matches.append((filename, matched))

        # Sort by number of matched entities (most matches first)
        matches.sort(key=lambda x: len(x[1]), reverse=True)
        return matches

    def get_all_documents(self) -> list[str]:
        """Returns list of all PDF files in the input folder."""
        if not os.path.exists(INPUT_DIR):
            return []
        return [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    def get_document_summary(self, filename: str, max_pages: int = 5) -> tuple[str, list[dict]]:
        """
        Fetches a brief summary of a document (for Batch Discovery).
        Loads only the first N pages.
        """
        context, refs, _ = self.retrieve_full_document(
            filename=filename,
            structured=True,
            intro_pages=max_pages,
            middle_pages=0,
            discussion_pages=0
        )
        return context, refs


# --- STANDALONE HELPER FUNCTIONS (for app.py compatibility) ---

def get_chroma_collection():
    """Creates a ChromaDB Collection instance (helper for setup)."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def extract_filename_from_prompt(prompt: str) -> str | None:
    """
    Attempts to extract a filename from the user prompt.
    """
    if not os.path.exists(INPUT_DIR):
        return None

    available_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    if not available_files:
        return None

    prompt_words = re.findall(r'\b\w+\b', prompt.lower())
    relevant_words = [w for w in prompt_words if len(w) >= 2 and w not in _FILENAME_STOPWORDS]

    if not relevant_words:
        return None

    best_match = None
    best_score = 0

    for filename in available_files:
        filename_norm = filename.lower().replace(".pdf", "").replace("_", " ")

        score = 0
        for word in relevant_words:
            if word in filename_norm:
                score += 1

        if score > 1:
            score += 0.5 * (score - 1)

        if score > best_score:
            best_score = score
            best_match = filename

    if best_score > 0:
        return best_match

    return None


def extract_filenames_from_prompt(prompt: str, max_files: int = 3) -> list[str]:
    """
    Detects MULTIPLE filenames in the user prompt.
    """
    if not os.path.exists(INPUT_DIR):
        return None

    available_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    if not available_files:
        return None

    prompt_words = re.findall(r'\b\w+\b', prompt.lower())
    relevant_words = [w for w in prompt_words if len(w) >= 2 and w not in _FILENAME_STOPWORDS]

    if not relevant_words:
        return None

    matched_files = []
    for filename in available_files:
        filename_clean = filename.lower().replace(".pdf", "").replace("_", " ")
        filename_tokens = set(re.findall(r'\b\w+\b', filename_clean))

        score = 0
        for word in relevant_words:
            if len(word) < 4:
                if word in filename_tokens:
                    score += 1
            else:
                if word in filename_clean:
                    score += 1

        if score > 0:
            matched_files.append((filename, score))

    matched_files.sort(key=lambda x: x[1], reverse=True)

    seen = set()
    unique_matched = []
    for filename, score in matched_files:
        if filename not in seen:
            seen.add(filename)
            unique_matched.append((filename, score))

    result = [f[0] for f in unique_matched[:max_files]]
    return result if result else None


def clean_filename(filename: str) -> str:
    """Cleans up filenames for display (standalone version)."""
    name = filename.replace(".pdf", "").replace("_", " ")
    if len(name) > 40:
        name = name[:37] + "..."
    return name
