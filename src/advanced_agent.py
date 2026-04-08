import json
import re
import logging
from llm_provider import get_llm_client, get_model_name

logger = logging.getLogger("docuinsight")

class AdvancedAgent:
    def __init__(self, model_name=None):
        self.client = get_llm_client()
        self.model_name = model_name or get_model_name()

        logger.info("AdvancedAgent initialised with model: %s", self.model_name)

    # =========================================================================
    # Query Optimizer & Quality Reasoner
    # =========================================================================

    def rewrite_follow_up(self, question: str, chat_history: list) -> str:
        """
        Rewrites a follow-up question into a standalone question using chat history.
        Resolves pronouns and references (e.g. "Tell me more about that" → full question).
        Returns the original question unchanged if it's already standalone or on topic switch.
        """
        if not chat_history:
            return question

        # Use last 3 turns (6 messages) for efficiency
        recent = chat_history[-6:]

        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in recent
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": """You are a query rewriter for a RAG system.

TASK: Rewrite the user's latest question into a STANDALONE question that can be understood without conversation history.

RULES:
1. Resolve pronouns ("it", "that", "they", "this") using the conversation context.
2. If the question is ALREADY standalone (no references to prior turns), return it UNCHANGED.
3. If the user switches to a completely NEW topic, return the question UNCHANGED.
4. Do NOT add information that wasn't in the conversation.
5. Return ONLY the rewritten question — no explanation, no quotes.

EXAMPLES:
History: User asked about rapamycin mechanisms
Follow-up: "What are the side effects?"
Rewritten: "What are the side effects of rapamycin?"

History: User asked about longevity biomarkers
Follow-up: "How does exercise affect aging?"
Rewritten: "How does exercise affect aging?" (standalone — new topic, unchanged)
"""},
                    {"role": "user", "content": f"CONVERSATION HISTORY:\n{history_text}\n\nLATEST QUESTION: {question}"}
                ],
                temperature=0.0
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else question
        except Exception as e:
            logger.error("Query Rewrite Error: %s", e)
            return question

    def optimize_query(self, original_query: str, intent: str = "FACTS") -> dict:
        """
        Translates and optimises the query for document retrieval.
        Prevents mixed-language queries and explains what was changed.

        Returns:
            dict: {"query": str, "reasoning": str, "original": str}
        """
        system_prompt = """
        You are an expert in Information Retrieval for document databases.
        Your task: Optimise the user query for a vector search in documents.

        RULES:
        1. TRANSLATE the query to English if the documents are likely in English. Keep the original language for non-English documents.
        2. REMOVE filler words ("What is", "Show me", "Explain").
        3. ADD relevant synonyms and related terms.
        4. Make the query SPECIFIC for the intent.

        EXAMPLES:
        Input: "What does the document say about quality management?"
        Intent: FACTS
        Output Query: "quality management methods processes standards requirements"
        Output Reasoning: "Removed filler words, added domain terms"

        Input: "How does the system work?"
        Intent: CONCEPTS
        Output Query: "system architecture functionality components workflow"
        Output Reasoning: "Optimised for concept search"

        RESPONSE FORMAT (JSON):
        {
            "query": "The optimised query",
            "reasoning": "Brief explanation (1 sentence) of what was changed"
        }
        """

        mode_instruction = "Optimise for FACTS search (numbers, definitions, details)." if intent == "FACTS" else "Optimise for CONCEPTS search (relationships, frameworks, structures)."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{mode_instruction}\n\nORIGINAL QUERY: {original_query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "query": result.get("query", original_query),
                "reasoning": result.get("reasoning", "No changes"),
                "original": original_query
            }
        except Exception as e:
            logger.error("Query Optimizer Error: %s", e)
            return {
                "query": original_query,
                "reasoning": f"Fallback (error: {str(e)[:30]})",
                "original": original_query
            }

    def expand_query_multilingual(self, question: str, doc_languages: list, intent: str = "SEARCH") -> dict:
        """
        Translates and keyword-optimises `question` into each language in `doc_languages`.
        Uses a single gpt-4o-mini call with JSON mode.

        Fast path: if only 1 language, delegates to optimize_query() — zero extra LLM call.
        Fallback: on any error, returns {"en": question} so the pipeline never aborts.

        Returns:
            dict mapping ISO 639-1 lang code → optimised search phrase.
            e.g. {"en": "CBT therapy techniques", "de": "KVT Therapie Techniken"}
        """
        if not doc_languages:
            return {"en": question}

        # Fast path: single language — no extra LLM call
        if len(doc_languages) == 1:
            lang = doc_languages[0]
            optimized = self.optimize_query(question, intent).get("query", question)
            return {lang: optimized}

        intent_hint = {
            "SEARCH": "keyword search for specific facts, definitions, numbers",
            "SUMMARIZE": "broad topic search for comprehensive overview",
            "COMPARE": "comparative analysis search terms",
            "CHAT": "general topic keywords",
        }.get(intent, "keyword search")

        system_prompt = f"""You are a multilingual search query optimizer.
Given a user question and a list of target languages, produce one keyword-optimized
search query per language. Remove filler words. Add relevant domain synonyms.
Preserve technical acronyms adapted to the target language where applicable
(e.g. "CBT" → "KVT" in German, "TCC" in Portuguese).
Search intent: {intent_hint}.

Return ONLY a JSON object with ISO 639-1 language codes as keys:
{{"<lang_code>": "<optimized query in that language>", ...}}"""

        user_msg = f"Question: {question}\nTarget languages: {doc_languages}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            # Ensure all requested languages are present; fill gaps with original question
            for lang in doc_languages:
                if lang not in result or not result[lang]:
                    result[lang] = question
            return result
        except Exception as e:
            logger.error("Multilingual Query Expansion Error: %s", e)
            return {"en": question}

    def translate_to_english(self, question: str) -> str:
        """
        Translates a question to English if it's not already English.
        Used to ensure cross-lingual extraction works (English docs + non-English query).
        Returns the original question if already English.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": """You are a translator. If the input is already in English, return it unchanged.
If it is in another language, translate it to English while preserving the exact meaning and all technical terms.
Return ONLY the translated text, nothing else."""},
                    {"role": "user", "content": question}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Translation Error: %s", e)
            return question

    def extract_entities(self, question: str) -> dict:
        """
        Extracts key entities (author names, study names, specific terms)
        from a comparison question. Used for document matching.

        Returns:
            dict: {"entities": ["Li", "Chen", ...]}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": """Extract key entity names from the user question.
Focus on: author names, study names, specific terms that identify documents.
Return ONLY short identifiers (surnames, abbreviations) — not full sentences.

RESPONSE FORMAT (JSON):
{"entities": ["Li", "Chen"]}

EXAMPLES:
"Vergleiche die Studie von Li mit der Studie von Chen" → {"entities": ["Li", "Chen"]}
"Compare the Lopez-Otin framework with Kaeberlein's findings" → {"entities": ["Lopez-Otin", "Kaeberlein"]}
"What does the rapamycin paper say vs the senolytics review?" → {"entities": ["rapamycin", "senolytics"]}
"""},
                    {"role": "user", "content": question}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error("Entity Extraction Error: %s", e)
            return {"entities": []}


    def explain_strategy(self, question: str, intent: str) -> str:
        """
        Explains WHY a particular intent/strategy was chosen.
        """
        explanations = {
            "SEARCH": f"Question '{question[:40]}...' requires fact-based search in documents. Dual-track activated: Facts + Concepts in parallel.",
            "SUMMARIZE": "Summarisation request detected. Single-track: Facts search only, then structured synthesis.",
            "COMPARE": "Comparison request detected. Focus on Concepts track for analysis.",
            "CHAT": "General conversation detected. No document retrieval needed."
        }
        return explanations.get(intent, f"Intent '{intent}' detected. Standard processing.")

    def summarize(self, context_text):
        """
        Summarises a long text into a concise executive summary.
        """
        system_prompt = """
        You are an analyst creating a structured "Executive Summary".

        Your task:
        Analyse the provided text (consisting of multiple pages/chunks with IDs [x]) and create an in-depth summary.

        RULES FOR CONTENT:
        1. Structure: Use bold headings for main topics.
        2. Depth: Go beyond the surface. Explain relationships and key findings.
        3. Neutrality: Write factually and precisely.

        RULES FOR CITATIONS (CRITICAL — MUST BE FOLLOWED):
        - The input text contains source IDs like SOURCE_ID [1], SOURCE_ID [5], SOURCE_ID [12].
        - EVERY statement you make MUST be backed by at least one of these IDs.
        - WITHOUT citations your answer is INCOMPLETE and UNUSABLE.
        - Use the IDs inline in the text, not just at the end.

        EXAMPLES OF CORRECT CITATION:
        - "The results show significant improvements [1] [2]."
        - "Methodology: Participants were randomised [4]. The control group received placebo [4]."

        IMPORTANT: If you do not use IDs, your answer will be rejected by the system. Every paragraph must contain at least one [ID].

        Respond in the language of the user.
        """

        user_prompt = f"""Please summarise the following text.

CRITICAL REQUIREMENT: You MUST back every statement with source IDs [1], [2], [3], etc. marked as "SOURCE_ID [x]" in the text.

Example of correct citation:
- "The results show an improvement [1]."
- "Methodology: Randomisation was performed [2], control group [2]."

WITHOUT source IDs your answer is incomplete!

Text:
{context_text}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def identify_comparison_criteria(self, intro_contexts, num_documents=2):
        """
        Analyses the intro/opening of documents and identifies relevant comparison criteria.

        Args:
            intro_contexts: Dict[str, str] - {filename: intro_text}
            num_documents: Number of documents

        Returns:
            List[str]: List of 3-5 comparison criteria
        """
        system_prompt = """
        You are an analyst identifying comparison criteria.

        Your task:
        Analyse the opening texts of multiple documents and identify 3-5 relevant comparison criteria
        that enable an in-depth comparison.

        Examples of good criteria:
        - "Methodology" (approach, structure, design)
        - "Key findings" (theses, results, conclusions)
        - "Target audience" (who is the document aimed at?)
        - "Data & facts" (numbers, statistics, evidence)
        - "Recommendations" (action items, next steps)

        RULES:
        - Choose criteria that are relevant for ALL documents.
        - Be specific where possible.
        - Return a JSON object: {"criteria": ["Criterion 1", "Criterion 2", ...]}
        """

        context_text = ""
        for i, (filename, intro_text) in enumerate(intro_contexts.items(), 1):
            context_text += f"\n\n=== DOCUMENT {i}: {filename} ===\n\n{intro_text}"

        user_prompt = f"""
        Analyse the following opening texts of {num_documents} documents:
        {context_text}

        Identify 3-5 relevant comparison criteria.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("criteria", [])[:5]
        except Exception as e:
            logger.error("Criteria Scout Error: %s", e)
            return []

    def verify_citation_integrity(self, draft_text, doc_ranges):
        """
        The 'Truth Layer'.
        Checks the generated text for incorrect ID assignments.
        """
        system_prompt = """
        You are a silent, precise copy editor (Citation Auditor).

        Your task:
        You receive a text and rules. You return ONLY the cleaned text.

        RULES FOR CLEANUP:
        1. Check every column in tables.
        2. Column "Document 1": Remove all IDs NOT within Document 1's range.
        3. Column "Document 2": Remove all IDs NOT within Document 2's range.
        4. Remove imprecise ranges like [1-10] if present.

        OUTPUT RULES:
        - Return ONLY the corrected text.
        - NEVER repeat the rules or the input.
        - Do NOT write any introduction ("Here is the text...").
        - Preserve markdown formatting.
        """

        rules_text = ""
        for i, (fname, start, end) in enumerate(doc_ranges, 1):
            rules_text += f"- Document {i}: Allowed IDs [{start}] to [{end}]\n"

        user_prompt = f"""
        RULES:
        {rules_text}

        TEXT TO CLEAN:
        {draft_text}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Verifier Error: %s", e)
            return draft_text

    def compare_documents(self, context_text, num_documents=2, criteria=None):
        """
        Compare multiple documents. Includes a verification step.
        """

        # 1. Extract ID ranges
        doc_ranges = re.findall(r'=== DOCUMENT: (.+?) \(IDs \[(\d+)\] to \[(\d+)\]\) ===', context_text)

        range_rules = ""
        table_rules = ""

        if doc_ranges:
            range_rules = "\nCITATION RULES PER DOCUMENT (DO NOT MIX!):\n"
            table_rules = "\nTABLE RULES (CRITICAL):\n"

            for i, (fname, start, end) in enumerate(doc_ranges, 1):
                range_rules += f"- Document {i}: Use CONCRETE IDs (e.g. [3], [12]) from range [{start}]-[{end}]. NO ranges like [1-10]!\n"
                table_rules += f"- Column 'Document {i}': Use ONLY concrete IDs [{start}] to [{end}]!\n"

            if len(doc_ranges) >= 2:
                d1_s = doc_ranges[0][1]
                d2_s = doc_ranges[1][1]
                table_rules += "\nEXAMPLE OF CORRECT TABLE (text + concrete ID):\n"
                table_rules += "  | Criterion | Document 1 | Document 2 |\n"
                table_rules += "  |-----------|------------|------------|\n"
                table_rules += f"  | Result    | Point A [{d1_s}]. | Point B [{d2_s}]. | ← CORRECT\n"

        system_prompt = """
        You are an analyst comparing multiple documents.

        Your task:
        1. Identify the main topics.
        2. Find similarities and differences.
        3. Create a structured comparative analysis.

        CITATION RULES (EXTREMELY IMPORTANT):
        - Use the ACTUAL SOURCE_ID numbers from the text.
        - ALWAYS cite with CONCRETE IDs (e.g. [3]), NEVER with ranges!

        FORMATTING RULES:
        - Use Markdown H3 (###) for headings. NEVER H1 (#) or H2 (##) — too large!
        - The conclusion at the end should be plain text, not a heading.

        TABLE RULES:
        - ALWAYS create Markdown tables.
        - CONTENT: Cells must contain TEXT, not just IDs.
        """

        if criteria:
            system_prompt += f"""
            - MUST structure by: {', '.join(criteria)}.
            - Each criterion = its own section (###) with a table.
            """
        else:
            system_prompt += "- Structure by content categories."

        user_prompt = f"""
        You have {num_documents} documents.
        {range_rules}
        {table_rules}

        CRITICAL REQUIREMENT:
        - Strictly adhere to the ID ranges.
        - Use ### for headings.
        - Write CONTENT in the tables (text + concrete ID).
        """

        if criteria:
            user_prompt += f"\nFOCUS ON: {', '.join(criteria)}\n"

        user_prompt += f"\n\nDocuments:\n{context_text}"

        # --- GENERATION (Step 1) ---
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        draft_content = response.choices[0].message.content

        # --- VERIFICATION (Step 2) ---
        if doc_ranges:
            verified_content = self.verify_citation_integrity(draft_content, doc_ranges)
            return verified_content
        else:
            return draft_content

    def extract_facts(self, question: str, raw_context: str, intent: str = None, entities: list = None):
        """
        Reader: extracts facts from document context.
        Tolerates PDF formatting artefacts and uses smart matching.
        Intent-aware: adapts extraction strategy based on query intent.
        """

        # Build intent-specific extraction guidance
        intent_guidance = ""
        if intent == "COMPARE" and entities:
            subjects = " vs ".join(entities)
            intent_guidance = f"""
COMPARE MODE — You are extracting facts for a comparison between: {subjects}
For EACH chunk, extract:
- Core focus / research question of this study
- Methodology and approach
- Key findings and conclusions
- Therapeutic implications or interventions proposed
Even if only ONE subject is discussed in this chunk, extract its key points — they will be compared with another chunk later.
Do NOT say "INSUFFICIENT" just because only one subject appears. Extract what IS there.
"""
        elif intent == "SUMMARIZE":
            intent_guidance = """
SUMMARIZE MODE — Extract the most important claims, definitions, and conclusions.
Focus on: key terms, classifications, hierarchies, and author conclusions.
Be thorough — extract MORE facts rather than fewer.
"""
        elif intent == "SEARCH":
            intent_guidance = """
SEARCH MODE — Extract specific facts, numbers, definitions, and criteria that directly answer the question.
"""

        system_prompt = f"""You are a highly precise Information Extractor.
Your goal: Extract facts from the provided document snippets that help answer the user's question.
{intent_guidance}
RULES FOR EXTRACTION:
1. **SOURCE OF TRUTH:** Extract ONLY information explicitly present in the text. Do not use outside knowledge.
2. **SMART MATCHING (CRITICAL):** The text comes from PDFs and acts like "raw data" (line breaks, hyphens, weird formatting).
   - If the user asks for a specific term and the text has a slightly different format, EXTRACT IT.
   - Do NOT reject a fact just because of formatting artifacts.
3. **ATTRIBUTION:** Maintain strict separation between different sources.
4. **DEFINITIONS & CRITERIA:** If the user asks for a concept, extract descriptions, lists of characteristics, or criteria defining that concept.

DECISION LOGIC:
A) Does the text contain RELEVANT information (even partial or badly formatted)?
   -> YES: Set "status": "SUFFICIENT" and extract facts.
   -> NO: Set "status": "INSUFFICIENT".

B) If INSUFFICIENT (RETRY STRATEGY):
   - Generate a "new_query" for the next search attempt.
   - Use synonyms and related terms.

OUTPUT FORMAT (JSON):
{{
    "status": "SUFFICIENT" | "INSUFFICIENT",
    "facts": [
        {{
            "source_id": "The numeric citation ID from the document header, e.g. 1, 2, 3",
            "fact": "The extracted information (cleaned up)",
            "quote": "The rough text passage supporting this (approximate match allowed)"
        }}
    ],
    "missing_reason": "Why is info missing? (only if INSUFFICIENT)",
    "new_query": "OPTIMIZED QUERY (for retry)"
}}

IMPORTANT: For source_id, use ONLY the numeric ID from the document header (e.g. "=== DOCUMENT: [3] file.pdf ===" → source_id: "3"). Never use filenames as source_id.
"""

        user_msg = f"""
        QUESTION: {question}

        DOCUMENT CONTEXT (excerpt):
        {raw_context}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error("Error in Reader Step: %s", e)
            return {"status": "INSUFFICIENT", "missing_reason": f"Error: {str(e)}", "facts": []}

    def draft_answer(self, question, extracted_facts, target_language=None):
        """
        The WRITER drafts the answer text.
        Soft grounding: honesty about gaps instead of silence.

        Args:
            target_language: ISO 639-1 code (e.g. "de", "fr"). When set and not "en",
                             an explicit language rule is appended to the system prompt.
                             None or "en" → existing behaviour (LLM responds in user's language).
        """

        facts_text_list = ""
        facts_list = extracted_facts.get("facts", [])

        for item in facts_list:
            facts_text_list += f"SOURCE_ID [{item['source_id']}]: {item['fact']}\n"

        system_prompt = """You are a precise analyst (like NotebookLM).
Your task: Answer the user's question based on the provided facts, with strict attribution.

GROUNDING RULES:
1. **PRIMARY SOURCE:** Every claim must cite the numeric source ID in brackets, e.g. [1], [2]. Never use filenames as citations.
2. **NUANCE AWARENESS:** If facts distinguish between groups or categories, preserve that distinction.
3. **PARTIAL ANSWERS ARE VALID:** If the facts only partially answer the question, state what IS covered and explicitly note what is missing.
   Example: "Regarding X, the sources mention Y [1]. Information on Z is not provided."
4. **NO HALLUCINATION:** Do not add information from your training data.

STRUCTURE:
- Use markdown headers (###) for sections.
- Use bullet points (-) for clarity.
- Be concise but information-dense.

OUTPUT GUIDELINES:
- If facts are sufficient: Provide a complete answer.
- If facts are sparse: Provide what you can, note gaps.
- If facts are empty: State "The provided sources contain no information on this question."
- Respond in the language of the user.
"""

        if target_language and target_language != "en":
            system_prompt += f'\nLANGUAGE RULE (MANDATORY): Write your entire response in language code "{target_language}". Do not translate source citations [1],[2],[3].'

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"QUESTION: {question}\n\nAVAILABLE FACTS (Strict Source Material):\n{facts_text_list}"}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

    def critique_draft(self, question: str, facts: list, draft: str, intent: str = None) -> dict:
        """
        Critic: fact-checks the writer's draft against extracted facts.
        Returns verdict (PASS/REVISE) with hallucination and missing-fact details.
        Fail-open: on any error, returns verdict="PASS" so the pipeline never aborts.
        """
        facts_text = ""
        for f in facts:
            facts_text += f"[{f.get('source_id', '?')}] {f.get('fact', '')}\n"

        intent_hint = ""
        if intent == "SUMMARIZE":
            intent_hint = "This is a SUMMARIZE response — check for completeness of key topics and correct prioritisation."
        elif intent == "COMPARE":
            intent_hint = "This is a COMPARE response — check that claims are attributed to the correct document/source."

        system_prompt = f"""You are a strict fact-checker for a RAG system.
You receive a DRAFT answer and the FACTS that were extracted from source documents.
Your job: verify every claim in the draft against the facts list.

{intent_hint}

CHECK EACH CLAIM IN THE DRAFT:
1. Is the claim supported by at least one fact? (semantic match allowed — paraphrases are ok)
2. If the draft cites a source ID [x], does that ID exist in the facts list?
3. Does the draft add information NOT present in any fact? → hallucination
4. Are there important facts relevant to the question that the draft omits?

RESPONSE FORMAT (JSON):
{{
    "verdict": "PASS" or "REVISE",
    "hallucinations": [
        {{"claim": "the hallucinated claim from the draft", "reason": "why this is unsupported"}}
    ],
    "missing_facts": [
        {{"source_id": "3", "fact": "the omitted fact", "reason": "why it matters for the question"}}
    ],
    "revision_instructions": "Concrete instructions for the writer. Empty string if verdict is PASS."
}}

RULES:
- verdict="PASS" if no hallucinations AND no critical missing facts.
- verdict="REVISE" if there are hallucinations OR important omissions.
- Do NOT flag stylistic issues, only factual errors.
- Minor omissions (tangential facts) should NOT trigger REVISE.
- Be conservative: when in doubt, PASS."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"QUESTION: {question}\n\nEXTRACTED FACTS:\n{facts_text}\n\nDRAFT ANSWER:\n{draft}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            # Ensure required keys exist
            return {
                "verdict": result.get("verdict", "PASS"),
                "hallucinations": result.get("hallucinations", []),
                "missing_facts": result.get("missing_facts", []),
                "revision_instructions": result.get("revision_instructions", "")
            }
        except Exception as e:
            logger.error("Critic Error (fail-open): %s", e)
            return {
                "verdict": "PASS",
                "hallucinations": [],
                "missing_facts": [],
                "revision_instructions": ""
            }

    def revise_draft(self, question: str, facts: list, draft: str, critic_feedback: str, target_language: str = None) -> str:
        """
        Reviser: rewrites the draft based on critic feedback.
        Preserves correct parts, fixes hallucinations, adds missing facts.
        Fallback: returns original draft on any error.
        """
        facts_text = ""
        for f in facts:
            facts_text += f"SOURCE_ID [{f.get('source_id', '?')}]: {f.get('fact', '')}\n"

        system_prompt = """You are a precise analyst revising your previous draft based on feedback from a fact-checker.

GROUNDING RULES (same as original writer):
1. Every claim must cite the numeric source ID in brackets, e.g. [1], [2].
2. No information from your training data — only from the provided facts.
3. Partial answers are valid — state what IS covered and note gaps.

REVISION RULES:
1. Remove or qualify any hallucinated claims identified in the feedback.
2. Incorporate missing facts where relevant, with proper [source_id] citations.
3. Preserve correct parts of the draft — do not rewrite from scratch.
4. Maintain the same structure, tone, and language as the original draft.
5. Use markdown headers (###) for sections, bullet points for clarity."""

        if target_language and target_language != "en":
            system_prompt += f'\n\nLANGUAGE RULE (MANDATORY): Write your entire response in language code "{target_language}". Do not translate source citations [1],[2],[3].'

        user_prompt = f"""QUESTION: {question}

AVAILABLE FACTS (Strict Source Material):
{facts_text}

PREVIOUS DRAFT:
{draft}

FACT-CHECKER FEEDBACK:
{critic_feedback}

Please revise the draft according to the feedback. Keep what is correct, fix what is wrong."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Reviser Error (returning original draft): %s", e)
            return draft

    def verify_citations(self, draft_answer, raw_context):
        """
        The VERIFIER: checks citations for plausibility.
        Soft check: removes only the ID, not the sentence.
        """
        system_prompt = """
        You are a fact-checker.
        Your task: Check whether citations [x] in the text are supported by the context.

        RULES:
        1. Check every citation [x].
        2. Look in the CHUNKS to see if the content is present SEMANTICALLY.
        3. It does not need to be a word-for-word match. If the meaning is correct, it is fine.
        4. IF a citation is wrong: Remove ONLY the number [x], but leave the sentence intact.
        5. Do not change anything else in the text.
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TEXT TO CHECK:\n{draft_answer}\n\nORIGINAL CHUNKS:\n{raw_context}"}
            ]
        )
        return response.choices[0].message.content

