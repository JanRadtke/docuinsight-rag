"""
DocuInsight Agent Router (v0.1)
================================
Intent classification for document queries.

Note: Input has already been validated as permitted by the InputGuardrail.
The router does not need to handle OFF_TOPIC.
"""

import logging
from llm_provider import get_llm_client, get_model_name, LLM_ERRORS

_ROUTER_ERRORS = (*LLM_ERRORS, KeyError, ValueError)
logger = logging.getLogger("docuinsight")


class AgentRouter:
    """
    Intent router for DocuInsight.
    Classifies already-validated requests into specific intents.
    """

    def __init__(self):
        self.client = get_llm_client()
        self.model = get_model_name()
        self.system_prompt = """
        You are the intent router for 'DocuInsight', a document analysis system.
        The input has already been validated as appropriate by the guardrail.

        Your task: What exactly does the user want to do?

        CATEGORIES (check in this order):

        1. COMPARE: Compare documents or concepts
           Keywords: "compare", "difference", "similarities", "vs.", "versus"

        2. SUMMARIZE: Summarise text or a document
           Keywords: "summarise", "summary", "overview", "key points", "tldr"

        3. CHAT: Social interaction or meta questions about the system
           Keywords: "hello", "who are you", "help", "thanks", "what can you do"

        4. SEARCH: Everything else (factual questions, definitions, explanations, information lookup)
           This is the default fallback for content questions.

        IMPORTANT: If conversation history is provided and the user asks a follow-up
        question about previous answers (e.g. "explain that in more detail",
        "what about point 2?", "and what does Chen say?"), classify as SEARCH — not CHAT.
        Only classify as CHAT if it is truly social interaction unrelated to documents.

        RESPONSE: Return ONLY the category word. No sentence.
        """

    def decide_intent(self, user_query: str, chat_history: list | None = None) -> str:
        """
        Classifies a request into an intent.

        Args:
            user_query: The (already validated) user input
            chat_history: Optional list of previous messages for follow-up detection

        Returns:
            Intent as string (COMPARE, SUMMARIZE, CHAT, SEARCH)
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]

            if chat_history:
                for msg in chat_history[-4:]:
                    messages.append(msg)

            messages.append({"role": "user", "content": user_query})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )

            intent = response.choices[0].message.content.strip().upper()

            # Exact match against known intents
            valid_intents = {"COMPARE", "SUMMARIZE", "CHAT", "SEARCH"}
            if intent in valid_intents:
                return intent

            return "SEARCH"  # Default fallback

        except _ROUTER_ERRORS as e:
            logger.error("Router Error: %s", e)
            return "SEARCH"
