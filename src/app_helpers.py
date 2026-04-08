"""
Helper functions for DocuInsight Streamlit app.

Cost tracking, sidebar metrics, and retrieval wrappers.
"""

import streamlit as st
from llm_provider import get_model_name

# Per-million-token pricing (input, output)
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


def refresh_sidebar_metrics():
    """Updates the compact token display in the sidebar."""
    if 'token_display' in st.session_state:
        s = st.session_state.stats
        st.session_state.token_display.caption(
            f"📊 In: **{s['total_input_tokens']:,}** · Out: **{s['total_output_tokens']:,}** · Cost: **${s['total_cost']:.4f}**"
        )


def update_costs(input_text, output_text, input_tokens=None, output_tokens=None):
    """Update token and cost estimates. Pass token counts directly when available,
    otherwise falls back to len/4 heuristic (OpenAI Tokenizer Cookbook approximation)."""
    in_tokens = input_tokens if input_tokens is not None else len(input_text) // 4
    out_tokens = output_tokens if output_tokens is not None else len(output_text) // 4
    in_price, out_price = PRICING.get(get_model_name(), (0.0, 0.0))
    cost = (in_tokens * in_price / 1_000_000) + (out_tokens * out_price / 1_000_000)
    st.session_state.stats["total_input_tokens"] += in_tokens
    st.session_state.stats["total_output_tokens"] += out_tokens
    st.session_state.stats["total_cost"] += cost
    refresh_sidebar_metrics()


def retrieve_knowledge(question):
    """Wrapper for Retriever.retrieve_knowledge()"""
    return st.session_state.retriever.retrieve_knowledge(question)


def retrieve_multiple_documents(filenames, total_token_budget=100000, strategy="balanced"):
    """Wrapper for Retriever.retrieve_multiple_documents()"""
    return st.session_state.retriever.retrieve_multiple_documents(filenames, total_token_budget, strategy)


def retrieve_intro_only(filenames, intro_pages=3):
    """Wrapper for Retriever.retrieve_intro_only()"""
    return st.session_state.retriever.retrieve_intro_only(filenames, intro_pages)


def get_document_page_count(filename):
    """Wrapper for Retriever.get_document_page_count()"""
    return st.session_state.retriever.get_document_page_count(filename)


def retrieve_full_document(filename=None, include_images=False, structured=False, intro_pages=3, discussion_pages=2, middle_pages=0):
    """Wrapper for Retriever.retrieve_full_document()"""
    return st.session_state.retriever.retrieve_full_document(
        filename, include_images, structured, intro_pages, discussion_pages, middle_pages
    )
