"""
Integration Tests — Mock-based tests for core pipeline components.
No API key required. Tests verify wiring, not LLM quality.
"""

from unittest.mock import MagicMock, patch
import json


class MockChoice:
    def __init__(self, content):
        self.message = MagicMock(content=content)


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


def _mock_client(response_text):
    """Creates a mock OpenAI client that returns a fixed response."""
    client = MagicMock()
    client.chat.completions.create.return_value = MockResponse(response_text)
    return client


class TestRouterIntents:
    """Tests that the router correctly classifies intents with exact matching."""

    def _make_router(self, llm_response):
        from agent_core import AgentRouter
        with patch("agent_core.get_llm_client", return_value=_mock_client(llm_response)):
            router = AgentRouter()
        router.client = _mock_client(llm_response)
        return router

    def test_search_intent(self):
        router = self._make_router("SEARCH")
        assert router.decide_intent("What is rapamycin?") == "SEARCH"

    def test_compare_intent(self):
        router = self._make_router("COMPARE")
        assert router.decide_intent("Compare paper A and B") == "COMPARE"

    def test_summarize_intent(self):
        router = self._make_router("SUMMARIZE")
        assert router.decide_intent("Summarize the document") == "SUMMARIZE"

    def test_chat_intent(self):
        router = self._make_router("CHAT")
        assert router.decide_intent("Hello") == "CHAT"

    def test_invalid_intent_falls_back_to_search(self):
        router = self._make_router("SEARCH_AND_COMPARE")
        assert router.decide_intent("Do something") == "SEARCH"

    def test_whitespace_handling(self):
        router = self._make_router("  compare  \n")
        assert router.decide_intent("Compare X and Y") == "COMPARE"


class TestGuardrail:
    """Tests that the guardrail correctly blocks/allows with mock LLM."""

    def _make_guardrail(self, status, reason="test reason"):
        from guardrail import InputGuardrail
        response_json = json.dumps({"status": status, "reason": reason})
        with patch("guardrail.get_llm_client", return_value=_mock_client(response_json)):
            guardrail = InputGuardrail()
        guardrail.client = _mock_client(response_json)
        return guardrail

    def test_allowed_passes(self):
        guardrail = self._make_guardrail("ALLOWED", "Medical question")
        is_blocked, reason = guardrail.check("What are the side effects?")
        assert not is_blocked
        assert reason == "Medical question"

    def test_blocked_stops(self):
        guardrail = self._make_guardrail("BLOCKED", "Consumer advice")
        is_blocked, reason = guardrail.check("Best phone to buy?")
        assert is_blocked
        assert reason == "Consumer advice"

    def test_llm_error_fails_open(self):
        from openai import APIError
        from guardrail import InputGuardrail
        with patch("guardrail.get_llm_client", return_value=MagicMock()):
            guardrail = InputGuardrail()
        guardrail.client = MagicMock()
        guardrail.client.chat.completions.create.side_effect = APIError(
            message="API down", request=MagicMock(), body=None
        )
        is_blocked, reason = guardrail.check("Anything")
        assert not is_blocked  # fail-open


class TestGraphCompilation:
    """Tests that the LangGraph compiles without errors."""

    def test_graph_compiles(self):
        from agent_graph import create_graph
        retriever = MagicMock()
        agent = MagicMock()
        graph = create_graph(retriever, agent)
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from agent_graph import create_graph
        retriever = MagicMock()
        agent = MagicMock()
        graph = create_graph(retriever, agent)
        node_names = set(graph.get_graph().nodes.keys())
        assert "planner" in node_names
        assert "quality_check" in node_names


class TestQualityGateFallback:
    """Tests the quality gate fallback when LLM fails."""

    def test_fallback_returns_borderline_score(self):
        from agent_graph import node_quality_check
        state = {
            "question": "What is X?",
            "context_text": "A" * 5000,
            "retry_count": 0,
            "entity_match": False,
        }
        with patch("llm_provider.get_llm_client") as mock:
            mock.return_value.chat.completions.create.side_effect = Exception("LLM down")
            result = node_quality_check(state)
        assert result["quality_score"] == 0.6
        assert not result["needs_recursion"]

    def test_empty_context_triggers_retry(self):
        from agent_graph import node_quality_check
        state = {
            "question": "What is X?",
            "context_text": "",
            "retry_count": 0,
            "entity_match": False,
        }
        result = node_quality_check(state)
        assert result["quality_score"] == 0.0
        assert result["needs_recursion"] is True
