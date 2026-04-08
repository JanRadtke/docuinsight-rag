"""
Phase 1 Tests — LLM Provider Switch
=====================================
Verifies that the provider abstraction works correctly.

Run:
    python -m pytest tests/test_phase1.py -v

Prerequisites:
    .env with OPENAI_API_KEY (for OpenAI tests)
    Ollama running locally (for Ollama tests, optional)
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_no_direct_openai_imports():
    """No src/ module (except llm_provider) should instantiate OpenAI directly."""
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    violations = []

    for filename in os.listdir(src_dir):
        if not filename.endswith('.py') or filename == 'llm_provider.py':
            continue

        filepath = os.path.join(src_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'OpenAI(' in line and 'llm_provider' not in line and '#' not in line.split('OpenAI(')[0]:
                violations.append(f"{filename}:{i}: {line.strip()}")

    assert not violations, (
        "Direct OpenAI() calls found (should use llm_provider):\n"
        + "\n".join(violations)
    )


def test_llm_provider_module_exists():
    """llm_provider.py exists and is importable."""
    import llm_provider
    assert hasattr(llm_provider, 'get_llm_client')
    assert hasattr(llm_provider, 'get_embedding_client')


def test_openai_provider():
    """OpenAI provider returns a working client."""
    os.environ['LLM_PROVIDER'] = 'openai'

    import llm_provider
    importlib.reload(llm_provider)

    client = llm_provider.get_llm_client()
    assert client is not None

    try:
        response = client.chat.completions.create(
            model=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5
        )
        assert response.choices[0].message.content is not None
    except Exception as e:
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            import pytest
            pytest.skip("No valid OPENAI_API_KEY configured")
        raise


def test_ollama_provider_config():
    """Ollama provider configures client with local URL."""
    os.environ['LLM_PROVIDER'] = 'ollama'

    import llm_provider
    importlib.reload(llm_provider)

    client = llm_provider.get_llm_client()
    assert client is not None
    assert 'localhost' in str(client.base_url) or '127.0.0.1' in str(client.base_url)


def test_guardrail_uses_provider():
    """Guardrail uses llm_provider instead of direct OpenAI client."""
    os.environ['LLM_PROVIDER'] = 'openai'
    import llm_provider
    importlib.reload(llm_provider)

    from guardrail import InputGuardrail
    guardrail = InputGuardrail()
    assert guardrail is not None


def test_router_uses_provider():
    """Router uses llm_provider instead of direct OpenAI client."""
    os.environ['LLM_PROVIDER'] = 'openai'
    import llm_provider
    importlib.reload(llm_provider)

    from agent_core import AgentRouter
    router = AgentRouter()
    assert router is not None
