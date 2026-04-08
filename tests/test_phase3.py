"""
Phase 3 Tests — README & Documentation Quality
================================================
Verifies that the README is complete and portfolio-ready.

Run:
    python -m pytest tests/test_phase3.py -v
"""

import os

ROOT = os.path.join(os.path.dirname(__file__), '..')


def test_readme_exists_and_english():
    """README exists and is written in English."""
    readme_path = os.path.join(ROOT, 'README.md')
    assert os.path.exists(readme_path)

    with open(readme_path) as f:
        content = f.read()

    assert 'Architecture' in content or 'architecture' in content
    assert 'Quick Start' in content or 'Getting Started' in content
    assert 'Features' in content or 'features' in content


def test_readme_has_architecture_diagram():
    """README contains a Mermaid diagram or architecture image."""
    readme_path = os.path.join(ROOT, 'README.md')
    with open(readme_path) as f:
        content = f.read()

    has_mermaid = '```mermaid' in content
    has_image = 'architecture' in content.lower() and ('![' in content or '<img' in content)

    assert has_mermaid or has_image, "README needs an architecture diagram (Mermaid or image)"


def test_readme_has_tradeoff_table():
    """README contains the trade-off table (Local vs Production)."""
    readme_path = os.path.join(ROOT, 'README.md')
    with open(readme_path) as f:
        content = f.read()

    assert 'Production' in content and ('Local' in content or 'MVP' in content), \
        "README needs a trade-off table (Local MVP vs Production)"


def test_readme_has_provider_switch():
    """README documents the LLM provider switch."""
    readme_path = os.path.join(ROOT, 'README.md')
    with open(readme_path) as f:
        content = f.read()

    assert 'ollama' in content.lower() or 'Ollama' in content, \
        "README must document the Ollama/OpenAI switch"


def test_no_secrets_in_repo():
    """No secrets committed to git."""
    import subprocess

    # Check .env is not tracked by git (exists locally but gitignored is fine)
    result = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=ROOT, capture_output=True, text=True
    )
    assert result.stdout.strip() == "", ".env is tracked by git — must not be committed!"

    # Scan committed files for leaked keys
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT, capture_output=True, text=True
    ).stdout.splitlines()

    for filename in tracked:
        if not filename.endswith(('.py', '.md', '.txt', '.json', '.toml', '.example')):
            continue
        filepath = os.path.join(ROOT, filename)
        try:
            with open(filepath) as f:
                content = f.read()
        except Exception:
            continue
        assert ("sk-" + "proj-") not in content, f"OpenAI key leaked in {filename}!"
        assert ("AZURE_SEARCH" + "_ADMIN_KEY") not in content or 'example' in filename, \
            f"Azure key leaked in {filename}!"


def test_env_example_has_provider_config():
    """.env.example documents the provider switch."""
    env_path = os.path.join(ROOT, '.env.example')
    assert os.path.exists(env_path)

    with open(env_path) as f:
        content = f.read()

    assert 'LLM_PROVIDER' in content
    assert 'ollama' in content.lower()
    assert 'openai' in content.lower()
