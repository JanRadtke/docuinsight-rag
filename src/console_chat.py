"""
console_chat.py - Terminal-based chat interface for DocuInsight

DESCRIPTION:
-----------
Terminal-based chat interface with conversational memory.
Uses the full LangGraph agent pipeline (run_agent) with
thread_id for multi-turn follow-up support.

USAGE:
------
1. Ingest documents:
   python src/ingest.py

2. Run script:
   python src/console_chat.py

3. Ask questions in the terminal (or type 'exit' to quit)

NOTE:
-----
For an enhanced UI with browser functionality,
agent pipeline and advanced features:

   streamlit run src/app.py
"""

from uuid import uuid4
from dotenv import load_dotenv
from llm_provider import get_llm_client, get_model_name


from retriever import Retriever, get_chroma_collection
from advanced_agent import AdvancedAgent
from agent_core import AgentRouter
from guardrail import InputGuardrail
from agent_graph import run_agent

# Setup
load_dotenv()
openai_client = get_llm_client()
collection = get_chroma_collection()
retriever = Retriever(collection, openai_client)
agent = AdvancedAgent()
router = AgentRouter()
guardrail = InputGuardrail()

CHAT_MODEL = get_model_name()


def chat_with_data():
    thread_id = str(uuid4())
    print(f"\n📄 DocuInsight Terminal (model: {CHAT_MODEL})")
    print(f"💬 Session: {thread_id[:8]}...")
    print("----------------------------------------------------")

    # Local history for Router intent classification (user perspective).
    # Graph-internal MemorySaver tracks its own state for query rewriting.
    history = []

    while True:
        question = input("\nQuestion (or 'exit'): ")
        if question.lower() in ['exit', 'quit']:
            break

        is_blocked, block_reason = guardrail.check(question)
        if is_blocked:
            print(f"\n🛡️ Blocked: {block_reason}")
            continue

        intent = router.decide_intent(question, chat_history=history[-4:] or None)

        if intent == "CHAT":
            system_prompt = """You are 'DocuInsight', an AI-powered document analysis assistant.
            If the user says "Hello": Briefly introduce yourself.
            If the user asks about capabilities: Explain document analysis features."""

            try:
                response = openai_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ]
                )
                print(f"\n🤖 {response.choices[0].message.content}")
            except Exception as e:
                print(f"❌ Error: {e}")
            continue

        print(f"🧠 Intent: {intent}")
        print("... Agent working ...")

        try:
            result = run_agent(
                question=question,
                retriever=retriever,
                agent=agent,
                intent=intent,
                thread_id=thread_id
            )

            print(f"\n{'='*60}")
            if result["success"]:
                # Show rewrite if it happened
                rewrite_logs = [entry for entry in result.get("logs", []) if "REWRITE:" in entry]
                for log in rewrite_logs:
                    print(log)

                print(f"🤖 ANSWER:\n{result['final_answer']}")
                print("-" * 60)

                refs = result.get("references", [])
                if refs:
                    print("📚 Sources:")
                    for ref in refs[:5]:
                        print(f"- {ref['file']} (p. {ref['page']})")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown')}")

            print("=" * 60)

            if result["success"]:
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": result["final_answer"]})

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat_with_data()
