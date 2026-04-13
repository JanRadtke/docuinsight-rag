import streamlit as st
import os
import re
import hashlib
from uuid import uuid4
from dotenv import load_dotenv
from llm_provider import get_llm_client, get_model_name, get_embedding_model
from guardrail import InputGuardrail
from agent_core import AgentRouter
from advanced_agent import AdvancedAgent
from exporter import ReportGenerator
from retriever import Retriever, get_chroma_collection, extract_filename_from_prompt, extract_filenames_from_prompt, clean_filename
from discovery import DiscoveryEngine
from agent_graph import run_agent
from app_render import render_search_result_extended, render_evidence_list, INPUT_DIR
from app_helpers import update_costs, refresh_sidebar_metrics, retrieve_multiple_documents, retrieve_intro_only, get_document_page_count, retrieve_full_document

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DocuInsight",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .streamlit-expanderHeader { font-weight: bold; }
    /* Disable auto-scroll-to-bottom after agent completes */
    .stChatFloatingInputContainer ~ div { overflow-anchor: none !important; }
    [data-testid="stChatMessageContainer"] { overflow-anchor: none !important; }
    /* Prevent overscroll bounce that blocks upward scrolling */
    html, body, [data-testid="stChatMessageContainer"], .main, .block-container { overscroll-behavior: none !important; }
</style>
""", unsafe_allow_html=True)

# --- SETUP ---
load_dotenv()

# --- IMPORTANT: Initialize Messages early (before sidebar rendering) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "openai_client" not in st.session_state:
    st.session_state.openai_client = get_llm_client()

if "chroma_collection" not in st.session_state:
    st.session_state.chroma_collection = get_chroma_collection()

if "guardrail" not in st.session_state:
    st.session_state.guardrail = InputGuardrail()

if "router" not in st.session_state:
    st.session_state.router = AgentRouter()

if "agent" not in st.session_state:
    st.session_state.agent = AdvancedAgent()

if "exporter" not in st.session_state:
    st.session_state.exporter = ReportGenerator()

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever(
        st.session_state.chroma_collection,
        st.session_state.openai_client
    )

if "discovery" not in st.session_state:
    st.session_state.discovery = DiscoveryEngine(
        st.session_state.retriever,
        st.session_state.agent
    )

if "stats" not in st.session_state:
    st.session_state.stats = {"total_cost": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

CHAT_MODEL = get_model_name()
EMBEDDING_MODEL = get_embedding_model()

# --- SIDEBAR ---
with st.sidebar:
    st.title("DocuInsight")
    pdf_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]) if os.path.exists(INPUT_DIR) else []
    pdf_count = len(pdf_files)
    model_name = get_model_name()
    provider = os.getenv("LLM_PROVIDER", "openai").capitalize()
    st.caption(f"v0.1 · {provider}: `{model_name}` · {pdf_count} PDFs")

    if pdf_files:
        with st.expander(f"Indexed Documents ({pdf_count})", expanded=False):
            for f in pdf_files:
                name = f.rsplit('.', 1)[0]  # strip .pdf
                if len(name) > 35:
                    name = name[:32] + "..."
                st.caption(name)

    st.markdown("---")

    # Developer Mode Toggle
    dev_mode = st.toggle("Deep Logs", value=True, help="Shows all internal agent logs (planner, retrieval, quality gate, reader, writer). When enabled, all log sections are expanded by default. Do not toggle while a query is running — it will interrupt the agent.")

    # Token metrics (compact) + Reset
    st.session_state.token_display = st.empty()
    if st.button("Reset Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.report_cache = None
        st.session_state.thread_id = str(uuid4())
        st.rerun()

    st.markdown("---")

    # Export (placeholder — filled at end of script after messages are updated)
    st.subheader("Export")
    st.session_state._export_placeholder = st.container()

    # Batch Discovery
    st.markdown("---")
    with st.expander("Batch Discovery", expanded=False):
        st.caption("Autonomous batch analysis of all PDFs")
        if pdf_count == 0:
            st.warning("No PDFs in input/ folder.")
        else:
            st.info(f"{pdf_count} documents ready")
            if st.button("Start Discovery Mode", type="secondary", use_container_width=True):
                with st.status("Batch Discovery running...", expanded=True) as status:
                    try:
                        results = st.session_state.discovery.run_batch_discovery(
                            progress_callback=lambda msg: status.write(msg),
                            max_pages_per_doc=3
                        )
                        status.update(label="Batch Discovery complete!", state="complete", expanded=False)
                        st.session_state.batch_discovery_report = st.session_state.discovery.export_results_to_markdown(results)
                        st.session_state.batch_discovery_stats = results.get('stats', {})
                        st.success(f"{results.get('stats', {}).get('successful', 0)} documents analysed!")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(f"Discovery Error: {str(e)}")

        if "batch_discovery_report" in st.session_state:
            st.download_button(
                label="Download Report (.md)",
                data=st.session_state.batch_discovery_report,
                file_name="Batch_Discovery_Report.md",
                mime="text/markdown",
                use_container_width=True
            )
            if "batch_discovery_stats" in st.session_state:
                stats = st.session_state.batch_discovery_stats
                st.caption(f"{stats.get('successful', 0)}/{stats.get('total_docs', 0)} successful")

# Initial loading of metrics at startup
refresh_sidebar_metrics()

# --- MAIN CHAT ---
st.header("DocuInsight: Document Analysis")

# --- NIGHT SHIFT RESULT DISPLAY ---
if "batch_discovery_report" in st.session_state:
    with st.expander("Batch Discovery Report", expanded=False):
        st.markdown(st.session_state.batch_discovery_report)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete Report", key="clear_batch_discovery"):
                del st.session_state.batch_discovery_report
                if "batch_discovery_stats" in st.session_state:
                    del st.session_state.batch_discovery_stats
                st.rerun()
        with col2:
            st.download_button(
                label="As Markdown",
                data=st.session_state.batch_discovery_report,
                file_name="Batch_Discovery_Report.md",
                mime="text/markdown",
                key="download_batch_discovery_main"
            )

# Display messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "extended_data" in msg:
            gr = msg["extended_data"].get("guardrail_reason")
            if dev_mode and gr:
                st.success(f"Guardrail: **ALLOWED** — *{gr}*")
            render_search_result_extended(
                result=msg["extended_data"],
                dev_mode=dev_mode,
                msg_id=f"hist_{idx}"
            )
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. GUARDRAIL
    is_blocked, block_reason = st.session_state.guardrail.check(prompt)

    if is_blocked:
        intent = "OFF_TOPIC"
    else:
        # 2. ROUTER (with chat history for follow-up detection)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-4:]
            if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
        ] or None
        intent = st.session_state.router.decide_intent(prompt, chat_history=chat_history)

    with st.chat_message("assistant"):

        if dev_mode and not is_blocked and block_reason:
            st.success(f"Guardrail: **ALLOWED** — *{block_reason}*")
            st.session_state._guardrail_reason = block_reason

        if intent == "OFF_TOPIC":
            with st.expander("Guardrail Decision", expanded=dev_mode):
                st.markdown(f"**Input:** `{prompt}`")
                st.markdown("**Status:** `BLOCKED`")
                st.error(f"**Reason:** {block_reason or 'No relevance to documents detected'}")
                st.caption("The Guardrail checks: Does the query relate to the documents?")

            response = "**Query Blocked**\n\n"
            response += "Apologies, but as a document analysis assistant I do not answer questions on this topic"

            if block_reason:
                response += f" (*{block_reason}*)."
            else:
                response += "."

            response += "\n\nI am happy to assist you with questions regarding your uploaded documents."

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            update_costs(prompt, response)

        elif intent == "CHAT":
            system_instruction = """
            You are 'DocuInsight', an AI-powered document analysis assistant.

            YOUR PERSONALITY:
            - Professional, precise, but helpful.
            - You are NOT a general chatbot.

            YOUR CAPABILITIES:
            - If the user says "Hello": Briefly introduce yourself and offer help with document analysis.
            - If the user asks about your capabilities: Explain that you can read documents, analyse diagrams, create summaries and compare documents.

            LANGUAGE:
            - Always respond in the same language the user writes in.
            """

            history = st.session_state.messages[-10:]

            stream = st.session_state.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "system", "content": system_instruction}] + history,
                stream=True
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

            full_input_str = system_instruction + str(history)
            update_costs(full_input_str, response)

        elif intent == "SUMMARIZE":
            # UI-specific orchestration: SUMMARIZE and COMPARE need interactive
            # progress updates, file-download buttons, and page-count display that
            # the stateless LangGraph pipeline (run_agent) cannot provide.
            # console_chat.py routes all intents through run_agent instead.
            with st.status("Agent summarizing...", expanded=True) as status:
                status.write("Extracting filenames from prompt...")
                filename = extract_filename_from_prompt(prompt)

                if filename:
                    page_count = get_document_page_count(filename)
                    status.write(f"Document found: {filename} ({page_count} pages)")

                    if page_count > 80:
                        status.write("Large document (>80 pages). Using Smart Sampling...")
                        context, refs, citation_map = retrieve_full_document(
                            filename=filename,
                            include_images=False,
                            structured=True,
                            intro_pages=5,
                            middle_pages=20,
                            discussion_pages=5
                        )
                    else:
                        status.write("Standard size. Loading full document...")
                        context, refs, citation_map = retrieve_full_document(
                            filename=filename,
                            include_images=False,
                            structured=False
                        )
                else:
                    status.write("No specific document detected. Loading intro/outro of all documents...")
                    context, refs, citation_map = retrieve_full_document(
                        filename=None,
                        include_images=False,
                        structured=True,
                        intro_pages=2,
                        discussion_pages=2
                    )

                est_tokens = len(context) / 4
                if est_tokens > 100000:
                    status.write(f"Context very large ({int(est_tokens)} tokens). Truncating automatically...")
                    context = context[:350000]

                if not context.strip():
                    response = "❌ No documents found or the filter was too strict."
                else:
                    status.write("Drafting structured summary...")
                    try:
                        response = st.session_state.agent.summarize(context)
                        update_costs(context + prompt, response)
                        status.update(label="Summary complete.", state="complete", expanded=False)
                    except Exception as e:
                        response = f"❌ Error during AI query: {str(e)}"
                        status.update(label="❌ Error", state="error")

            st.markdown(response)
            msg_data = {"role": "assistant", "content": response}
            if refs:
                msg_data["extended_data"] = {"references": refs, "final_answer": response}
            st.session_state.messages.append(msg_data)

            render_evidence_list(refs, response, key_prefix="sum")

        elif intent == "COMPARE":
            criteria = []
            refs = []
            with st.status("Comparison Agent analysing...", expanded=True) as status:
                status.write("Extracting filenames from prompt...")
                filenames = extract_filenames_from_prompt(prompt, max_files=3)

                if not filenames or len(filenames) < 2:
                    if filenames is None:
                        response = "❌ Could not identify any documents in the prompt. Please mention the documents by name (e.g. 'Compare Report_A with Report_B')."
                    elif len(filenames) == 1:
                        response = f"❌ Only one document found ({clean_filename(filenames[0])}). I need at least two documents for a comparison."
                    else:
                        response = "❌ Could not identify two clear documents for comparison."
                    status.update(label="❌ Insufficient Information", state="error")
                else:
                    status.write(f"Loading {len(filenames)} documents...")

                    # STEP 1: CRITERIA SCOUT
                    status.write("Step 1: Identifying comparison criteria...")

                    intro_contexts = retrieve_intro_only(filenames)

                    criteria = st.session_state.agent.identify_comparison_criteria(
                        intro_contexts, num_documents=len(filenames)
                    )

                    if criteria:
                        status.write(f"Strategy defined: {', '.join(criteria)}")
                    else:
                        status.write("No specific criteria found. Using default.")

                    intro_text_combined = "\n".join(intro_contexts.values())
                    if criteria:
                        update_costs(intro_text_combined + prompt, str(criteria))

                    # STEP 2: FULL LOAD
                    status.write("Step 2: Loading full documents...")
                    context, refs, citation_map = retrieve_multiple_documents(filenames, total_token_budget=100000)

                    if not context.strip():
                        response = "❌ Error loading documents."
                        status.update(label="❌ Error", state="error")
                    else:
                        # STEP 3: TARGETED COMPARISON & VERIFICATION
                        status.write("Step 3: Generating comparison & verifying citations...")
                        try:
                            response = st.session_state.agent.compare_documents(
                                context,
                                num_documents=len(filenames),
                                criteria=criteria
                            )

                            update_costs(context + prompt, response)
                            status.update(label="Comparison complete.", state="complete", expanded=False)
                        except Exception as e:
                            response = f"❌ Error: {str(e)}"
                            status.update(label="❌ Error", state="error")

            if criteria and refs:
                with st.expander("Analysis Details", expanded=False):
                    st.markdown("**Internal Process Steps:**")
                    st.text(f"1. Scout: Identified {len(criteria)} criteria.")
                    st.text(f"2. Context: Loaded {len(refs)} references.")
                    st.text("3. Reasoning: Generated comparison (Temp 0.1).")
                    st.text("4. Truth Layer: Checked citations against document IDs.")
                    st.markdown("**Criteria used:**")
                    for c in criteria:
                        st.caption(f"• {c}")

            st.markdown(response)
            msg_data = {"role": "assistant", "content": response}
            if refs:
                msg_data["extended_data"] = {"references": refs, "final_answer": response}
            st.session_state.messages.append(msg_data)

            render_evidence_list(refs, response, key_prefix="cmp")

        else:  # SEARCH INTENT

            status_container = st.empty()
            with status_container.status("LangGraph Agent working...", expanded=True) as status:

                result = {}
                try:
                    result = run_agent(
                        question=prompt,
                        retriever=st.session_state.retriever,
                        agent=st.session_state.agent,
                        intent="SEARCH",
                        thread_id=st.session_state.thread_id
                    )

                    if not result["success"]:
                        status.update(label="❌ Error", state="error")

                except Exception as e:
                    status.update(label="❌ System Crash", state="error")
                    result = {
                        "success": False,
                        "error": str(e),
                        "logs": [f"❌ Critical Error: {str(e)}"],
                        "references": [],
                        "final_answer": f"Error: {str(e)}"
                    }

            # Remove the status widget on success (no empty clickable expander)
            if result.get("success"):
                status_container.empty()

            if result.get("success"):
                # Persist guardrail reason so it renders from history
                if hasattr(st.session_state, '_guardrail_reason'):
                    result["guardrail_reason"] = st.session_state._guardrail_reason
                    del st.session_state._guardrail_reason

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["final_answer"],
                    "extended_data": result
                })

                accumulated_refs = result.get("references", [])
                estimated_input_tokens = len(accumulated_refs) * 800
                update_costs("", result["final_answer"], input_tokens=estimated_input_tokens)

                # Render inline (no st.rerun — avoids forced scroll-to-bottom)
                render_search_result_extended(
                    result=result,
                    dev_mode=dev_mode,
                    msg_id=f"live_{len(st.session_state.messages)}"
                )

            else:
                err_msg = result.get("error", "Unknown Error")
                st.error(f"❌ An error occurred: {err_msg}")

                if result.get("logs"):
                    with st.expander("Show Error Logs"):
                        for log in result["logs"]:
                            st.text(log)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {err_msg}"
                })

# --- FILL EXPORT PLACEHOLDER (runs after all messages are final) ---
if "_export_placeholder" in st.session_state:
    with st.session_state._export_placeholder:
        if len(st.session_state.messages) > 0:
            try:
                first_user_message = next((msg['content'] for msg in st.session_state.messages if msg['role'] == 'user'), None)
                report_title = (first_user_message[:60] + "...") if first_user_message and len(first_user_message) > 60 else (first_user_message or "DocuInsight Report")

                last_msg_hash = hashlib.md5(str(st.session_state.messages[-1]['content']).encode()).hexdigest()[:8]
                cache_key = f"{len(st.session_state.messages)}_{last_msg_hash}"

                if not st.session_state.get("report_cache") or st.session_state.get("report_cache_key") != cache_key:
                    report_file = st.session_state.exporter.create_word_report(
                        title=report_title,
                        messages=st.session_state.messages
                    )
                    st.session_state.report_cache = report_file.getvalue()
                    st.session_state.report_cache_key = cache_key

                safe_filename = re.sub(r'[^\w\s-]', '', report_title[:40])
                safe_filename = re.sub(r'[-\s]+', '-', safe_filename) or "DocuInsight_Report"

                st.download_button(
                    label="Report as Word (.docx)",
                    data=st.session_state.report_cache,
                    file_name=f"{safe_filename}_{len(st.session_state.messages) // 2}_QA.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="export_word"
                )
                report_size_kb = len(st.session_state.report_cache) / 1024
                n_queries = sum(1 for m in st.session_state.messages if m['role'] == 'user')
                st.caption(f"{n_queries} Q&A · ~{report_size_kb:.1f} KB")

            except ImportError:
                st.error("python-docx not installed.")
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")
        else:
            st.caption("Research first, then export.")
