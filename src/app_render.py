"""
Glass Box rendering for DocuInsight search results.

Extracted from app.py to keep the main Streamlit entrypoint manageable.
"""

import os
import re
import streamlit as st
from retriever import clean_filename

INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")


def render_search_result_extended(result: dict, dev_mode: bool, msg_id: str) -> None:
    """Renders Glass Box Logs + Sources with Download Buttons."""

    all_logs = result.get("logs", [])
    facts = result.get("facts", [])
    accumulated_refs = result.get("references", [])
    quality_score = result.get("quality_score", 1.0)
    final_answer = result.get("final_answer", "")

    # Build renumber map: internal chunk IDs → sequential display numbers (1, 2, 3...)
    _cited_raw = sorted(set(int(x) for x in re.findall(r'\[(\d+)\]', final_answer)))
    _renumber = {old: new for new, old in enumerate(_cited_raw, start=1)}

    def _renumbered(text: str) -> str:
        """Replace internal chunk IDs with sequential display numbers."""
        if not _renumber:
            return text
        return re.sub(r'\[(\d+)\]', lambda m: f"[{_renumber[int(m.group(1))]}]" if int(m.group(1)) in _renumber else m.group(0), text)

    final_answer_display = _renumbered(final_answer)

    # --- 1. SPLIT LOGS INTO "ROUNDS" ---
    rounds = []
    current_round = []

    for log in all_logs:
        current_round.append(log)
        if "ACTION:" in log and "Retry" in log:
            rounds.append(current_round)
            current_round = []

    if current_round:
        rounds.append(current_round)

    # --- 2. HEADER ---
    st.markdown("---")
    if dev_mode:
        header = "### Deep Dive Logs"
        if len(rounds) > 1:
            header += f" — {len(rounds)} Iterations"
        st.markdown(header)
    else:
        st.markdown("### Thought Process")

    # --- 3. TABS FOR ATTEMPTS (if Retries) ---
    if len(rounds) > 1:
        tab_labels = []
        for i in range(len(rounds)):
            if i < len(rounds) - 1:
                tab_labels.append(f"Attempt {i+1} (discarded)")
            else:
                tab_labels.append(f"Attempt {i+1} (final)")
        tabs = st.tabs(tab_labels)
    else:
        tabs = [st.container()]

    # --- 4. RENDER LOOP PER ROUND ---
    for round_idx, (tab, round_logs) in enumerate(zip(tabs, rounds)):
        with tab:
            is_final_round = (round_idx == len(rounds) - 1)
            expand_default = dev_mode

            planner_logs = [entry for entry in round_logs if "PLANNER" in entry or "QUESTION:" in entry]
            original_logs = [entry for entry in round_logs if "_ORIGINAL:" in entry]
            optimized_logs = [entry for entry in round_logs if "_QUERY:" in entry and "_ORIGINAL" not in entry]
            found_logs = [entry for entry in round_logs if "_FOUND:" in entry]
            chunk_logs = [entry for entry in round_logs if "_CHUNKS:" in entry]
            agg_logs = [entry for entry in round_logs if "AGGREGATOR" in entry]
            quality_logs = [entry for entry in round_logs if "QUALITY" in entry or "REASON" in entry or "ACTION:" in entry]
            reader_logs = [entry for entry in round_logs if "READER" in entry or "FACTS_PREVIEW" in entry or "facts extrac" in entry.lower()]
            writer_logs = [entry for entry in round_logs if "WRITER" in entry]

            if planner_logs and round_idx == 0:
                with st.expander("**Strategy** (Planner)", expanded=expand_default):
                    for entry in planner_logs:
                        if "PLANNER:" in entry:
                            st.info(entry.replace("PLANNER: ", ""))
                        elif "QUESTION:" in entry:
                            st.caption(entry)

            if original_logs or optimized_logs or found_logs:
                with st.expander("**Research**", expanded=expand_default):
                    if original_logs:
                        st.markdown("**Original Query:**")
                        seen = set()
                        for entry in original_logs:
                            content = entry.split(': ', 1)[1] if ': ' in entry else entry
                            if content not in seen:
                                st.code(content.strip("'"), language=None)
                                seen.add(content)

                    if optimized_logs:
                        st.markdown("**Optimised Queries:**")
                        for entry in optimized_logs:
                            if "FACTS_QUERY" in entry:
                                match = re.search(r"FACTS_QUERY: '([^']+)'", entry)
                                if match:
                                    st.code(f"Facts: {match.group(1)}", language=None)
                            elif "CONCEPTS_QUERY" in entry:
                                match = re.search(r"CONCEPTS_QUERY: '([^']+)'", entry)
                                if match:
                                    st.code(f"Concepts: {match.group(1)}", language=None)

                    if found_logs:
                        st.markdown("**Results:**")
                        seen = set()
                        for entry in found_logs:
                            if entry not in seen:
                                st.caption(entry)
                                seen.add(entry)

                    if chunk_logs:
                        for entry in chunk_logs:
                            if "FACTS_CHUNKS" in entry:
                                raw = entry.replace("FACTS_CHUNKS: ", "")
                                st.code("Facts:\n" + raw.replace("  ", "\n"), language=None)
                            elif "CONCEPTS_CHUNKS" in entry:
                                raw = entry.replace("CONCEPTS_CHUNKS: ", "")
                                st.code("Concepts:\n" + raw.replace("  ", "\n"), language=None)

            if agg_logs:
                with st.expander("**Synthesis**", expanded=expand_default):
                    seen = set()
                    for entry in agg_logs:
                        if entry in seen:
                            continue
                        seen.add(entry)
                        if entry.startswith("AGGREGATOR_CHUNKS:"):
                            raw = entry.replace("AGGREGATOR_CHUNKS: ", "")
                            st.code("All Chunks (deduplicated):\n" + raw.replace("  ", "\n"), language=None)
                        else:
                            st.text(entry)

            if quality_logs:
                local_score = 0.0
                for entry in quality_logs:
                    match = re.search(r"Score (\d+\.?\d*)", entry)
                    if match:
                        local_score = float(match.group(1))
                        break

                is_retry = any("Retry" in entry for entry in quality_logs)
                is_best_effort = any("Best Effort" in entry for entry in quality_logs)

                icon = "✅" if local_score >= 0.6 else "⚠️"  # semantic status icons
                title = f"{icon} **Relevance Check** (Score: {local_score:.2f})"
                if is_retry:
                    title += " — Retry required"
                elif is_best_effort:
                    title += " — Best Effort"

                with st.expander(title, expanded=True):
                    for entry in quality_logs:
                        if "QUALITY_CHECK:" in entry:
                            if local_score >= 0.6:
                                st.success(entry)
                            else:
                                st.warning(entry)
                        elif "REASON:" in entry:
                            st.info(f"**Reason:** {entry.split(':', 1)[1].strip()}")
                        elif "ACTION:" in entry:
                            if "Retry" in entry:
                                st.error(entry)
                            elif "Best Effort" in entry:
                                st.warning(entry)
                            else:
                                st.success(entry)

            if reader_logs and is_final_round:
                with st.expander(f"**Extraction** ({len(facts)} facts)", expanded=expand_default):
                    for entry in reader_logs:
                        if "FACTS_PREVIEW" in entry:
                            lines = entry.replace("FACTS_PREVIEW:\n", "").split("\n")
                            for line in lines:
                                if line.strip():
                                    st.caption(line.strip())
                        else:
                            st.text(entry)

            if writer_logs and is_final_round:
                with st.expander("**Drafting**", expanded=expand_default):
                    for entry in writer_logs:
                        if "WRITER_SEES:" in entry:
                            lines = entry.replace("WRITER_SEES:\n", "").split("\n")
                            for line in lines:
                                if line.strip():
                                    st.caption(line.strip())
                        else:
                            st.text(entry)

    # --- 4b. RENUMBERING NOTE (dev mode only) ---
    if dev_mode and _renumber:
        mapping_str = "  ".join(f"[{old}] → [{new}]" for old, new in sorted(_renumber.items()))
        with st.expander("**Source Renumbering** (Display only)", expanded=False):
            st.caption(
                "The internal chunk IDs (assigned during retrieval) are renumbered for display. "
                "Logs above show original IDs, the answer and source list use display IDs."
            )
            st.code(mapping_str)

    st.markdown("---")

    # --- 5. WARNING ON LOW SCORE ---
    if quality_score < 0.6:
        st.error(f"Retrieved information may not be specific enough (Score: {quality_score:.2f}). The following answer is provided on a 'Best Effort' basis.")

    # --- 6. FINAL ANSWER ---
    st.markdown(final_answer_display)

    # --- 7. SOURCES WITH DOWNLOADS ---
    if accumulated_refs:
        # Filter to only cited sources, then assign sequential display numbers
        seen_ids = set()
        unique_refs = []
        for r in accumulated_refs:
            if r['id'] not in seen_ids and (_cited_raw == [] or r['id'] in _cited_raw):
                unique_refs.append(r)
                seen_ids.add(r['id'])

        if not unique_refs:
            # Fallback: show all if verifier removed all citations
            for r in accumulated_refs:
                if r['id'] not in seen_ids:
                    unique_refs.append(r)
                    seen_ids.add(r['id'])

        # Assign sequential display IDs to all filtered refs
        unique_refs = [dict(r, display_id=i) for i, r in enumerate(unique_refs, start=1)]

        st.markdown(f"<br><hr><h4>Verified Evidence ({len(unique_refs)} sources):</h4>", unsafe_allow_html=True)

        for r in unique_refs:
            clean_file = clean_filename(r['file'])
            marker = f"[{r['display_id']}]"
            full_text = r.get('text', '')
            text_length = len(full_text)

            type_label = "[IMG]" if r.get('type') == 'image' else ""
            with st.expander(f"**{marker} {type_label} {clean_file} (P. {r['page']})** ({text_length} chars)", expanded=False):
                if r.get('type') == 'image':
                    if r.get('image_path') and os.path.exists(r['image_path']):
                        st.image(r['image_path'])
                    st.caption(f"**AI Analysis:** {full_text}")
                else:
                    st.markdown("**Full Context:**")
                    st.text_area("Context", value=full_text, height=150, disabled=True, key=f"ta_{msg_id}_{r['display_id']}", label_visibility="collapsed")

                c1, c2 = st.columns(2)
                with c1:
                    clean_text = re.sub(r'[^\S\n]{2,}', ' ', full_text)
                    txt_data = f"=== SOURCE [{r['display_id']}] ===\nFile: {r['file']}\nPage: {r['page']}\n\n{clean_text}"
                    st.download_button(
                        "Download TXT",
                        data=txt_data,
                        file_name=f"Source_{r['display_id']}_{clean_file}_P{r['page']}.txt",
                        mime="text/plain",
                        key=f"dl_txt_{msg_id}_{r['display_id']}",
                        use_container_width=True
                    )
                with c2:
                    file_path = os.path.join(INPUT_DIR, os.path.basename(r['file']))
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            st.download_button(
                                "Download PDF",
                                data=f.read(),
                                file_name=r['file'],
                                mime="application/pdf",
                                key=f"dl_pdf_{msg_id}_{r['display_id']}",
                                use_container_width=True
                            )

        # --- ALL SOURCES AS ONE FILE ---
        st.markdown("---")
        all_sources_txt = f"""===============================================
DOCUINSIGHT - ALL SOURCES FOR THIS QUERY
===============================================
Generated: {msg_id}
Total sources: {len(unique_refs)}

This file contains ALL texts that were passed to the AI as input.
===============================================

"""
        for r in unique_refs:
            all_sources_txt += f"""
{'='*50}
SOURCE [{r['display_id']}]
{'='*50}
File: {r['file']}
Page: {r['page']}
Type: {r.get('type', 'text')}
Chars: {len(r.get('text', ''))}
{'='*50}

{re.sub(r'[^\S\n]{2,}', ' ', r.get('text', '[No Text]'))}

"""

        st.download_button(
            f"Download All Sources as TXT ({len(unique_refs)} sources)",
            data=all_sources_txt,
            file_name=f"All_Sources_{msg_id}.txt",
            mime="text/plain",
            key=f"dl_all_{msg_id}",
            use_container_width=True
        )


def render_evidence_list(refs: list, response: str, key_prefix: str) -> None:
    """Renders cited sources with TXT/PDF download buttons.

    Shared by SUMMARIZE and COMPARE intent handlers in app.py.
    The SEARCH intent uses render_search_result_extended() instead
    (which includes Glass Box logs and renumbering).
    """
    if not refs:
        return

    st.markdown("<br><hr><h4 style='margin-bottom:10px;'>Verified Evidence:</h4>", unsafe_allow_html=True)

    displayed_refs = [r for r in refs if f"[{r['id']}]" in response]
    if not displayed_refs:
        displayed_refs = refs[:3]

    for r in displayed_refs:
        clean_file = clean_filename(r['file'])
        marker = f"[{r['id']}]"
        title = f"**{marker} {clean_file} (P. {r['page']})**"

        with st.expander(title, expanded=False):
            full_text = r.get('text', '')
            st.markdown("**Full Context:**")
            st.text_area("Context", value=full_text, height=150, disabled=True, key=f"ta_{key_prefix}_{r['id']}", label_visibility="collapsed")

            c1, c2 = st.columns(2)
            with c1:
                clean_text = re.sub(r'[^\S\n]{2,}', ' ', full_text)
                txt_data = f"=== SOURCE [{r['id']}] ===\nFile: {r['file']}\nPage: {r['page']}\n\n{clean_text}"
                st.download_button(
                    "Download TXT",
                    data=txt_data,
                    file_name=f"Source_{r['id']}_{clean_file}_P{r['page']}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{key_prefix}_{r['id']}",
                    use_container_width=True
                )
            with c2:
                file_path = os.path.join(INPUT_DIR, os.path.basename(r['file']))
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "Download PDF",
                            data=f.read(),
                            file_name=r['file'],
                            mime="application/pdf",
                            key=f"dl_pdf_{key_prefix}_{r['id']}",
                            use_container_width=True
                        )

    # --- ALL SOURCES AS ONE FILE ---
    st.markdown("---")
    all_sources_txt = f"DOCUINSIGHT - ALL SOURCES\nTotal sources: {len(displayed_refs)}\n{'='*50}\n\n"
    for r in displayed_refs:
        all_sources_txt += f"{'='*50}\nSOURCE [{r['id']}]\nFile: {r['file']}\nPage: {r['page']}\n{'='*50}\n\n{re.sub(r'[^\S\n]{2,}', ' ', r.get('text', '[No Text]'))}\n\n"

    st.download_button(
        f"Download All Sources as TXT ({len(displayed_refs)} sources)",
        data=all_sources_txt,
        file_name=f"All_Sources_{key_prefix}.txt",
        mime="text/plain",
        key=f"dl_all_{key_prefix}",
        use_container_width=True
    )
