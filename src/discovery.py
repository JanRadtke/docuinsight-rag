"""
DocuInsight Discovery Engine (v0.1)
====================================
Batch Discovery — Autonomous Analysis

Analyses a batch of PDFs and generates:
1. Summaries of all documents
2. Trend analysis: What are the overarching themes?
3. Insights & connections: What could be interesting?
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from llm_provider import get_llm_client, get_model_name

from advanced_agent import AdvancedAgent
from retriever import Retriever, get_chroma_collection

# Setup
load_dotenv()

MODEL_FAST = get_model_name()
MODEL_SMART = os.getenv("SMART_MODEL", "gpt-4o")


class DiscoveryEngine:
    """
    Batch Discovery — Analyses PDFs autonomously in batch mode.
    """

    def __init__(self, retriever: Retriever, agent: AdvancedAgent | None = None) -> None:
        """
        Args:
            retriever: Retriever instance for data access
            agent: Optional AdvancedAgent for more complex analyses
        """
        self.retriever = retriever
        self.agent = agent or AdvancedAgent()
        self.client = get_llm_client()

    @classmethod
    def from_env(cls) -> "DiscoveryEngine":
        """
        Factory method: Creates DiscoveryEngine from environment variables.
        Useful for standalone usage without Streamlit.
        """
        openai_client = get_llm_client()
        collection = get_chroma_collection()

        retriever = Retriever(collection, openai_client)
        agent = AdvancedAgent()

        return cls(retriever, agent)

    def run_batch_discovery(self, file_list: list[str] | None = None, progress_callback: object = None, max_pages_per_doc: int = 5) -> dict:
        """
        Analyses a list of files in batch mode.

        Args:
            file_list: List of files to analyse (None = all in input/)
            progress_callback: Callback for progress updates
            max_pages_per_doc: How many pages per document to load for summary

        Returns:
            Dict with summaries, trends, insights
        """
        if file_list is None:
            file_list = self.retriever.get_all_documents()

        if not file_list:
            return {"error": "No PDFs found in input/ folder."}

        if progress_callback:
            progress_callback(f"Batch Discovery started: {len(file_list)} documents found.")

        # PHASE 1: Batch Summarization
        summaries = []
        for i, filename in enumerate(file_list):
            if progress_callback:
                progress_callback(f"[{i+1}/{len(file_list)}] Analysing: {filename[:40]}...")

            try:
                context, refs = self.retriever.get_document_summary(filename, max_pages=max_pages_per_doc)

                if not context.strip():
                    summaries.append({
                        "file": filename,
                        "summary": "No text extractable.",
                        "status": "error"
                    })
                    continue

                summary = self._generate_quick_summary(filename, context)

                summaries.append({
                    "file": filename,
                    "summary": summary,
                    "status": "success",
                    "pages_analyzed": len(refs)
                })

            except Exception as e:
                summaries.append({
                    "file": filename,
                    "summary": f"❌ Error: {str(e)}",
                    "status": "error"
                })


        if progress_callback:
            progress_callback(f"✅ Phase 1 complete: {len([s for s in summaries if s['status'] == 'success'])}/{len(summaries)} successful.")

        # PHASE 2: Meta-analysis (Trends & Insights)
        if progress_callback:
            progress_callback("Phase 2: Generating meta-analysis (trends & insights)...")

        successful_summaries = [s for s in summaries if s['status'] == 'success']

        if len(successful_summaries) < 2:
            return {
                "summaries": summaries,
                "trends": "Too few successful analyses for trend detection.",
                "insights": None,
                "timestamp": datetime.now().isoformat()
            }

        trends_report = self._generate_trend_report(successful_summaries)
        insights = self._generate_insights(successful_summaries)

        if progress_callback:
            progress_callback("Batch Discovery complete!")

        return {
            "summaries": summaries,
            "trends": trends_report,
            "insights": insights,
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "total_docs": len(file_list),
                "successful": len(successful_summaries),
                "failed": len(file_list) - len(successful_summaries)
            }
        }

    def _generate_quick_summary(self, filename: str, context: str) -> dict:
        """Generates a quick summary of a single document."""
        system_prompt = """
        You are an analyst. Create a SHORT summary (3-5 sentences).

        FOCUS ON:
        1. Main topic / core question
        2. Method or approach (1 sentence)
        3. Key results or findings
        4. Conclusion

        FORMAT: Flowing text, no bullet points. Maximum 150 words.
        Respond in the language of the document.
        """

        response = self.client.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document: {filename}\n\nCONTENT:\n{context[:8000]}"}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content

    def _generate_trend_report(self, summaries: list[dict]) -> str:
        """Meta-analysis: What are the trends across this batch of documents?"""
        combined_text = "\n\n".join([
            f"**{s['file']}:**\n{s['summary']}"
            for s in summaries
        ])

        system_prompt = """
        You are an analytical document expert.
        Your task: Meta-analysis of a batch of documents.

        IDENTIFY:

        ### Overarching Themes
        What topics do multiple documents cover? Which themes recur?

        ### Common Patterns
        Which methods, terms, or concepts appear across documents?

        ### Notable Findings
        Is there a divergent perspective or surprising result?

        ### Potential Contradictions
        Where might documents contradict each other?

        FORMAT: Markdown with emojis. Concise and actionable.
        Respond in the language of the documents.
        """

        response = self.client.chat.completions.create(
            model=MODEL_SMART,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyse these {len(summaries)} documents:\n\n{combined_text}"}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def _generate_insights(self, summaries: list[dict]) -> str:
        """Generates insights and connections based on the documents."""
        combined_text = "\n\n".join([
            f"Document: {s['file']}\n{s['summary']}"
            for s in summaries
        ])

        system_prompt = """
        You are an analytical document expert. Based on a batch of documents, identify NEW insights and connections.

        RULES:
        1. Combine findings from DIFFERENT documents.
        2. Look for gaps: What was NOT covered?
        3. Be creative but plausible.

        FORMAT:
        ### Insight 1: [Title]
        **Based on:** Document A + Document B
        **Connection:** "If X, then Y could..."
        **Recommendation:** [Suggestion for further analysis]

        Generate 2-3 insights.
        Respond in the language of the documents.
        """

        response = self.client.chat.completions.create(
            model=MODEL_SMART,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate insights based on:\n\n{combined_text}"}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content

    def export_results_to_markdown(self, results: dict) -> str:
        """Exports Batch Discovery results as a Markdown report."""

        if not results:
            return "# Batch Discovery Report\n\nNo results available."

        report = f"""# DocuInsight Batch Discovery Report
**Generated:** {results.get('timestamp', 'N/A')}
**Documents analysed:** {results.get('stats', {}).get('total_docs', 'N/A')}

---

## Individual Summaries

"""
        for s in results.get('summaries', []):
            status_icon = "✅" if s['status'] == 'success' else "❌"
            report += f"### {status_icon} {s['file']}\n{s['summary']}\n\n"

        report += f"""---

## Trend Analysis

{results.get('trends', 'No trends identified.')}

---

## Insights & Connections

{results.get('insights', 'No insights generated.')}

---

*Generated by DocuInsight Batch Discovery v0.1*
"""
        return report


# --- CLI usage (for testing without Streamlit) ---
if __name__ == "__main__":
    print("DocuInsight Batch Discovery - Standalone Mode")
    print("=" * 50)

    try:
        engine = DiscoveryEngine.from_env()
        results = engine.run_batch_discovery(
            progress_callback=lambda msg: print(msg)
        )

        print("\n" + "=" * 50)
        print(engine.export_results_to_markdown(results))

    except Exception as e:
        print(f"❌ Error: {e}")
