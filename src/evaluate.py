"""
DocuInsight Evaluator (Parallel & Clean Ordered Logs)
======================================================
"""

import json
import os
import time
import concurrent.futures
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
from llm_provider import get_llm_client

current_dir = os.path.dirname(os.path.abspath(__file__))

# Disable logs BEFORE importing agent_graph!
os.environ["DOCUINSIGHT_LOGS"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from agent_graph import run_agent
from retriever import Retriever, get_chroma_collection
from advanced_agent import AdvancedAgent

load_dotenv()

TESTSET_PATH = os.path.join(os.path.dirname(current_dir), "data", "testset.json")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

def setup_system() -> tuple:
    openai_client = get_llm_client()
    collection = get_chroma_collection()
    retriever = Retriever(collection, openai_client)
    agent = AdvancedAgent()
    return openai_client, retriever, agent

def evaluate_single_case_llm(client, case: Dict, agent_response: str) -> Dict:
    question = case["question"]
    reference_truth = case.get("reference_truth") or case.get("ground_truth", "")
    must_contain = case.get("must_contain", [])

    judge_prompt = f"""You are a strict evaluator (LLM-as-Judge) for a RAG system.
Compare the AI answer (Candidate) against the Reference Truth.

QUESTION: "{question}"

REFERENCE TRUTH:
"{reference_truth}"

AI ANSWER:
"{agent_response}"

REQUIRED KEYWORDS (must appear or be clearly paraphrased):
{json.dumps(must_contain, ensure_ascii=False)}

EVALUATION CRITERIA:
1. Factual correctness — does the answer match the reference truth?
2. Required keywords — are all keywords present or paraphrased?
3. Hallucinations — does the answer contain unsupported claims?
4. Score 0-100.

OUTPUT FORMAT (JSON):
{{
    "score": 85,
    "reasoning": "Brief explanation of the score.",
    "missing_keywords": []
}}
"""
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge Error: {str(e)}", "missing_keywords": []}

def run_single_test(i: int, case: dict, client: object, retriever: object, agent: object) -> dict:
    """Worker: runs a test and COLLECTS logs without printing."""
    logs = []
    start_time = time.time()

    response = run_agent(
        question=case["question"],
        retriever=retriever,
        agent=agent,
        intent=case.get("intent", "SEARCH")
    )

    duration = time.time() - start_time
    agent_answer = response.get("final_answer", "No answer")
    agent_logs = response.get("logs", [])

    eval_result = evaluate_single_case_llm(client, case, agent_answer)
    score = eval_result.get("score", 0)

    status_icon = "✅" if score >= 80 else "⚠️" if score >= 50 else "❌"

    logs.append(f"\n{'='*80}")
    logs.append(f"Test {i}: {case['question'][:60]}...")
    logs.append(f"{'='*80}")

    if agent_logs:
        logs.append("--- AGENT LOGS (excerpt) ---")
        for log_line in agent_logs:
            if "PLANNER" in log_line or "ACTION" in log_line or "Score" in log_line or "Synthesiz" in log_line:
                logs.append(log_line)

    logs.append(f"\nRESULT: {status_icon} Score: {score}/100 | Time: {duration:.1f}s")
    if score < 100:
        logs.append(f"CRITIQUE: {eval_result.get('reasoning')}")
    logs.append("-" * 80)

    return {
        "id": i,
        "score": score,
        "logs": logs
    }

def run_evaluation(testset_path: str | None = None) -> None:
    client, retriever, agent = setup_system()
    testset_path = testset_path or TESTSET_PATH

    if not os.path.exists(testset_path):
        print("❌ Testset not found!")
        return

    with open(testset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\nStarting evaluation: {len(test_cases)} cases ({MAX_WORKERS} workers)...")
    print("   (Agent logs are silenced and printed per case at the end)\n")

    results = []
    total_score = 0
    start_global = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_test, i, case, client, retriever, agent): i for i, case in enumerate(test_cases, 1)}

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            total_score += res["score"]

            for line in res["logs"]:
                print(line)

    results.sort(key=lambda x: x["id"])
    duration_global = time.time() - start_global
    avg = total_score / len(test_cases) if test_cases else 0

    # Logging implementation feature request
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_evaluate.txt"
    log_filepath = os.path.join(os.path.dirname(current_dir), "data", log_filename)

    with open(log_filepath, "w", encoding="utf-8") as file:
        file.write(f"DocuInsight Evaluation Report - {timestamp}\n")
        file.write(f"{'#'*60}\n\n")

        for case, res in zip(test_cases, results):
             file.write(f"TEST {res['id']}\n")
             file.write(f"Prompt (Question): {case.get('question')}\n\n")
             file.write(f"Reference Truth:\n{case.get('reference_truth')}\n\n")
             eval_out = ""
             for log in res["logs"]:
                 if "CRITIQUE" in log or "RESULT" in log:
                    eval_out += log + "\n"

             # Need the final answer from original case
             file.write("Agent Output (LLM Result):\n")
             agent_flag = False
             agent_txt = ""
             for val in res["logs"]:
                 if agent_flag:
                     agent_txt += val + "\n"
                 if "RESULT" in val:
                     agent_flag = False
                 if "AGENT LOGS" in val:
                     agent_flag = True

             file.write(agent_txt)
             file.write("\nEvaluation Results:\n" + eval_out)
             file.write(f"\n{'-'*80}\n\n")

        file.write(f"\n{'#'*60}\n")
        file.write(f"OVERALL SCORE: {avg:.1f}/100 (completed in {duration_global:.1f}s)\n")
        file.write(f"{'#'*60}\n")

    print(f"\n{'#'*60}")
    print(f"OVERALL SCORE: {avg:.1f}/100 (completed in {duration_global:.1f}s)")
    print(f"Detailed report saved to: {log_filepath}")
    print(f"{'#'*60}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=None, help="Path to testset JSON file")
    args = parser.parse_args()
    run_evaluation(testset_path=args.testset)
