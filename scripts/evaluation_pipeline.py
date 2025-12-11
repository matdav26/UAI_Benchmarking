import os
import json
import traceback
from datetime import datetime
import argparse
from utils import load_jsonl, save_jsonl
from ask_model import ask_model
from score_answer import score_answer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ---------------------------------------------------------
# Evaluation Loop (PDF-Based)
# ---------------------------------------------------------

TRACK_CONFIG = {
    "A": {
        "questions_pattern": "{doc}_benchmark_a_vision_questions.jsonl",
        "ground_truth_pattern": "{doc}_benchmark_a_vision.jsonl",
        "label": "benchmark_a"
    },
    "B": {
        "questions_pattern": "{doc}_benchmark_b_semantic_questions.jsonl",
        "ground_truth_pattern": "{doc}_benchmark_b_semantic.jsonl",
        "label": "benchmark_b"
    }
}


def evaluate_document(doc_name, extractor_model, judge_model, benchmark_type="B"):
    """
    Evaluate a single PDF document using:
    - Extractor model (Claude 4.5 Opus, GPT-4o, GPT-5.1, etc.)
    - Judge model (DeepSeek V3.2, Grok 4.1)
    """

    benchmark_key = benchmark_type.upper()
    if benchmark_key not in TRACK_CONFIG:
        raise ValueError(f"Unknown benchmark type '{benchmark_type}'. Expected {list(TRACK_CONFIG)}")

    track_info = TRACK_CONFIG[benchmark_key]

    # Input data paths
    questions_path = os.path.join(
        DATA_DIR,
        "questions",
        track_info["questions_pattern"].format(doc=doc_name)
    )
    gt_path = os.path.join(
        DATA_DIR,
        "ground_truth",
        track_info["ground_truth_pattern"].format(doc=doc_name)
    )
    pdf_path = os.path.join(DATA_DIR, "documents", f"{doc_name}.pdf")

    # Output results path
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model_name = extractor_model.replace("/", "_")
    out_path = os.path.join(
        RESULTS_DIR,
        f"{doc_name}_{track_info['label']}_{safe_model_name}.jsonl"
    )

    print("\n--- Running Evaluation ---")
    print(f"Document:       {doc_name}")
    print(f"Extractor:      {extractor_model}")
    print(f"Judge Model:    {judge_model}")
    print(f"Benchmark:      {benchmark_key}")
    print(f"Saving to:      {out_path}\n")

    # -----------------------------------------------------
    # Load data files
    # -----------------------------------------------------
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    questions = load_jsonl(questions_path)
    ground_truth_entries = load_jsonl(gt_path)

    # Map questions → ground truth answers (no evidence field expected anymore)
    gt_map = {entry["question"].strip(): entry for entry in ground_truth_entries}

    results = []
    failures = 0

    # -----------------------------------------------------
    # Main evaluation loop
    # -----------------------------------------------------
    for idx, q in enumerate(questions, start=1):
        question_text = q["question"].strip()
        category = q.get("category", "Uncategorized")

        print(f"[{idx}/{len(questions)}] {question_text}")

        try:
            if question_text not in gt_map:
                raise KeyError(f"Missing ground truth for question:\n{question_text}")

            gt = gt_map[question_text]

            # ---------------------------------------------
            # 1) Extractor model → answer object
            # ---------------------------------------------
            model_answer_obj = ask_model(
                question=question_text,
                pdf_path=pdf_path,
                model_name=extractor_model
            )

            # ---------------------------------------------
            # 2) Judge scoring (correctness + rationale)
            # ---------------------------------------------
            score_obj = score_answer(
                model_answer_obj=model_answer_obj,
                ground_truth_obj=gt,
                judge_model=judge_model,
                benchmark_type=benchmark_key
            )

            # ---------------------------------------------
            # 3) Append results row
            # ---------------------------------------------
            results.append({
                "question": question_text,
                "model_answer": model_answer_obj,
                "scores": score_obj,
                "metadata": {
                    "document": doc_name,
                    "extractor_model": extractor_model,
                    "judge_model": judge_model,
                    "benchmark": benchmark_key,
                    "category": category,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })

        except Exception as e:
            failures += 1
            print("\n❗ ERROR evaluating question:")
            print(question_text)
            print(traceback.format_exc())

            # Store fallback error row (updated fields)
            results.append({
                "question": question_text,
                "model_answer": None,
                "scores": {
                    "correctness_score": 0.0,
                    "correctness_justification": f"Error: {str(e)}",
                    "rationale_score": 0.0,
                    "rationale_justification": f"Error: {str(e)}",
                },
                "metadata": {
                    "document": doc_name,
                    "extractor_model": extractor_model,
                    "judge_model": judge_model,
                    "benchmark": benchmark_key,
                    "category": category,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            })

        print()

    # -----------------------------------------------------
    # Save JSONL output
    # -----------------------------------------------------
    save_jsonl(out_path, results)

    print("✓ Evaluation complete.")
    print(f"✓ Saved to {out_path}")
    print(f"⚠ Total failures: {failures}\n")

    return out_path


# ---------------------------------------------------------
# Judge-only run for precomputed answers
# ---------------------------------------------------------

def evaluate_precomputed_answers(doc_name, answers_path, judge_model, benchmark_type="B"):
    """
    Evaluate an existing answers JSONL file (no extractor call).
    Each line must include:
      {
        "question": "...",
        "model_answer": {"answer": "...", "rationale": "..."},
        ...
      }
    """

    benchmark_key = benchmark_type.upper()
    if benchmark_key not in TRACK_CONFIG:
        raise ValueError(f"Unknown benchmark type '{benchmark_type}'. Expected {list(TRACK_CONFIG)}")

    track_info = TRACK_CONFIG[benchmark_key]

    gt_path = os.path.join(
        DATA_DIR,
        "ground_truth",
        track_info["ground_truth_pattern"].format(doc=doc_name)
    )

    if not os.path.exists(answers_path):
        raise FileNotFoundError(f"Answers file not found: {answers_path}")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    ground_truth_entries = load_jsonl(gt_path)
    answers_entries = load_jsonl(answers_path)

    if not answers_entries:
        raise ValueError(f"No entries found in answers file: {answers_path}")

    gt_map = {entry["question"].strip(): entry for entry in ground_truth_entries}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    answers_tag = os.path.splitext(os.path.basename(answers_path))[0]
    judge_tag = judge_model.replace("/", "_")
    out_path = os.path.join(
        RESULTS_DIR,
        f"{answers_tag}_judged_{judge_tag}.jsonl"
    )

    print("\n--- Judging Precomputed Answers ---")
    print(f"Document:       {doc_name}")
    print(f"Answers file:   {answers_path}")
    print(f"Judge Model:    {judge_model}")
    print(f"Benchmark:      {benchmark_key}")
    print(f"Saving to:      {out_path}\n")

    results = []
    failures = 0

    for idx, entry in enumerate(answers_entries, start=1):
        question_text = str(entry.get("question", "")).strip()
        print(f"[{idx}/{len(answers_entries)}] {question_text}")
        category = None

        try:
            if not question_text:
                raise ValueError("Missing question text in answers file.")
            if question_text not in gt_map:
                raise KeyError(f"Missing ground truth for question:\n{question_text}")

            gt_entry = gt_map[question_text]
            category = gt_entry.get("category", "Uncategorized")

            model_answer_obj = entry.get("model_answer")
            if model_answer_obj is None:
                # Support simplified format where answer/rationale live at top level
                model_answer_obj = {
                    "answer": entry.get("answer", ""),
                    "rationale": entry.get("rationale", "")
                }

            score_obj = score_answer(
                model_answer_obj=model_answer_obj,
                ground_truth_obj=gt_entry,
                judge_model=judge_model,
                benchmark_type=benchmark_key
            )

            metadata = entry.get("metadata", {}).copy()
            metadata.update({
                "document": doc_name,
                "judge_model": judge_model,
                "source_answers_file": answers_path,
                "benchmark": benchmark_key,
                "category": category,
                "timestamp": datetime.utcnow().isoformat()
            })

            results.append({
                "question": question_text,
                "model_answer": model_answer_obj,
                "scores": score_obj,
                "metadata": metadata
            })

        except Exception as e:
            failures += 1
            print("\n❗ ERROR judging question:")
            print(question_text)
            print(traceback.format_exc())

            if entry.get("metadata"):
                fallback_cat = entry["metadata"].get("category", "Uncategorized")
            else:
                fallback_cat = "Uncategorized"
            err_category = category or fallback_cat

            results.append({
                "question": question_text,
                "model_answer": entry.get("model_answer"),
                "scores": {
                    "correctness_score": 0.0,
                    "correctness_justification": f"Error: {str(e)}",
                    "rationale_score": 0.0,
                    "rationale_justification": f"Error: {str(e)}",
                },
                "metadata": {
                    "document": doc_name,
                    "judge_model": judge_model,
                    "source_answers_file": answers_path,
                    "benchmark": benchmark_key,
                    "category": err_category,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            })

        print()

    save_jsonl(out_path, results)

    print("✓ Judging complete.")
    print(f"✓ Saved to {out_path}")
    print(f"⚠ Total failures: {failures}\n")

    return out_path


# ---------------------------------------------------------
# Manual Run (example)
# ---------------------------------------------------------
# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmark Evaluation Pipeline")
    parser.add_argument("--doc", type=str, required=True, help="Document name key (e.g., 'microsoft_esg_2025')")
    parser.add_argument("--benchmark", type=str, required=True, choices=["A", "B"], help="Benchmark Type: 'A' (Vision) or 'B' (Semantic)")
    parser.add_argument("--extractor", type=str, default="openai/gpt-5.1", help="Extractor model name (default: openai/gpt-5.1)")
    parser.add_argument("--judge", type=str, default="deepseek/deepseek-v3.2", help="Judge model name (default: deepseek/deepseek-v3.2)")
    args = parser.parse_args()

    print(f"🚀 Starting Run: Doc={args.doc} | Benchmark={args.benchmark} | Extractor={args.extractor} | Judge={args.judge}")

    evaluate_document(
        doc_name=args.doc,
        extractor_model=args.extractor,
        judge_model=args.judge,
        benchmark_type=args.benchmark
    )