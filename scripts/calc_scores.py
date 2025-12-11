import argparse
import os
from collections import defaultdict
import re # Import the regular expression module

from utils import load_jsonl


def load_results(paths):
    data = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        entries = load_jsonl(path)
        
        # Determine the extractor model name from the filename
        # This is robust because the metadata may not be present on error entries.
        filename = os.path.basename(path)
        # Regex to capture the model name (the part after the last underscore, before .jsonl)
        # e.g., 'anthropic-claude-4.5-opus' from '..._a_anthropic-claude-4.5-opus.jsonl'
        match = re.search(r"benchmark_([ab])_(.+?)\.jsonl$", filename, re.IGNORECASE)
        if match:
            benchmark_from_file = match.group(1).upper()
            model_name = match.group(2)
        else:
            benchmark_from_file = None
            model_name = "unknown_model"

        for entry in entries:
            # Attach the derived model name to the entry for easier grouping
            entry.setdefault("metadata", {})
            entry["metadata"].setdefault("benchmark", benchmark_from_file)
            entry["metadata"]["extractor_model"] = model_name
            data.append((path, entry))
    return data


def extract_question_score(entry):
    scores = entry.get("scores", {})
    if "question_score" in scores:
        return float(scores["question_score"])
    if "correctness_score" in scores:
        return float(scores["correctness_score"])
    return 0.0


def aggregate(paths):
    # CRITICAL CHANGE: Grouping key now includes model, doc, and benchmark
    grouped = defaultdict(lambda: defaultdict(list))

    for path, entry in load_results(paths):
        meta = entry.get("metadata", {})
        model = meta.get("extractor_model", "unknown_model") # Use the extracted model name
        doc = meta.get("document", "unknown_document")
        benchmark = meta.get("benchmark", "unknown_benchmark")
        category = meta.get("category", "Uncategorized")
        score = extract_question_score(entry)
        
        # New grouping key: (model, doc, benchmark)
        grouped[(model, doc, benchmark)][category].append(score)

    # Change naming to reflect grouping by Model, Document, and Benchmark
    model_summaries = {}
    benchmark_totals = defaultdict(list)
    model_benchmark_totals = defaultdict(list)

    for (model, doc, benchmark), category_map in grouped.items():
        category_scores = {}
        all_scores = []
        for category, values in category_map.items():
            if not values:
                continue
            category_scores[category] = sum(values) / len(values)
            all_scores.extend(values)

        if all_scores:
            doc_score = sum(all_scores) / len(all_scores)
        else:
            doc_score = 0.0

        model_summaries[(model, doc, benchmark)] = {
            "categories": category_scores,
            "functional_score": doc_score
        }
        benchmark_totals[benchmark].append(doc_score)
        model_benchmark_totals[(model, benchmark)].append(doc_score)

    global_scores = {}
    for benchmark, doc_scores in benchmark_totals.items():
        if doc_scores:
            # NOTE: Global score will now be the average score of all models on that benchmark.
            global_scores[benchmark] = sum(doc_scores) / len(doc_scores)
        else:
            global_scores[benchmark] = 0.0

    model_benchmark_scores = {}
    for key, scores in model_benchmark_totals.items():
        if scores:
            model_benchmark_scores[key] = sum(scores) / len(scores)
        else:
            model_benchmark_scores[key] = 0.0

    return model_summaries, global_scores, model_benchmark_scores


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark scores from results JSONL files."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Path(s) to results JSONL files produced by evaluation_pipeline."
    )
    args = parser.parse_args()

    model_summaries, global_scores, model_benchmark_scores = aggregate(args.results)

    print("\n=== Model Benchmark Percentage Scores ===")
    for (model, benchmark), avg_score in sorted(model_benchmark_scores.items()):
        print(f"{model} - Benchmark {benchmark}: {avg_score * 100:.1f}%")


if __name__ == "__main__":
    main()