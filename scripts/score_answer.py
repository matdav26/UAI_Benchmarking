import os
import os.path as osp
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Resolve prompts relative to this file so CLI location doesn't matter.
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
PROMPTS_DIR = osp.join(SCRIPT_DIR, "..", "prompts")

# ----------------------------------------------------
# Allowed Judge Models (STRICT)
# ----------------------------------------------------
JUDGE_MODELS = {
    "deepseek/deepseek-v3.2",
    "xai/grok-4.1",
}

# ----------------------------------------------------
# Load Benchmark-Specific Prompts
# ----------------------------------------------------
def load_prompt(filename):
    full_path = osp.join(PROMPTS_DIR, filename)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Prompt file not found: {full_path}") from err

BENCHMARK_PROMPTS = {
    "A": load_prompt("benchmark_a_correctness_prompt.txt"),
    "B": load_prompt("benchmark_b_correctness_prompt.txt"),
}

# ----------------------------------------------------
# Reasoning configuration for JUDGES
# ----------------------------------------------------
def reasoning_config_for(model_name):
    return {
        "reasoning": {"effort": "high"},
        "include_reasoning": False
    }


# ----------------------------------------------------
# Low-level: Call OpenRouter Judge Model
# ----------------------------------------------------
def call_judge_llm(system_prompt, user_prompt, model):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env")

    if model not in JUDGE_MODELS:
        raise ValueError(f"Judge model '{model}' is not supported. Allowed: {JUDGE_MODELS}")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Benchmark-Judge"
    }

    payload = {
        "model": model,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        **reasoning_config_for(model)
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Clean cases where judge wraps JSON in ``` fences
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "").strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except:
        pass

    # Try extracting JSON inside text
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start:end+1])
    except:
        raise ValueError(f"Judge returned malformed JSON:\n\n{raw}")


# ----------------------------------------------------
# High-level scoring function
# ----------------------------------------------------
def score_answer(model_answer_obj, ground_truth_obj, judge_model, benchmark_type="B"):

    benchmark_type = benchmark_type.upper()
    if benchmark_type not in BENCHMARK_PROMPTS:
        raise ValueError(f"Unknown benchmark type '{benchmark_type}'. Expected one of {list(BENCHMARK_PROMPTS)}")

    # Support both "ground_truth_answer" and "answer" field names
    gt_answer = ground_truth_obj.get("ground_truth_answer", ground_truth_obj.get("answer", ""))
    gt_rationale = ground_truth_obj.get("rationale", "")

    model_answer    = model_answer_obj.get("answer", "")
    model_rationale = model_answer_obj.get("rationale", "")

    prompt_template = BENCHMARK_PROMPTS[benchmark_type]
    user_prompt = (
        prompt_template
        .replace("{{QUESTION}}", str(ground_truth_obj.get("question", "")))
        .replace("{{GROUND_TRUTH_ANSWER}}", str(gt_answer))
        .replace("{{GROUND_TRUTH_RATIONALE}}", str(gt_rationale))
        .replace("{{MODEL_ANSWER}}", str(model_answer))
        .replace("{{MODEL_RATIONALE}}", str(model_rationale))
    )

    judge_result = call_judge_llm(
        "You are an evaluation judge.",
        user_prompt,
        judge_model
    )

    is_correct = bool(judge_result.get("is_correct"))
    has_value = bool(judge_result.get("has_value"))
    error_type = judge_result.get("error_type")
    judge_reasoning = judge_result.get(
        "judge_reasoning",
        judge_result.get("justification", "")
    )

    question_score = judge_result.get("question_score")
    if question_score is None:
        question_score = 1.0 if is_correct else 0.0
    else:
        try:
            question_score = float(question_score)
        except (TypeError, ValueError):
            question_score = 1.0 if is_correct else 0.0

    return {
        # Legacy fields maintained for downstream compatibility
        "correctness_score": question_score,
        "correctness_justification": judge_reasoning,
        "rationale_score": 1.0 if is_correct else 0.0,
        "rationale_justification": error_type or judge_reasoning,
        # New structured fields for the benchmark rulebooks
        "is_correct": is_correct,
        "has_value": has_value,
        "error_type": error_type,
        "judge_reasoning": judge_reasoning,
        "question_score": question_score
    }


# ----------------------------------------------------
# Manual Test
# ----------------------------------------------------
if __name__ == "__main__":
    fake_model_answer = {
        "answer": "Microsoft achieved carbon negativity by 2030.",
        "rationale": "Because it says so in the introduction text."
    }

    fake_gt = {
        "question": "What is Microsoft's carbon goal?",
        "ground_truth_answer": "Microsoft aims to be carbon negative by 2030.",
        "rationale": "The introduction text states the goal clearly."
    }

    print(score_answer(fake_model_answer, fake_gt, judge_model="deepseek/deepseek-v3.2"))