import os
import json
import base64
import requests
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()

# --------------------------------------------------------
# Extractor models you selected
# --------------------------------------------------------
EXTRACTOR_MODELS = {
    "anthropic/claude-4.5-opus",
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
}

# --------------------------------------------------------
# Encode PDF → base64 data URL
# --------------------------------------------------------
def encode_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"

# --------------------------------------------------------
# Safe JSON extraction (LLMs sometimes wrap JSON)
# --------------------------------------------------------
def parse_extractor_output(raw):
    raw = raw.strip()

    # Remove code fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "").strip()

    # Direct parse attempt
    try:
        return json.loads(raw)
    except:
        pass

    # Fallback: extract JSON object inside text
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        return json.loads(raw[start:end+1])
    except:
        raise ValueError(f"Extractor returned malformed JSON:\n\n{raw}")

def call_openrouter_extractor(question, pdf_path, model_name, system_prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in environment.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Benchmark-Pipeline",
        "Content-Type": "application/json",
    }

    encoded_pdf = encode_pdf(pdf_path)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "file",
                        "file": {
                            "filename": os.path.basename(pdf_path),
                            "mime_type": "application/pdf",
                            "file_data": encoded_pdf,
                        },
                    },
                ],
            },
        ],
        "include_reasoning": False,
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    raw_output = response.json()["choices"][0]["message"]["content"]
    return parse_extractor_output(raw_output)


def call_gemini_native(question, pdf_path, system_prompt, model_name):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in environment.")
    if genai is None:
        raise ImportError("google-generativeai is not installed. Please pip install google-generativeai.")

    genai.configure(api_key=api_key)

    uploaded_file = genai.upload_file(path=pdf_path)
    try:
        gemini_model_name = model_name.split("/", 1)[1] if "/" in model_name else model_name
        model = genai.GenerativeModel(gemini_model_name)
        prompt = (
            system_prompt
            + "\n\nQuestion:\n"
            + question
            + "\nReturn JSON only."
        )

        response = model.generate_content(
            [
                {"text": prompt},
                {"file_data": {"file_uri": uploaded_file.uri, "mime_type": uploaded_file.mime_type}},
            ],
            generation_config={"response_mime_type": "application/json"}
        )
        raw_output = response.text
    finally:
        try:
            genai.delete_file(uploaded_file.name)
        except Exception:
            pass

    return parse_extractor_output(raw_output)


# --------------------------------------------------------
# Main function — send PDF + question to extractor
# --------------------------------------------------------
def ask_model(question, pdf_path, model_name="openai/gpt-5.1"):
    """
    Sends PDF + question to the extractor model.
    Requires strict JSON output:
    {
      "answer": "...",
      "rationale": "..."
    }
    """

    if model_name not in EXTRACTOR_MODELS:
        raise ValueError(
            f"Model '{model_name}' not allowed.\nAllowed = {EXTRACTOR_MODELS}"
        )

    system_prompt = (
        "You are a high-accuracy PDF extraction agent.\n"
        "A PDF (or PDF slice) is attached. Use ONLY its contents.\n"
        "Follow the user's question exactly—whether it asks for structured lists "
        "or semantic boundaries.\n\n"
        "IMPORTANT NAVIGATION RULES:\n"
        "- When the user says \"On page X\", they mean the printed number on the page "
        "(in headers/footers/corners). Ignore the PDF file index.\n"
        "- Scan visually across headers, footers, and corners to locate the page with "
        "that printed number before answering.\n"
        "- Do not assume a location (top vs. bottom); search the entire page.\n\n"
        "Output VALID JSON ONLY in this format:\n"
        "{\n"
        "  \"answer\": <value copied or derived from the PDF>,\n"
        "  \"rationale\": \"<brief explanation pointing to the relevant PDF content>\"\n"
        "}\n\n"
        "Notes:\n"
        "- \"answer\" may be a string, number, list, or object depending on the question.\n"
        "- Always include a rationale (even if brief) grounded in what you saw.\n"
        "- Do NOT add markdown, commentary, or page numbers unless they visibly appear."
    )

    if model_name.startswith("google/"):
        parsed = call_gemini_native(question, pdf_path, system_prompt, model_name)
    else:
        parsed = call_openrouter_extractor(question, pdf_path, model_name, system_prompt)

    if not isinstance(parsed, dict):
        raise ValueError(f"Extractor response must be a JSON object, got: {parsed}")

    if "answer" not in parsed or "rationale" not in parsed:
        raise ValueError(
            f"Extractor JSON missing required keys: {parsed}"
        )

    return parsed