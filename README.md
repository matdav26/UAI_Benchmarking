# Two-Track Evaluation Framework for Multimodal LLMs

This repository contains the official implementation of the **Two-Track Evaluation Framework**, designed to distinguish **Structural Fidelity** (Vision) from **Contextual Reasoning** (Brain) in frontier Multimodal Large Language Models (MLLMs).

This framework was used to evaluate **GPT-5.1**, **Gemini 3 Pro**, and **Claude 4.5 Opus** on the *Microsoft 2025 Environmental Sustainability Report*.

## 📂 Repository Structure

```text
.
├── data/
│   ├── documents/          # Source PDFs (e.g., microsoft_esg_2025.pdf)
│   ├── ground_truth/       # Golden Set JSONL files (Track A & B)
│   └── questions/          # Question sets for evaluation
├── scripts/
│   ├── evaluation_pipeline.py  # Main entry point for running benchmarks
│   ├── ask_model.py            # API wrappers for OpenAI, Anthropic, Gemini
│   ├── score_answer.py         # LLM-as-a-Judge logic (DeepSeek-v3.2)
│   └── calc_scores.py          # Aggregation logic for final percentages
└── results/                    # Output logs and scored JSONL files
