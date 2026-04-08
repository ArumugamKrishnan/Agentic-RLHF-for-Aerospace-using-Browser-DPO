Agentic RLHF for Aerospace Engineering using Browser + DPO

This project demonstrates a complete Agentic RLHF pipeline for aerospace engineering tasks by combining:

🌐 Browser-based knowledge retrieval
👥 Human preference style data generation
🧠 Direct Preference Optimization (DPO)
📊 Base vs Fine-Tuned model evaluation

The goal is to train a small language model to behave like an Aerospace Engineering Assistant that gives structured, regulation-aware, engineering-quality responses.

🚀 Motivation

Generic LLMs produce:

Hallucinated aviation regulations
Repetitive, non-technical answers
Poor engineering checklist quality

This project shows how to align a model using aerospace web knowledge + preference learning, without expensive human labeling.

🧠 True RLHF Logic Implemented
Browser Search
      ↓
Aerospace Text Extraction
      ↓
Preference Pair Creation (chosen vs rejected)
      ↓
DPO Training
      ↓
Evaluation (Base vs Fine-tuned)

🛠️ Tech Stack
Component	Tool
Model	Qwen 0.5B
Training Method	DPO (Direct Preference Optimization)
Data Source	Live browser search (SerpAPI + newspaper3k)
Framework	HuggingFace TRL
Environment	Google Colab Pro
Quantization	4-bit (bitsandbytes)
Evaluation	Keyword engineering scoring

📂 Notebook Workflow
Step 1 — Install Dependencies

Installs TRL, Transformers, SerpAPI, newspaper3k, bitsandbytes.

Step 2 — Browser Search Function

Uses SerpAPI to fetch aerospace links and extracts article text.

Step 3 — Create Preference Dataset

For each aerospace prompt:

Chosen → Web-based summarized answer
Rejected → Generic useless answer

Saved as:

browser_aerospace_dataset.json

Step 4 — DPO Fine-Tuning

Model: Qwen/Qwen1.5-0.5B

Trained using:

DPOTrainer

Output:

final_dpo_Aerospace_agent

Step 5 — Base vs Fine-Tuned Evaluation

Both models are tested on aerospace prompts such as:

FAA UAV certification
Composite inspection checklist
BVLOS rules
Structural validation
MIL wiring standards

Step 6 — Engineering Quality Scoring

A keyword-based metric evaluates technical relevance:

keywords = ["FAA","EASA","NASA","inspection","checklist",
            "composite","airworthiness","structural","MIL","standard"]

Step 7 — Numerical Summary

Example output:

Average Base Model Score: 2.00
Average Fine-Tuned Model Score: 1.60

This shows that naive DPO with small data is insufficient, highlighting the need for:

Reward model + larger dataset → true RLHF

This is an important research finding of the project.

Step 8 — Export Results

Saved to:

evaluation_results.json
📊 Key Observation (Important for Report)

Even after DPO:

Model still hallucinates
Needs reward model for true alignment
Needs larger aerospace dataset

This validates the need for full RLHF pipeline:

Browser → Preference pairs → Reward Model → DPO → Evaluate
🧪 How to Run (Reproducibility)
Open the Colab notebook

Add your SerpAPI key:

SERP_API_KEY = "YOUR_KEY"
Run all cells sequentially
Wait for DPO training (~30–60 mins on Colab Pro)
Evaluation results will print automatically
Download:
evaluation_results.json
final_dpo_Aerospace_agent

📁 Files Generated
File	Purpose
browser_aerospace_dataset.json	Preference dataset
final_dpo_Aerospace_agent/	Fine-tuned model
evaluation_results.json	Model comparison results
🎯 What This Project Proves

✅ How to build dataset using browser
✅ How to create preference pairs automatically
✅ How to fine-tune with DPO
✅ How to evaluate LLM for engineering quality
✅ Why reward model is required for real RLHF

🔮 Future Work (True RLHF Extension)
Train a Reward Model on preference pairs
Use reward inside DPO
Increase aerospace dataset size (1000+ prompts)
Add regulation PDFs and standards
Deploy as Aerospace Engineering Agent
👨‍💻 Author

M.Tech Capstone Project — Agentic RLHF for Aerospace Engineering

🧾 Citation / Concept References
RLHF & Preference Learning
DPO (Direct Preference Optimization)
Agentic AI using browser tools
Aerospace engineering alignment of LLMs
✅ Final Outcome

This repository demonstrates a practical, reproducible Agentic RLHF pipeline and highlights the gap between naive DPO and true RLHF, which is the core research contribution of this project.
