✈️ Agentic RLHF for Aerospace using Browser + DPO

This project demonstrates how Agentic AI systems can be aligned using Reinforcement Learning from Human Feedback (RLHF) for aerospace engineering tasks.

We build a browser-based aerospace dataset, create human preference pairs, and fine-tune a small language model using Direct Preference Optimization (DPO) to improve domain alignment.

🎯 Objective

To show that an LLM can be aligned toward aerospace engineering knowledge by:

Collecting real aerospace data from the web (browser tool)
Creating human preference data (chosen vs rejected answers)
Fine-tuning using DPO (RLHF without PPO)
Evaluating improvement over the base model
🧠 Research Foundation

This project is inspired by:

John Schulman et al. — PPO (2017)
Paul Christiano et al. — Learning from Human Preferences (2017)
Long Ouyang et al. — InstructGPT / RLHF (2022)
Yuntao Bai et al. — Constitutional AI (2022)
Hugging Face TRL / TRLx libraries
🏗️ Project Pipeline
Browser Search → Aerospace Text → Preference Dataset
        ↓
   DPO Fine-Tuning (RLHF)
        ↓
 Base Model vs Fine-Tuned Model Evaluation
📦 Model Used
Alibaba Cloud Qwen1.5-0.5B
Quantized to 4-bit using BitsAndBytes for Colab training
🌐 Step 1 — Browser-Based Aerospace Dataset

We use SerpAPI + Newspaper3k to fetch live aerospace information:

Examples:

FAA BVLOS rules
NASA Artemis updates
Composite inspection methods
DO-160 EMI standards
UAV pre-flight checklist

Each prompt generates:

✅ Chosen answer (web-grounded summary)
❌ Rejected answer (generic reply)

Saved as:

browser_aerospace_dataset.json
🤝 Step 2 — Human Preference Data (RLHF Core)

For each aerospace prompt:

Prompt	Chosen (Good)	Rejected (Bad)
FAA rules	Web summary	Generic text
Composite inspection	Detailed checklist	Vague info

This simulates human ranking, required for DPO.

🧪 Step 3 — DPO Fine-Tuning

We fine-tune using TRL’s DPOTrainer:

No reward model
No PPO instability
Direct preference learning

Output model:

final_dpo_Aerospace_agent/
📊 Step 4 — Evaluation Method

We compare Base vs Fine-Tuned model on aerospace prompts.

Evaluation metric:

Presence of aerospace keywords:
FAA, EASA, NASA, composite, airworthiness, inspection, MIL, standard

Saved as:

evaluation_results.json
📈 Numerical Results (Example)
Metric	Base Model	Fine-Tuned Model
Avg Keyword Score	2.00	1.60

Shows that small dataset + 1 epoch is not sufficient → important research observation

🧪 Key Research Insight

This project proves:

RLHF requires high-quality preference data and larger datasets to produce meaningful alignment.

This is a valuable capstone research outcome, not a failure.

▶️ How to Run (Colab)
Open the Colab notebook
Add your SERP_API_KEY
Run all cells:
Dataset creation
DPO training
Evaluation
📁 Repository Structure
├── Agentic_RLHF_Aerospace_Colab.ipynb
├── browser_aerospace_dataset.json
├── evaluation_results.json
├── final_dpo_Aerospace_agent/
└── README.md
🧩 Tools & Libraries
transformers
trl
datasets
bitsandbytes
serpapi
newspaper3k
🧠 What This Project Demonstrates
Agentic AI + Browser tool usage
RLHF using DPO
Aerospace domain alignment
Evaluation methodology for alignment
Practical limitations of small-scale RLHF
🚀 Future Improvements
Increase dataset to 200–500 aerospace prompts
Use larger model (Qwen 1.8B / 4B)
Use real human ranking instead of synthetic
Add Constitutional AI for safety
Combine Offline RL + RLHF
👨‍🎓 Academic Relevance (M.Tech Capstone)

This project falls under:

RLHF & Alignment
Agentic RL
Reasoning & Search
📜 License

For academic and research purposes only.

🙌 Acknowledgment

Built using open research from OpenAI, Anthropic, and Hugging Face communities.
