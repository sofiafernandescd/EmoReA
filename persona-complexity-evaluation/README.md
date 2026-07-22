# 🎭 Persona x Complexity Evaluation

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

This module evaluates two different personas (Academic Advisor vs. Pitch Mentor; Simple vs. Detailed) in  to derive comparative metrics.

The system processes text and audio (speech). First it generates emotion-aware feedback (`feedback_coach`), then GPT-4.1 and GPT-4o are utilized to assess the quality of that feedback (`llm_judge`).



## Project Structure

```
persona-complexity-evaluation/
├── README.md                          # This file
├── analysis/                          # Analysis scripts and notebooks
│   ├── 1_feedback_coach.ipynb        # Coach feedback analysis
│   ├── 2_llm_judges.ipynb            # LLM-based judgment evaluation
│   ├── 3_mixed_effects_model.ipynb   # Statistical mixed effects modeling
│   ├── emotion_recognition.py        # Emotion recognition utilities
│   ├── feedback_coach.py              # Coach feedback processing
│   ├── llm_judge.py                   # LLM judgment interface
│   ├── prompts.py                     # Prompt templates
│   ├── archive/                       # Archived analysis files
│   └── results/                       # Analysis results and outputs
└── data/                              # Data files
    ├── llm_judge_results_gpt41.jsonl  # GPT-4.1 judge results
    ├── llm_judge_results_gpt4o.jsonl  # GPT-4o judge results
    ├── output_predictions.jsonl       # Model predictions
    └── dataset/                       # Input dataset
```



### Analysis Notebooks

- **1_feedback_coach.ipynb**: Analyzes coaching feedback provided on persona complexity
- **2_llm_judges.ipynb**: Evaluates persona complexity using LLM judges (GPT-4.1, GPT-4o)
- **3_mixed_effects_model.ipynb**: Statistical modeling of complexity factors with mixed effects analysis

### Utilities

- **emotion_recognition.py**: Functions for emotion recognition across modalities
- **feedback_coach.py**: Processing and parsing of coaching feedback
- **llm_judge.py**: Interface for LLM-based complexity judgment
- **prompts.py**: Prompt templates for LLM evaluation

### Data Files

- `llm_judge_results_gpt41.jsonl`: LLM judge results from GPT-4.1
- `llm_judge_results_gpt4o.jsonl`: LLM judge results from GPT-4o
- `output_predictions.jsonl`: Model predictions for persona complexity
- `dataset/`: Raw input datasets for evaluation


## Environment Setup

1. Install dependencies from the parent project's requirements
2. Set up `.env` file in `analysis/` directory with necessary API keys
3. Ensure access to LLM services (GPT-4.1, GPT-4o)

## Results

Analysis results and outputs are stored in `analysis/results/` including:
- Judge agreement metrics
- Statistical model summaries
- Visualizations


## Requirements

- Dependencies from parent EmoReA project
- LLM API access (OpenAI GPT-4 models)
