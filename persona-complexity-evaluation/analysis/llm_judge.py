import os
import re
import json
from litellm import completion
from tqdm import tqdm 
import pandas as pd
import json
import logging
from prompts import JUDGE_SYSTEM_PROMPT, JUDGE_PROMPT_TEMPLATE, \
                    PAIRWISE_JUDGE_SYSTEM, PAIRWISE_JUDGE_PROMPT

# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# OpenAI Setup
# ============================================================
os.environ["OPENAI_API_KEY"] = "" # your API key
API_BASE = ""
JUDGE_MODEL = "gpt-4.1"
#JUDGE_MODEL = "gpt-4o" 

# ============================================================
# Local Setup
# ============================================================
#API_BASE = "http://localhost:11434"
#JUDGE_MODEL = "ollama_chat/gemma4:e4b"

logger.info(f"Using model: {JUDGE_MODEL} with API base: {API_BASE}")

# conditions to test (2x2 factorial design - Persona x Prompt Complexity)
CONDITIONS = [
    "academic_simple",
    "academic_detailed",
    "pitch_simple",
    "pitch_detailed"
]
# metrics to be evaluated by scalar judge
SCALAR_METRICS = [
    "grounding",
    "actionability",
    "persona_consistency",
    "clarity"
]
# pairs to be evaluated by pairw3ise judge
PAIRWISE_CONDITIONS = [
    ("academic_simple", "academic_detailed"),
    ("pitch_simple", "pitch_detailed"),
    ("academic_simple", "pitch_simple"),
    ("academic_detailed", "pitch_detailed")
]

# ============================================================
# JUDGES FUNCTIONS
# ============================================================
# auxiliar function to clean judge output and extract JSON
def extract_json(text):
    # Remove markdown code blocks if they exist (```json or ```)
    text = re.sub(r"```json|```", "", text)
    return text.strip()

# scalar judge function
def run_scalar_judge(model, emotion_timeline, output, persona):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        emotion_timeline=emotion_timeline,
        model_output=output,
        persona=persona
    )
    logger.info(prompt)

    res = completion(
        model=model,
        api_base=API_BASE,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    raw_content = res.choices[0].message.content
    cleaned_content = extract_json(raw_content)
    logger.info(f"Judge response received. Cleaned content: \n{cleaned_content}")

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        
        logger.info(f"Failed to parse JSON. Raw output: {raw_content}")
        return {
            "grounding": 0.0, 
            "actionability": 0.0, 
            "persona_consistency": 0.0, 
            "clarity": 0.0, 
            "explanation": "Parsing Error"
        }

# pairwise judge function
def run_pairwise(model, emotion_timeline, a, b):
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        emotion_timeline=emotion_timeline,
        response_a=a,
        response_b=b
    )
    logger.info(prompt)

    res = completion(
        model=model,
        api_base=API_BASE,
        messages=[
            {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    raw_content = res.choices[0].message.content
    cleaned_content = extract_json(raw_content)

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        logger.info(f"Failed to parse JSON. Raw output: {raw_content}")
        return {
            "better_grounding": "", 
            "better_actionability": "", 
            "better_clarity": "", 
            "better_overall": "", 
            "explanation": "Parsing Error"
        }

# ============================================================
# EXPERIMENT FUNCTIONS
# ============================================================

def evaluate_row(row):
    outputs = {
        "academic_simple": row["academic_simple"],
        "academic_detailed": row["academic_detailed"],
        "pitch_simple": row["pitch_simple"],
        "pitch_detailed": row["pitch_detailed"]
    }

    scalar_results = {}
    pairwise_results = []

    # scalar evaluation
    for cond, text in outputs.items():
        persona_type = "Academic" if "academic" in cond else "Pitch" 

        scalar_results[cond] = run_scalar_judge(
            JUDGE_MODEL,
            row["emotion_timeline"],
            text,
            persona_type
        )

    # pairwise comparisons
    for a, b in PAIRWISE_CONDITIONS:
        pairwise_results.append(
            run_pairwise(
                JUDGE_MODEL,
                row["emotion_timeline"],
                outputs[a],
                outputs[b]
            )
        )

    return {
        "scalar": scalar_results,
        "pairwise": pairwise_results
    }


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def run_experiment(input_file, output_file):
    df = pd.read_excel(input_file)

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        try:
            result = evaluate_row(row)
            record = {
                "index": idx,
                "event": row["event"],
                "title": row["title"],
                "category": row["category"],
                "scalar": result["scalar"],
                "pairwise": result["pairwise"]
            }
            # save in every iteration to avoid data loss in case of crashes
            append_jsonl(output_file, record)

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            continue

# example usage
if __name__ == "__main__":

    input_file = "./data/llm_evaluations_gemma4_format_anon_user_2_top3.xlsx"
    output_file = "./data/llm_evaluation_results_checkpoint_top3_gpt41.jsonl"

    run_experiment(input_file, output_file)