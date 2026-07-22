import json
import pandas as pd
from tqdm.auto import tqdm
from litellm import completion
from prompts import ACADEMIC_SYSTEM, PITCH_SYSTEM, DETAILED_ACADEMIC_SYSTEM, DETAILED_PITCH_SYSTEM, USER_PROMPT_TEMPLATE

# ============================================================
# Local Setup
# ============================================================
API_BASE = "http://localhost:11434"
MODEL = "ollama_chat/gemma4:e4b"

# ============================================================
# OpenAI Setup
# ============================================================
#os.environ["OPENAI_API_KEY"] = "" # your API key
#API_BASE = ""
#MODEL = "gpt-4.1-mini"
#MODEL = "gpt-4o-mini" 

class FeedbackCoach:
    """
    Generates four evaluations for each presentation:
        - academic_simple
        - academic_detailed
        - pitch_simple
        - pitch_detailed
    """

    PROMPTS = {
        "academic_simple": ACADEMIC_SYSTEM,
        "pitch_simple": PITCH_SYSTEM,
        "academic_detailed": DETAILED_ACADEMIC_SYSTEM,
        "pitch_detailed": DETAILED_PITCH_SYSTEM,
    }

    def __init__(self, api_host=API_BASE,
                 model="ollama_chat/gemma4:e4b"
                 #model="ollama_chat/mistral:latest"
                 ):
        self.api_host = api_host
        self.model = model

    # ---------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------

    @staticmethod
    def _format_top3(top3):
        if not top3:
            return "unknown"

        return ", ".join(
            f"{x['label']} ({x['score']:.2f})"
            for x in top3[:3]
        )
    
    @staticmethod
    def _format_top1(top3):
        if not top3:
            return "unknown"

        return f"{top3[0]['label']} ({top3[0]['score']:.2f})"
            
    

    def build_timeline(self, predictions: dict, top3=False) -> str:
        timeline = []

        text_segments = predictions.get("text_emotion", [])
        audio_segments = predictions.get("audio_emotion", [])

        for t_seg, a_seg in zip(text_segments, audio_segments):
            if top3:
                timeline.append(
                    f"[{t_seg['start']:.1f}s - {t_seg['end']:.1f}s] "
                    f"Audio: {self._format_top3(a_seg.get('top3', []))} | "
                    f"Text: {self._format_top3(t_seg.get('top3', []))} | "
                    f'Transcript: "{t_seg["text"]}"'
                )
            else:
                timeline.append(
                    f"[{t_seg['start']:.1f}s - {t_seg['end']:.1f}s] "
                    f"Audio: {self._format_top1(a_seg.get('top3', []))} | "
                    f"Text: {self._format_top1(t_seg.get('top3', []))} | "
                    f'Transcript: "{t_seg["text"]}"'
                )


        print("\n".join(timeline))

        return "\n".join(timeline)

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    def _llm_call(self, system_prompt: str, timeline: str) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    emotion_timeline=timeline
                ),
            },
        ]

        response = completion(
            model=self.model,
            messages=messages,
            api_base=self.api_host,
            temperature=0.0,
            stream=False,
        )

        #print(response.choices[0].message.content.strip())

        return response.choices[0].message.content.strip()

    # ---------------------------------------------------------
    # Main API
    # ---------------------------------------------------------

    def evaluate_all(self, predictions: dict) -> dict:
        timeline = self.build_timeline(predictions)

        outputs = {
            "emotion_timeline": timeline
        }

        for mode, system_prompt in self.PROMPTS.items():
            outputs[mode] = self._llm_call(
                system_prompt,
                timeline
            )
            print(mode + '\n' + outputs[mode])

        return outputs



import json
import pandas as pd
from tqdm.auto import tqdm

# ============================================================
# Generate LLM Evaluations and Save to Excel
# ============================================================

def generate_llm_evaluations(
    input_jsonl: str,
    output_excel: str,
    api_host: str = API_BASE,
    model: str = 'ollama_chat/gemma4:e4b'
):
    chatbot = FeedbackCoach(api_host=api_host, model=model)

    df = pd.read_json(input_jsonl, lines=True)

    outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        predictions = {
            "text_emotion": json.loads(row["text_emotion"]),
            "audio_emotion": json.loads(row["speech_emotion"]),
        }

        evaluations = chatbot.evaluate_all(predictions)

        outputs.append({
            **row.to_dict(),

            # Structured timeline
            "emotion_timeline": evaluations["emotion_timeline"],

            # 4 experimental conditions
            "academic_simple": evaluations["academic_simple"],
            "academic_detailed": evaluations["academic_detailed"],
            "pitch_simple": evaluations["pitch_simple"],
            "pitch_detailed": evaluations["pitch_detailed"],
        })

        df_out = pd.DataFrame(outputs)

        # Save Excel
        df_out.to_excel(output_excel, index=False)

    

    print(f"Saved results to: {output_excel}")

    return df_out


