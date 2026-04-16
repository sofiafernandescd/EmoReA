'''
 # @ Author: Sofia Condesso (50308)
 # @ Create Time: 2025-04-09 14:35:30
 # @ Description: This module provides an interface for interacting with a chatbot assistant
 #                  that can discuss the results of an emotion analysis.
 #                  It uses the Ollama API to generate responses based on the analysis results.
 # 
 # @ References: https://docs.litellm.ai/docs/proxy/docker_quick_start
 '''

#from ollama import completion
from litellm import completion
from dotenv import load_dotenv

# Load at the module level or in your main entry point
load_dotenv()

class ChatbotAssistant:
    def __init__(self, llm_model="gemma4:e4b"):
        self.llm_model = llm_model
        self.analysis_summary = ""
        self.conversation_history = [{"role": "system", "content": "You are a helpful assistant that can discuss the results of an emotion analysis, namely in what the person can improve in his/her presentation."}]

    def load_analysis(self, analysis_results):
        print("---- Pre-results:\n", analysis_results)
        timeline = []

        # 1. Process Text Segments
        # These represent the semantic meaning of the words spoken
        text_segments = analysis_results.get("text_emotion", [])
        if isinstance(text_segments, list):
            for seg in text_segments:
                start = seg.get('start', 0.0)
                text = seg.get('text', '')
                emotion = seg.get('emotion', 'unknown')
                timeline.append(f"[{start:.1f}s] Text Content: '{text}' (Sentiment: {emotion})")

        # 2. Process Audio Segments (GeMAPS-lite)
        # These represent the prosody (how it was said)
        audio_segments = analysis_results.get("audio_emotion", [])
        if isinstance(audio_segments, list):
            for seg in audio_segments:
                start = seg.get('start', 0.0)
                res = seg.get("emotion", {})
                label = res.get("label", "unknown")
                metrics = res.get("metrics", {})
                
                # GeMAPS-lite features: Pitch and Jitter (Stability)
                pitch = metrics.get('pitch_mean', 0)
                # Jitter measures frequency instability; 1 - jitter is a simple proxy for 'Smoothness'
                stability = 1 - metrics.get('jitter', 0)
                
                timeline.append(
                    f"[{start:.1f}s] Audio Tone: {label} "
                    f"(Pitch: {pitch:.1f}Hz, Voice Stability: {stability:.2%})"
                )

        # 3. Process Face Segments
        face_segments = analysis_results.get("face_emotion", [])
        if isinstance(face_segments, list):
            for i, res in enumerate(face_segments):
                if isinstance(res, dict) and "dominant_emotion" in res:
                    timeline.append(f"Frame {i}: Visual Expression: {res['dominant_emotion']}")

        # Final Assembly
        if timeline:
            # Sort timeline by timestamp if necessary (extracted from the string or original objects)
            # For now, we'll join them as processed
            full_context = "Detailed Multimodal Analysis:\n" + "\n".join(timeline)
            self.conversation_history.append({"role": "system", "content": full_context})
        else:
            self.conversation_history.append({
                "role": "system", 
                "content": "No emotional data was extracted from the provided input."
            })

    def old_load_analysis(self, analysis_results):
        summary_parts = []
        if "text_emotion" in analysis_results and isinstance(analysis_results["text_emotion"], list):
            summary_parts.append(f"The detected emotion in the text was: {analysis_results['text_emotion']}.")
        elif "text_emotion" in analysis_results and "error" in analysis_results["text_emotion"]:
            summary_parts.append(f"There was an error analyzing the text: {analysis_results['text_emotion']['error']}.")

        if "audio_emotion" in analysis_results and isinstance(analysis_results["audio_emotion"], dict) and "emotions" in analysis_results["audio_emotion"]:
            summary_parts.append(f"The detected emotions in the audio were: {analysis_results['audio_emotion']['emotions']}.")
        elif "audio_emotion" in analysis_results and "error" in analysis_results["audio_emotion"]:
            summary_parts.append(f"There was an error analyzing the audio: {analysis_results['audio_emotion']['error']}.")

        if "face_emotion" in analysis_results:
            if isinstance(analysis_results["face_emotion"], dict) and "dominant_emotion" in analysis_results["face_emotion"]:
                summary_parts.append(f"The dominant facial emotion was: {analysis_results['face_emotion']['dominant_emotion']}.")
            elif isinstance(analysis_results["face_emotion"], list):
                dominant_emotions = [res.get('dominant_emotion') for res in analysis_results["face_emotion"] if isinstance(res, dict) and 'dominant_emotion' in res]
                if dominant_emotions:
                    summary_str = ", ".join(dominant_emotions)
                    summary_parts.append(f"The dominant facial emotions across frames were: {summary_str}.")
                elif analysis_results["face_emotion"] and all("error" in res for res in analysis_results["face_emotion"]):
                    summary_parts.append(f"There were errors analyzing faces in the frames.")
            elif isinstance(analysis_results["face_emotion"], dict) and "error" in analysis_results["face_emotion"]:
                summary_parts.append(f"There was an error analyzing the face: {analysis_results['face_emotion']['error']}.")

        if summary_parts:
            self.analysis_summary = "Here's a summary of the emotion analysis: " + " ".join(summary_parts)
            self.conversation_history.append({"role": "system", "content": self.analysis_summary})
        else:
            self.conversation_history.append({"role": "system", "content": "No emotions were detected or there were errors during analysis."})

    def send_message(self, message):
        print("Emotion Summary:\n", self.conversation_history)
        self.conversation_history.append({"role": "user", "content": message})
        try:
            response = completion(
                model=f"ollama_chat/{self.llm_model}",
                messages=self.conversation_history,
                api_base="", # Replace with your ollama URL
                extra_headers={
                    "ngrok-skip-browser-warning": "true"
                },
                stream=False
            )
            bot_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": bot_response})
            return bot_response
        except Exception as e:
            return f"Error generating chatbot response: {str(e)}"
