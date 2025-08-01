'''
 # @ Author: Sofia Condesso (50308)
 # @ Create Time: 2025-04-09 14:35:30
 # @ Description: This module provides an interface for interacting with a chatbot assistant
 #                  that can discuss the results of an emotion analysis.
 #                  It uses the Ollama API to generate responses based on the analysis results.
 # @ References:
 '''

#from ollama import completion
from litellm import completion

class ChatbotAssistant:
    def __init__(self, llm_model="phi4-mini"):
        self.llm_model = llm_model
        self.analysis_summary = ""
        self.conversation_history = [{"role": "system", "content": "You are a helpful assistant that can discuss the results of an emotion analysis."}]

    def load_analysis(self, analysis_results):
        summary_parts = []
        if "text_emotion" in analysis_results and isinstance(analysis_results["text_emotion"], str):
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
        self.conversation_history.append({"role": "user", "content": message})
        try:
            response = completion(
                model=f"ollama_chat/{self.llm_model}",
                messages=self.conversation_history,
                stream=False
            )
            bot_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": bot_response})
            return bot_response
        except Exception as e:
            return f"Error generating chatbot response: {str(e)}"
