'''
 # @ Author: Sofia Condesso (50308)
 # @ Create Time: 2025-04-09 14:41:23
 # @ Description: This module provides an interface for analyzing emotions from various media types
 #                  (text, audio, image, and video) and interacting with a chatbot assistant.
 #                  It uses the FileProcessor to handle file processing, and different emotion recognizers
 #                  for text, speech, and facial emotion analysis.
 # @ References:
 # 
 '''

from PIL import Image
from src.processor import FileProcessor
from src.recognizers import TextEmotionRecognizer, SpeechEmotionRecognizer, FaceEmotionRecognizer
from src.chatbot import ChatbotAssistant 


class EmotionRecognitionAssistant:
    def __init__(self):
        self.file_processor = FileProcessor()
        self.text_recognizer = TextEmotionRecognizer()
        self.speech_recognizer = SpeechEmotionRecognizer()
        self.face_recognizer = FaceEmotionRecognizer()
        self.chatbot = ChatbotAssistant() 

    def analyze(self, file_path, chatbot=True):
        processed_data = self.file_processor.process_file(file_path)

        if "error" in processed_data:
            return processed_data

        analysis_results = {}

        if "text" in processed_data and processed_data["text"]:
            analysis_results["text_emotion"] = {"transcription": processed_data["text"], "emotion": self.text_recognizer.analyze(processed_data["text"])}

        if "audio" in processed_data and processed_data["audio"] and processed_data["audio"]["raw"] is not None:
            analysis_results["audio_emotion"] = self.speech_recognizer.analyze(
                processed_data["audio"]["raw"], processed_data["audio"]["sample_rate"])
            
            #analysis_results["audio_emotion"] = self.speech_recognizer.analyze_parts(
            #    processed_data["audio"]["audio_chunks"], processed_data["audio"]["sample_rate"])

        if "image" in processed_data and isinstance(processed_data["image"], Image.Image):
            analysis_results["face_emotion"] = self.face_recognizer.analyze_image(processed_data["image"])

        if "frames" in processed_data and processed_data["frames"]:
            analysis_results["face_emotion"] = self.face_recognizer.analyze_video_frames(processed_data["frames"])

        if chatbot:
            # Initialize chatbot with the analysis results
            self.chatbot.load_analysis(analysis_results)

        return analysis_results

    def chat(self, user_input):
        return self.chatbot.send_message(user_input)
    

if __name__ == "__main__":
    assistant = EmotionRecognitionAssistant()
    file_path = '/Users/sofiafernandes/Documents/Repos/MEIM-ano1-sem2/TFM-SC/01-01-05-02-02-01-01.mp4' # Replace with a valid file path
    analysis_result = assistant.analyze(file_path)

    print("Analysis Results:")
    print(analysis_result)

    if assistant.chatbot.analysis_summary:
        print("\nChat with the Assistant:")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = assistant.chat(user_input)
            print(f"Assistant: {response}")
    else:
        print("\nNo analysis summary available to chat about.")
