'''
 # @ Author: Sofia Condesso (50308)
 # @ Create Time: 2025-04-09 15:15:28
 # @ Description: This module provides classes for analyzing emotions from text, audio, and images.
 #                  It uses the DeepFace library for facial emotion analysis, OpenSMILE for speech emotion analysis,
 #                  and a language model for text emotion analysis.
 # @ References:
 #          - https://docs.litellm.ai/docs/providers/ollama
 '''
import os
import warnings
#import tensorflow as tf
# Suppress TensorFlow warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#warnings.filterwarnings('ignore')
#tf.get_logger().setLevel('ERROR')

import pickle
from joblib import load
import numpy as np
from deepface import DeepFace
from litellm import completion 
import opensmile
from concurrent.futures import ThreadPoolExecutor
import threading


class TextEmotionRecognizer:
    def __init__(self, 
                 #llm_model="deepseek-r1:1.5b"
                 #llm_model="qwen"
                 #llm_model="phi4-mini"
                 #llm_model="stablelm2:latest"
                 #llm_model="tinyllama:latest"
                 llm_model="openhermes:latest"
                 #llm_model="openhermes2.5-mistral:latest"
                 #llm_model="mistral:7b"
                 ):
        self.llm_model = llm_model
        self.executor = ThreadPoolExecutor(max_workers=2)

    def analyze_async(self, text):
        """Asynchronous method to analyze text for emotions"""
        """🔹1. LLM Calls Are Blocking Everything
            💡 Problem:
            You are calling the LLM synchronously, which blocks the entire program (including UI or video/audio processing) until a reply comes back.
            ✅ Solution:
            Use concurrent execution:
            - Replace synchronous completion() calls with a thread or async wrapper.
            For Jupyter: use asyncio + nest_asyncio.
            Or run LLM calls inside a ThreadPoolExecutor:
            TODO: remove this when we have a proper async LLM client
            - Use a queue to handle LLM requests and responses.
            - Use a separate thread for LLM calls to avoid blocking the main thread.
            - Use a timeout to avoid waiting indefinitely for LLM responses."""
        return self.executor.submit(self.analyze, text)

    def analyze(self, text, one_word=True):
        """Analyze text for emotions using a language model"""
        
        try:
            if one_word:

                response = completion(
                    model=f"ollama_chat/{self.llm_model}",
                    #max_tokens=1,  # Only allow one token response
                    #stop=["\n", ".", " ", "?", "!", ",", ";", ":"],  # Stop on any whitespace or punctuation
                    temperature=0.0,  # Set temperature to 0 for deterministic output
                    top_p=1.0,  # Use top-p sampling to ensure only the most likely token is returned
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an emotion classification assistant. "
                                "You must respond with ONLY ONE WORD (lowercase, no punctuation), from: "
                                "['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']."
                            )
                        },
                        {"role": "user", "content": "I'm feeling really down about everything."},
                        {"role": "assistant", "content": "sad"},
                        {"role": "user", "content": "This is the best day of my life!"},
                        {"role": "assistant", "content": "happy"},
                        {"role": "user", "content": f"{text}\nRemember to respond with only one word from ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']."}
                    ]
                )

            else:
                response = completion(
                    model=f"ollama_chat/{self.llm_model}",
                    messages=[{
                        "content": f"Analyze the emotions expressed in the following text and provide a detailed emotional analysis: {text}",
                        "role": "user"}],
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return {"error": str(e)}



class SpeechEmotionRecognizer:
    def __init__(self, model_path='/Users/sofiafernandes/Documents/Repos/EmoReA/emorea-backend/notebooks/speech/ser_svm_model_iemocap.joblib'):
        try:
            with open(model_path, 'rb') as file:
                #self.model = pickle.load(file)
                self.model = load(file)
        except FileNotFoundError:
            self.model = None
            print(f"Warning: Speech emotion recognizer model not found at {model_path}")
        except Exception as e:
            self.model = None
            print(f"Error loading speech emotion recognizer: {e}")

    def transcribe_async(self, audio):
        """Asynchronous method to transcribe audio using Whisper"""
        """🔹2. Synchronous Whisper Usage
            Whisper transcription is done inline with transcribe(audio), which is blocking.

            ✅ Suggestion:
            Transcribe first to segments (timestamps), then process audio in chunks later or in parallel.
            Use a thread or async wrapper to handle transcription without blocking the main thread.
        """
        thread = threading.Thread(target=self.transcriber.transcribe, args=(audio,))
        thread.start()

    def extract_features(self, audio, sr):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_signal(audio, sr)
        return features.values

    def analyze(self, audio, sr):
        if self.model is None:
            return {"error": "Speech emotion recognizer model not loaded"}
        features = self.extract_features(audio, sr)
        try:
            prediction = self.model.predict(features)
            return {"emotions": prediction.tolist()} 
        except Exception as e:
            return {"error": f"Error during audio emotion analysis: {e}"}
        
    def analyze_parts(self, audio_list, sr):
        if self.model is None:
            return {"error": "Speech emotion recognizer model not loaded"}
        
        predictions = []
        for audio in audio_list:
            features = self.extract_features(audio, sr)
            try:
                prediction = self.model.predict(features)
                predictions.append({prediction})
            except Exception as e:
                print(f"Error during audio emotion analysis: {e}")
        return {"emotions": predictions}

class FaceEmotionRecognizer:
    def __init__(self, backend = 'mtcnn'):
        """Initialize the face emotion recognizer with a specified backend
        Args:
            backend (str): The backend to use for face detection. Default is 'mtcnn'.
            detector_backend : string
            Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip' (default is opencv).
        """
        self.backend = backend

    def analyze_image(self, image):
        """Analyze a single image for facial emotions"""
        try:
            image_array = np.array(image)
            results = DeepFace.analyze(
                img_path=image_array,
                actions=['emotion'],
                detector_backend=self.backend,
                enforce_detection=True,  # Ensure face detection is enforced
                #progress_bar=0,  # Disable progress bar for silent operation
                #normalize=True,  # Normalize the image for better results
                #silent=True
            )
            if results:
                return {'emotions': results[0]['emotion'], 'dominant_emotion': results[0]['dominant_emotion']}
            else:
                return {'error': 'No face detected'}
        except Exception as e:
            return {'error': str(e)}

    def analyze_video_frames(self, frames):
        """🔹5. Video Face Detection Needs Optimization
            You detect face on every N-th frame, but DeepFace will still be slow.
            ✅ Suggestions:
            Resize frames before detection: frame = cv2.resize(frame, (320, 240))
            Add a max_faces or confidence_threshold to reduce false positives
            Consider face_recognizer.analyze_video_frames(frames[:10]) to limit evaluation on very long videos
        """
        return [self.analyze_image(frame) for frame in frames]
