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
import tensorflow as tf
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

import pickle
import numpy as np
from deepface import DeepFace
from litellm import completion 
import opensmile

class TextEmotionRecognizer:
    def __init__(self, llm_model="phi4-mini"):
        self.llm_model = llm_model

    def analyze(self, text, one_word=True):
        """Analyze text for emotions using a language model"""
        
        try:
            if one_word:
                response = completion(
                    model=f"ollama_chat/{self.llm_model}",
                    messages=[{
                        "content": f"Respond with only one word (lower case and no extra characters) from these emotions ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'] respecting to the most expressed emotion in the following piece of text: {text}\nRemember, answer with only one word in lower case without punctuation from ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'].",
                        "role": "user"}],
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
    def __init__(self, model_path='/Users/sofiafernandes/Documents/Repos/TFM/src/svm_model.pkl'):
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            self.model = None
            print(f"Warning: Speech emotion recognizer model not found at {model_path}")
        except Exception as e:
            self.model = None
            print(f"Error loading speech emotion recognizer: {e}")

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
            return {"emotions": prediction.tolist()} # Return as list for JSON
        except Exception as e:
            return {"error": f"Error during audio emotion analysis: {e}"}

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
        try:
            image_array = np.array(image)
            results = DeepFace.analyze(
                img_path=image_array,
                actions=['emotion'],
                detector_backend=self.backend,
                enforce_detection=True,  # Ensure face detection is enforced
                #progress_bar=0,  # Disable progress bar for silent operation
                normalize=True,  # Normalize the image for better results
                silent=True
            )
            if results:
                return {'emotions': results[0]['emotion'], 'dominant_emotion': results[0]['dominant_emotion']}
            else:
                return {'error': 'No face detected'}
        except Exception as e:
            return {'error': str(e)}

    def analyze_video_frames(self, frames):
        return [self.analyze_image(frame) for frame in frames]
