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
os.environ["LITELLM_API_BASE"] = "http://localhost:11434"
os.environ["LITELLM_API_KEY"] = "ollama"
#import tensorflow as tf
import pickle
from joblib import load
import numpy as np
from deepface import DeepFace
from litellm import completion 
import opensmile
#from concurrent.futures import ThreadPoolExecutor, as_completed
#import threading
import librosa
import cv2
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tensorflow.keras.models import load_model


class TextEmotionRecognizerHF:
    def __init__(self, model_path="./emotion_model_distilbert", device=None):
        """
        Load a fine-tuned Hugging Face emotion recognition model.
        """
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # keep the label mappings for interpretability
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def analyze(self, text):
        """
        Return the predicted emotion label for the input text.
        """
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        return self.id2label[pred_id]

    def analyze_with_scores(self, text):
        """
        Return emotion probabilities (softmax scores) for the input text.
        """
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}

class TextEmotionRecognizer:
    def __init__(self, 
                 llm_model="gemma2"):
        """
        LLM-based text emotion recognizer using LiteLLM with Ollama backend.
        Produces one-word labels or probabilistic scores depending on method.
        """
        self.llm_model = llm_model
        self.api_base = "http://localhost:11434"
        self.api_key = "ollama"
        self.source = "text"

    def analyze(self, 
                text, 
                one_word=True, 
                few_shot=False, 
                emo_list=None):
        """
        Analyze text for emotions using an LLM (Ollama local API).
        Returns standardized dict with label and optional scores.
        """
        if emo_list is None:
            emo_list = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

        try:
            # --- ONE-WORD CLASSIFICATION ---
            if one_word:
                messages = []

                if few_shot:
                    messages = [
                        {"role": "system", "content": (
                            "You are an emotion classification assistant. "
                            "Always respond with ONLY ONE WORD (lowercase, no punctuation) "
                            f"from: {emo_list}."
                        )},
                        {"role": "user", "content": "This is so exciting!"},
                        {"role": "assistant", "content": "happy"},
                        {"role": "user", "content": "I'm feeling really down about everything."},
                        {"role": "assistant", "content": "sad"},
                        {"role": "user", "content": "Why did you do that? I'm so upset!"},
                        {"role": "assistant", "content": "angry"},
                        {"role": "user", "content": text}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": (
                            f"Given the following text, identify the underlying emotion. "
                            f"Respond with ONLY ONE WORD (lowercase, no punctuation) "
                            f"from {emo_list}.\n\nText: {text}"
                        )}
                    ]

                response = completion(
                    model=f"ollama/{self.llm_model}",
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=0.0,
                    top_p=1.0,
                    messages=messages
                )

                label = response.choices[0].message.content.strip().lower()
                if label not in emo_list:
                    label = "neutral"

                return {
                    "label": label,
                    "scores": None,
                    "source": self.source
                }

            # --- DETAILED ANALYSIS (NOT ONE WORD) ---
            else:
                response = completion(
                    model=f"ollama/{self.llm_model}",
                    api_base=self.api_base,
                    api_key=self.api_key,
                    temperature=0.2,
                    messages=[
                        {"role": "user", 
                         "content": f"Analyze the emotions expressed in the following text and provide a detailed emotional analysis: {text}"}
                    ],
                )

                return {
                    "label": "neutral",  # placeholder for descriptive output
                    "scores": None,
                    "source": self.source,
                    "description": response.choices[0].message.content.strip()
                }

        except Exception as e:
            return {"error": str(e), "source": self.source}

    def analyze_with_scores(self, text, emo_list=None):
        """
        Ask the LLM to return scores/probabilities for all emotions.
        Returns standardized dict: {label, scores, source}.
        """
        if emo_list is None:
            emo_list = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

        try:
            prompt = (
                f"Given the following text, provide a score between 0 and 1 "
                f"for each emotion in {emo_list}. Ensure the sum of scores is 1. "
                f"Respond ONLY with a JSON dictionary. Text: \"{text}\""
            )

            response = completion(
                model=f"ollama/{self.llm_model}",
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.0,
                top_p=1.0,
                messages=[
                    {"role": "system", "content": "You are an emotion scoring assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw_output = response.choices[0].message.content.strip()

            try:
                scores = json.loads(raw_output)
            except json.JSONDecodeError:
                try:
                    scores = eval(raw_output)  # fallback if it's dict-like text
                except Exception:
                    scores = {}

            # sanitize and normalize
            scores = {emo: float(scores.get(emo, 0.0)) for emo in emo_list}
            total = sum(scores.values())
            if total > 0:
                scores = {emo: val / total for emo, val in scores.items()}

            label = max(scores, key=scores.get)

            return {
                "label": label,
                "scores": scores,
                "source": self.source
            }

        except Exception as e:
            return {"error": str(e), "source": self.source}



class SpeechEmotionRecognizer:
    def __init__(self, model_path='/Users/sofiafernandes/Documents/Repos/EmoReA/emorea-backend/notebooks/speech/ser_svm_model.joblib'):
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


    def extract_features_opensmile(self, audio, sr):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_signal(audio, sr)
        return features.values
    
    # Feature extraction func
    def extract_audio_features_2d(self, audio, sr, mfcc=True, chroma=True, mel=True):
        feats = []
        if mfcc:
            mf = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            feats.append(np.array(np.mean(mf.T, axis=0)))
        if chroma:
            st = np.abs(librosa.stft(audio))
            ch = librosa.feature.chroma_stft(S=st, sr=sr)
            feats.append(np.mean(ch.T, axis=0))
        if mel:
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            feats.append(np.mean(mel.T, axis=0))
        return np.array(feats)
    
    # Feature extraction func
    def extract_audio_features(self, audio, sr, mfcc=True, chroma=True, mel=True, spectral=True):
        feats = []
        if mfcc:
            mf = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            #feats.append(np.array(np.mean(mf.T, axis=0)))
            feats.extend(np.mean(mf, axis=1))
        if chroma:
            st = np.abs(librosa.stft(audio))
            ch = librosa.feature.chroma_stft(S=st, sr=sr)
            #feats.append(np.mean(ch.T, axis=0))
            feats.extend(np.mean(ch, axis=1))
        if mel:
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            feats.extend(np.mean(mel.T, axis=0))

        if spectral:
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            flatness = librosa.feature.spectral_flatness(y=y)
            feats.extend([
                np.mean(centroid), np.mean(bandwidth), np.mean(flatness)
            ])
        return np.array(feats)

    def analyze_old(self, audio, sr):
        if self.model is None:
            return {"error": "Speech emotion recognizer model not loaded"}
        #features = self.extract_audio_features(audio, sr)
        features = self.extract_features_opensmile(audio, sr)
        try:
            prediction = self.model.predict(features)
            return {"emotions": prediction.tolist()} 
        except Exception as e:
            return {"error": f"Error during audio emotion analysis: {e}"}
    
    def analyze(self, audio, sr=16000):
        if self.model is None:
            return {"error": "Speech emotion recognizer model not loaded"}
        
        features = self.extract_features_opensmile(audio, sr)
        try:
            pred = self.model.predict(features)[0]
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                classes = self.model.classes_
                scores = {cls: float(prob) for cls, prob in zip(classes, probs)}
            else:
                scores = None

            return {"label": pred, "scores": scores, "source": "speech"}
        except Exception as e:
            return {"error": f"Error during audio emotion analysis: {e}"}


    def analyze_parts(self, audio_list, sr):
        if self.model is None:
            return {"error": "Speech emotion recognizer model not loaded"}
        
        predictions = []
        for audio in audio_list:
            features = self.extract_audio_features(audio, sr)
            #features = self.extract_features(audio, sr)
            try:
                prediction = self.model.predict(features)
                predictions.append({prediction})
            except Exception as e:
                print(f"Error during audio emotion analysis: {e}")
        return {"emotions": predictions}

class FaceEmotionRecognizer:
    def __init__(self, backend='mtcnn', cnn_model_path=None):
        self.backend = backend
        self.cnn_model = None
        if cnn_model_path:
            self.load_cnn(cnn_model_path)

    def load_cnn(self, model_path):
        self.cnn_model = load_model(model_path)

    def analyze_image(self, image):
        if self.cnn_model:
            return self._analyze_with_cnn(image)
        else:
            return self._analyze_with_deepface(image)

    def _analyze_with_deepface(self, image):
        try:
            result = DeepFace.analyze(
                img_path=np.array(image),
                actions=['emotion'],
                detector_backend=self.backend,
                enforce_detection=False,
            )
            emo_scores = result[0]["emotion"]
            label = max(emo_scores, key=emo_scores.get)
            return {"label": label, "scores": emo_scores, "source": "face"}
        except Exception as e:
            return {"error": str(e)}

    def _analyze_with_cnn(self, image):
        """Use a trained CNN for emotion recognition"""
        import numpy as np
        import cv2
        img = cv2.resize(np.array(image), (48, 48)) / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        preds = self.cnn_model.predict(img)[0]
        classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        scores = {c: float(p) for c, p in zip(classes, preds)}
        label = classes[np.argmax(preds)]
        return {"label": label, "scores": scores, "source": "face"}

    def analyze_video_frames(self, frames):
        """Analize frames extracted from video."""
        """ TODO:
            Resize frames before detection: frame = cv2.resize(frame, (320, 240))
            Add a max_faces or confidence_threshold to reduce false positives
            Consider face_recognizer.analyze_video_frames(frames[:10]) to limit evaluation on very long videos
        """
        return [self.analyze_image(frame) for frame in frames]
