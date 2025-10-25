'''
 # @ Author: Sofia Condesso (50308) - refactor by assistant
 # @ Create Time: 2025-04-09 15:15:28 (updated)
 # @ Description: Safe, concurrency-aware multimodal emotion recognizer.
'''
import os
# ---- Limit native thread pools to reduce mutex contention ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing
import numpy as np
import cv2
import librosa

# third-party libs (may import heavy libs after env vars set)
from deepface import DeepFace
from litellm import completion
import opensmile
from joblib import load

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Global lock to prevent concurrent DeepFace/TensorFlow use in same process
deepface_lock = threading.Lock()

# --- Helper: safe DeepFace analyze wrapper (serialized) ---
def _deepface_analyze_image(image_array, backend="mtcnn", enforce_detection=False):
    """
    Run DeepFace.analyze under a lock. Returns DeepFace result or raises.
    Must be called from the same process (Lock is threading.Lock).
    """
    with deepface_lock:
        # DeepFace expects either path or numpy array; we pass numpy array.
        return DeepFace.analyze(
            img_path=image_array,
            actions=['emotion'],
            detector_backend=backend,
            enforce_detection=enforce_detection,
        )


class TextEmotionRecognizer:
    def __init__(self, llm_model="openhermes:latest", max_workers=2, request_timeout=30):
        self.llm_model = llm_model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.request_timeout = request_timeout

    def analyze(self, text, one_word=True):
        """Synchronous text analysis using LLM (wraps litellm.completion)."""
        try:
            if one_word:
                response = completion(
                    model=f"ollama_chat/{self.llm_model}",
                    temperature=0.0,
                    top_p=1.0,
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
                        {"role": "user", "content": f"{text}\nRespond with only one word from ['neutral','happy','sad','angry','fear','disgust','surprise']."}
                    ],
                )
            else:
                response = completion(
                    model=f"ollama_chat/{self.llm_model}",
                    messages=[{"content": f"Analyze the emotions expressed in the following text: {text}", "role": "user"}],
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("LLM text analyze error")
            return {"error": str(e)}

    def analyze_async(self, text, one_word=True):
        """Return a Future for async analysis (non-blocking)."""
        return self.executor.submit(self.analyze, text, one_word)

    def batch_analyze(self, texts, one_word=True, max_workers=4, timeout=None):
        """Batch analyze using threads; returns list in same order as input."""
        if timeout is None:
            timeout = self.request_timeout
        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.analyze, txt, one_word): i for i, txt in enumerate(texts)}
            for fut in as_completed(futures, timeout=timeout * len(futures)):
                idx = futures[fut]
                try:
                    results[idx] = fut.result(timeout=timeout)
                except Exception as e:
                    logger.exception("Error in batch text analyze")
                    results[idx] = {"error": str(e)}
        return results


class SpeechEmotionRecognizer:
    """
    Speech emotion recognizer using a pre-trained model (joblib) and opensmile for features.
    - For parallel processing of many audio chunks, use analyze_parts with use_process_pool=True.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        if model_path:
            try:
                # joblib load returns the estimator
                self.model = load(model_path)
                logger.info("Loaded speech model from %s", model_path)
            except FileNotFoundError:
                logger.warning("Speech model not found at %s", model_path)
            except Exception:
                logger.exception("Error loading speech model")

        # Precreate an opensmile object for this process (thread-safe if not used concurrently)
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_features_opensmile(self, audio, sr):
        """Return numpy array of features (1D or 2D depending on feature set)."""
        try:
            feats = self.smile.process_signal(audio, sr)
            return feats.values
        except Exception:
            logger.exception("opensmile feature extraction failed")
            raise

    @staticmethod
    def _load_model_and_predict(model_path, features):
        """
        Worker helper for ProcessPool: loads model inside process and predict.
        Note: this reloads model per process; acceptable for heavy CPU parallelism.
        """
        try:
            model = load(model_path)
            pred = model.predict(features)
            return pred.tolist()
        except Exception:
            import traceback
            logger = logging.getLogger("speech_worker")
            logger.exception("worker predict failed")
            return {"error": traceback.format_exc()}

    def analyze(self, audio, sr):
        """Synchronous analyze single audio sample (numpy array)."""
        if self.model is None:
            return {"error": "Speech emotion recognizer model not loaded"}
        try:
            features = self.extract_features_opensmile(audio, sr)
            pred = self.model.predict(features)
            return {"emotions": pred.tolist()}
        except Exception as e:
            logger.exception("Speech analyze error")
            return {"error": str(e)}

    def analyze_parts(self, audio_list, sr, use_process_pool=False, max_workers=None):
        """
        Analyze multiple audio snippets.
        - If use_process_pool=True, uses ProcessPoolExecutor (safer for CPU-bound libs).
        - audio_list: list of numpy arrays
        """
        if self.model is None and not self.model_path:
            return {"error": "Speech model not loaded and no model_path provided"}

        results = []
        if use_process_pool:
            max_workers = max_workers or max(1, (multiprocessing.cpu_count() // 2))
            # Run feature extraction in current process, then dispatch prediction to workers that load model.
            # Build (features) list first
            features_list = []
            for audio in audio_list:
                feats = self.extract_features_opensmile(audio, sr)
                features_list.append(feats)
            # Use ProcessPool to predict (each worker loads its own model)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(self._load_model_and_predict, self.model_path, feats) for feats in features_list]
                for fut in as_completed(futures):
                    results.append(fut.result())
            return {"emotions": results}
        else:
            # Threaded/simple approach (single-process)
            for audio in audio_list:
                try:
                    feats = self.extract_features_opensmile(audio, sr)
                    pred = None
                    if self.model is not None:
                        pred = self.model.predict(feats)
                        results.append(pred.tolist())
                    else:
                        # fallback to process-based if no loaded model
                        results.append({"error": "no model loaded in main process"})
                except Exception as e:
                    logger.exception("Error analyzing audio part")
                    results.append({"error": str(e)})
            return {"emotions": results}


class FaceEmotionRecognizer:
    """
    Face emotion analyzer using DeepFace.
    - Serializes DeepFace calls with a lock to avoid mutex/blocking problems.
    - Resizes frames and samples to reduce cost.
    """

    def __init__(self, backend='mtcnn', frame_sample_rate=5, resize_to=(320, 240), enforce_detection=False):
        """
        backend: DeepFace detector backend
        frame_sample_rate: process every nth frame (int >=1)
        resize_to: tuple (w, h) to downsize frames before analysis
        enforce_detection: pass to DeepFace.analyze
        """
        self.backend = backend
        self.frame_sample_rate = max(1, int(frame_sample_rate))
        self.resize_to = resize_to
        self.enforce_detection = enforce_detection

    def _prepare_image(self, image):
        """Convert input to RGB uint8 numpy array and resize."""
        image_array = np.array(image)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        if image_array.ndim == 2:
            # grayscale -> RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            # RGBA -> RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        # resize
        if self.resize_to:
            image_array = cv2.resize(image_array, self.resize_to)
        return image_array

    def analyze_image(self, image):
        """Analyze a single image for facial emotions (serialized DeepFace)."""
        try:
            img = self._prepare_image(image)
            results = _deepface_analyze_image(img, backend=self.backend, enforce_detection=self.enforce_detection)
            if isinstance(results, list) and len(results) > 0:
                r = results[0]
            else:
                r = results
            # Normalize result format
            emotion_scores = r.get('emotion') if isinstance(r, dict) else None
            dominant = r.get('dominant_emotion') if isinstance(r, dict) else None
            if emotion_scores is not None:
                return {'emotions': emotion_scores, 'dominant_emotion': dominant}
            else:
                return {'error': 'No face detected or unexpected DeepFace output', 'raw': r}
        except Exception as e:
            logger.exception("DeepFace analyze_image failed")
            return {'error': str(e)}

    def analyze_video_frames(self, frames, max_frames=50):
        """
        Process a list/iterable of frames:
          - samples every `frame_sample_rate` frame,
          - downsizes frames,
          - processes up to `max_frames` frames total to avoid long runs.
        Returns list of DeepFace outputs (in order of processed frames).
        """
        results = []
        processed = 0
        for i, frame in enumerate(frames):
            if i % self.frame_sample_rate != 0:
                continue
            if processed >= max_frames:
                break
            try:
                resized = self._prepare_image(frame)
                results.append(self.analyze_image(resized))
            except Exception:
                logger.exception("Error processing frame %s", i)
                results.append({'error': 'frame processing failed'})
            processed += 1
        return results


# ---------------------------
# Example usage / tips:
# ---------------------------
# text_rec = TextEmotionRecognizer(llm_model="openhermes:latest")
# future = text_rec.analyze_async("I'm sad right now")
# print(future.result())
#
# speech_rec = SpeechEmotionRecognizer(model_path="path/to/ser_svm_model.joblib")
# resp = speech_rec.analyze(audio_np, sr)
#
# face_rec = FaceEmotionRecognizer(backend='mtcnn', frame_sample_rate=5)
# results = face_rec.analyze_video_frames(list_of_frames)
#
# Notes:
# - If you need heavy parallel audio processing, pass use_process_pool=True to analyze_parts.
# - Keep max_workers small at first (1-4) and increase only if stable.
# - If you still get '[mutex.cc] RAW: Lock blocking ...' reduce OMP/BLAS threads further or avoid concurrent native calls.
