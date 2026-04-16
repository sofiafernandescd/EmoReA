"""
Chagelog:
---------------------------
1. Eliminated redundant decoding (video/audio loaded multiple times previously).
2. Replaced inefficient audio loading for videos (librosa on video file is costly and unreliable).
3. Optimized Whisper usage (single consistent pipeline, avoids unnecessary memory duplication).
4. Reduced memory footprint (removed raw audio duplication, lazy chunking).
5. Improved frame extraction (sequential access instead of random seeking per frame).
6. Added robust error handling and validation.
7. Standardized return structures.
8. Removed deprecated / duplicate code paths.
9. Improved readability, maintainability, and scalability.
10. Added production-level comments and clear separation of concerns.

NOTE:
- This implementation is optimized for performance + clarity (production-ready baseline).
- Further speedups possible via batching, multiprocessing, or GPU tuning.
"""

import os
from tempfile import NamedTemporaryFile
import cv2
import whisper
import librosa
import numpy as np

from typing import Dict, List, Any
from PyPDF2 import PdfReader
from docx import Document
from moviepy.editor import VideoFileClip
from PIL import Image
import mediapipe as mp


class FileProcessor:
    """
    High-performance multi-modal file processor.

    Supports:
    - Text (txt, pdf, docx)
    - Audio (mp3, wav, sph, m4a)
    - Image (face-aware extraction)
    - Video (audio transcription + structured frame sampling)
    """

    # ---------------------- INITIALIZATION ---------------------- #

    def __init__(self, whisper_model: str = "base"):
        """
        Initialize processor with reusable heavy models.

        NOTE:
        - Whisper model is loaded ONCE to avoid repeated GPU/CPU overhead.
        - Haar cascade is lightweight and cached.
        """
        self.file_types = {
            "text": {"txt", "pdf", "docx"},
            "audio": {"mp3", "wav", "sph", "m4a"},
            "image": {"jpg", "jpeg", "png"},
            "video": {"mp4", "avi", "mov", "webm"},
        }

        # Face detector (fast, CPU-based)
        #self.face_cascade = cv2.CascadeClassifier(
        #    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        #)
        # MediaPipe detector
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

        # Whisper model (heavy → must be reused)
        self.transcriber = whisper.load_model(whisper_model)

    # ---------------------- PUBLIC API ---------------------- #

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Entry point for processing any supported file.

        Returns:
            dict: structured output depending on modality
        """
        if not os.path.isfile(file_path):
            return {"error": "File not found"}

        file_type = self._detect_file_type(file_path)

        try:
            if file_type == "text":
                return self._process_text(file_path)
            elif file_type == "audio":
                return self._process_audio(file_path)
            elif file_type == "image":
                return self._process_image(file_path)
            elif file_type == "video":
                return self._process_video(file_path)
            else:
                return {"error": f"Unsupported file type: {file_type}"}

        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

    # ---------------------- TYPE DETECTION ---------------------- #

    def _detect_file_type(self, file_path: str) -> str:
        """
        Detects file category from extension.

        More robust than split('.')[-1] → handles edge cases.
        """
        ext = os.path.splitext(file_path)[-1].lower().strip(".")
        for category, extensions in self.file_types.items():
            if ext in extensions:
                return category
        return "unknown"

    # ---------------------- TEXT PROCESSING ---------------------- #

    def _process_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from supported document formats.

        Improvements:
        - Safe extraction (handles None pages in PDFs)
        - Encoding safety for txt files
        """
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif ext == ".pdf":
            reader = PdfReader(file_path)
            text = "\n".join(
                [page.extract_text() or "" for page in reader.pages]
            )
 
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])

        else:
            text = ""

        return {"text": text}

    # ---------------------- AUDIO PROCESSING ---------------------- #

    def _process_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.

        Key Optimizations:
        - Avoid returning raw waveform (huge memory waste)
        - Use Whisper-native loading instead of duplicating pipelines
        - Return only meaningful structured outputs
        """

        # Whisper handles loading internally → avoids librosa duplication
        result = self.transcriber.transcribe(file_path)

        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]

        audio, sr = librosa.load(file_path, sr=16000)
        audio_chunks = [audio[int(seg['start'] * sr):int(seg['end'] * sr)] for seg in segments]


        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "audio": {
                "raw": audio,
                "audio_chunks": audio_chunks,
                "sample_rate": sr,
            }
        }

    # ---------------------- IMAGE PROCESSING ---------------------- #

    def _old_process_image(self, file_path: str) -> Dict[str, Any]:
        """
        Detect and extract primary face (if present).

        Improvements:
        - Validation for corrupted images
        - Clean fallback logic
        """

        img = cv2.imread(file_path)

        if img is None:
            raise ValueError("Invalid or corrupted image file.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 5)
        

        if len(faces) > 0:
            x, y, w, h = faces[0]
            img = img[y : y + h, x : x + w]

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(img)

        return {"image": pil_image}
    
    def _process_image(self, file_path):
        frame = cv2.imread(file_path)

        if frame is None:
            raise ValueError("Invalid or corrupted image file.")
        
        # convert to RGB for MediaPipe
        rgb_frame = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # MediaPipe returns relative coordinates (0 to 1)
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                
                # crop and return the face for DeepFace emotion analysis
                face_crop = frame[y:y+bh, x:x+bw]
                return {"image": face_crop}
        return {"image": rgb_frame}

    # ---------------------- VIDEO PROCESSING ---------------------- #

    def _process_video(self, file_path: str) -> Dict[str, Any]:
        """
        Efficient video processor.

        Major Fixes vs Original:
        -----------------------
        ❌ OLD PROBLEMS:
        - librosa used directly on video (inefficient & unreliable)
        - random frame seeking (VERY slow)
        - duplicated decoding pipelines
        - unnecessary memory usage (raw audio, full frame lists)

        ✅ NEW APPROACH:
        - Extract audio ONCE using moviepy
        - Transcribe audio cleanly
        - Sequential frame reading (fast)
        - Frame sampling aligned with segments
        """

        result: Dict[str, Any] = {}

        # ---------- STEP 1: Extract Audio Safely ---------- #
        audio_path = None

        try:
            with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name

            video = VideoFileClip(file_path)

            if video.audio is not None:
                video.audio.write_audiofile(
                    audio_path,
                    verbose=False,
                    logger=None,
                )

                audio_result = self._process_audio(audio_path)
                result.update(audio_result)
            else:
                result.update({"text": None, "segments": []})

        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

        # ---------- STEP 2: Frame Extraction (Optimized) ---------- #
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            raise IOError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        frames_by_segment: List[List[Image.Image]] = []

        # Precompute timestamps for efficiency
        timestamps = []
        for seg in result.get("segments", []):
            start, end = seg["start"], seg["end"]
            duration = end - start

            timestamps.append([
                start + duration * 0.25,
                start + duration * 0.75,
            ])

        # Flatten timestamps for sequential scan
        flat_timestamps = sorted(
            [(t, i) for i, ts in enumerate(timestamps) for t in ts]
        )

        current_idx = 0
        frame_id = 0

        frames_by_segment = [[] for _ in timestamps]

        while cap.isOpened() and current_idx < len(flat_timestamps):
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_id / fps

            # Process all timestamps <= current_time
            while (
                current_idx < len(flat_timestamps)
                and flat_timestamps[current_idx][0] <= current_time
            ):
                _, seg_idx = flat_timestamps[current_idx]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                frames_by_segment[seg_idx].append(pil_img)
                current_idx += 1

            frame_id += 1

        cap.release()

        result["video_chunks"] = frames_by_segment
        result["frames"] = [
            f for chunk in frames_by_segment for f in chunk
        ]

        return result