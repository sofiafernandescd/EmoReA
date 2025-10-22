import os
import tempfile
import cv2
import whisper
import librosa
from PyPDF2 import PdfReader
from docx import Document
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np


class FileProcessor:
    def __init__(self, whisper_model="base"):
        self.file_types = {
            "text": ["txt", "pdf", "docx"],
            "audio": ["mp3", "wav", "sph"],
            "image": ["jpg", "jpeg", "png"],
            "video": ["mp4", "avi", "mov", "webm"],
        }

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.transcriber = whisper.load_model(whisper_model)


    def process_file(self, file_path):
        """Detect file type and process accordingly"""
        if not os.path.isfile(file_path):
            return {"error": f"File not found: {file_path}"}

        file_type = self._detect_file_type(file_path)
        processors = {
            "text": self._process_text,
            "audio": self._process_audio,
            "image": self._process_image,
            "video": self._process_video,
        }

        processor = processors.get(file_type)
        if not processor:
            return {"error": f"Unsupported file type: {file_path}"}

        try:
            return processor(file_path)
        except Exception as e:
            return {"error": str(e), "modality": file_type}

    
    def _detect_file_type(self, file_path):
        ext = file_path.split(".")[-1].lower()
        for category, extensions in self.file_types.items():
            if ext in extensions:
                return category
        return "unknown"

   
    def _process_text(self, file_path):
        """Read and extract text content from txt, pdf, docx"""
        ext = file_path.split(".")[-1].lower()
        text = ""

        if ext == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])

        elif ext == "docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        return {
            "modality": "text",
            "data": {"text": text.strip()},
            "meta": {"length_chars": len(text)},
        }

    
    def _process_audio(self, file_path):
        """Load audio and transcribe with Whisper"""
        audio, sr = librosa.load(file_path, sr=16000)
        transcript = self.transcriber.transcribe(file_path)

        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in transcript.get("segments", [])
        ]

        audio_chunks = [
            audio[int(seg["start"] * sr) : int(seg["end"] * sr)] for seg in segments
        ]

        return {
            "modality": "audio",
            "data": {
                "transcript": transcript.get("text", ""),
                "segments": segments,
                "audio_chunks": audio_chunks,
                "raw_audio": audio,
            },
            "meta": {"sample_rate": sr, "duration_s": len(audio) / sr},
        }

    
    def _process_image(self, file_path):
        """Detect faces and return face region (or full image)"""
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        cropped_faces = []
        for (x, y, w, h) in faces:
            face_img = img[y : y + h, x : x + w]
            cropped_faces.append(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)))

        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return {
            "modality": "image",
            "data": {
                "full_image": pil_image,
                "faces": cropped_faces if cropped_faces else [pil_image],
            },
            "meta": {"num_faces": len(cropped_faces)},
        }


    def _process_video(self, file_path, frame_rate=1):
        """Extract audio and face frames from video"""
        result = {"modality": "video", "data": {}, "meta": {}}

        # AUDIO 
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
            video = VideoFileClip(file_path)
            duration = video.duration
            fps = video.fps

            if video.audio:
                video.audio.write_audiofile(tmpfile.name, verbose=False, logger=None)
                audio_result = self._process_audio(tmpfile.name)
                result["data"]["audio"] = audio_result["data"]
                result["meta"]["audio_duration_s"] = audio_result["meta"]["duration_s"]
            else:
                result["data"]["audio"] = None

        # FRAMES 
        cap = cv2.VideoCapture(file_path)
        frames = []
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) * frame_rate)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    frames.append(
                        Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    )
            frame_count += 1

        cap.release()
        result["data"]["frames"] = frames
        result["meta"]["num_frames"] = len(frames)
        result["meta"]["video_duration_s"] = duration
        result["meta"]["fps"] = fps

        return result
