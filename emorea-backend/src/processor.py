import os
import tempfile
import cv2
import whisper #pip install git+https://github.com/openai/whisper.git
import librosa
from PyPDF2 import PdfReader
from docx import Document
from moviepy.editor import VideoFileClip
from PIL import Image


class FileProcessor:
    def __init__(self):
        self.file_types = {
            'text': ['txt', 'pdf', 'docx'],
            'audio': ['mp3', 'wav', 'sph'],
            'image': ['jpg', 'jpeg', 'png'],
            'video': ['mp4', 'avi', 'mov', 'webm']
        }
        self.face_cascade = None        
        self.transcriber = whisper.load_model("base") 

    def process_file(self, file_path):
        """Main processing method"""
        if not os.path.isfile(file_path):
            return {"error": "File not found"}

        file_type = self._detect_file_type(file_path)

        try:
            if file_type == 'text':
                return self._process_text(file_path)
            elif file_type == 'audio':
                return self._process_audio(file_path)
            elif file_type == 'image':
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                return self._process_image(file_path)
            elif file_type == 'video':
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                return self._process_video(file_path)
            else:
                return {"error": "Unsupported file type"}
        except Exception as e:
            return {"error": str(e)}

    def _detect_file_type(self, file_path):
        """Detect file type category"""
        ext = file_path.split('.')[-1].lower()
        for category, extensions in self.file_types.items():
            if ext in extensions:
                return category
        return 'unknown'

    def _process_text(self, file_path):
        """Process text-based files"""
        ext = file_path.split('.')[-1].lower()
        text = ''

        if ext == 'txt':
            with open(file_path, 'r') as f:
                text = f.read()
        elif ext == 'pdf':
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                text = '\n'.join([page.extract_text() for page in reader.pages])
        elif ext == 'docx':
            doc = Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])

        return {"text": text}

    def _process_audio(self, file_path, chunk_duration=3, hop_duration=1):
        """Load audio files and transcribe them"""
        audio, sr = librosa.load(file_path, sr=16000)
        # tracribe audio and split into segments
        transcript = self.transcriber.transcribe(audio)
        segments = transcript['segments']
        audio_chunks = [audio[int(seg['start'] * sr):int(seg['end'] * sr)] for seg in segments]

        return {
            "text": transcript['text'],
            "segments": segments,
            "audio": {
                "raw": audio,
                "audio_chunks": audio_chunks,
                "sample_rate": sr,
            }
        }


    def _process_image(self, file_path):
        """Process image files with face detection"""
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return {"image": pil_image}

    def _process_video(self, file_path):
        """Process video files with frame extraction and audio processing"""
        result = {}
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmpfile:
            video = VideoFileClip(file_path)
            if video.audio:
                video.audio.write_audiofile(tmpfile.name)
                audio_result = self._process_audio(tmpfile.name)
                result.update(audio_result)
            else:
                result["audio"] = None
                result["text"] = None

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # Every 1 second
        frame_count = 0
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    frames.append(pil_image)

            frame_count += 1

        cap.release()
        result["frames"] = frames
        return result

