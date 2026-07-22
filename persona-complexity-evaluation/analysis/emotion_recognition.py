from typing import Dict, List, Any
import os
import whisper
import librosa
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import torch
import logging

# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set device 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# check allocated memory and empty cache
logger.info(f"Current allocated memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
torch.mps.empty_cache()


class EmotionRecognition:
    """
        Text and speech emotion recognition using models published on HuggingFace.
        The models used:
            - TER: https://huggingface.co/boltuix/bert-emotion#credits
            - SER: https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition
       
    """
    def __init__(self):

        # Whisper model (heavy, must be reused)
        self.transcriber = whisper.load_model("base", device=DEVICE)
        logger.info("Whisper model loaded successfully.\nCurrent allocated memory: {:.2f} GB".format(torch.mps.current_allocated_memory() / 1e9))
        # TER model and tokenizer
        self.ter_model = AutoModelForSequenceClassification.from_pretrained("boltuix/bert-emotion").to(DEVICE)
        logger.info("TER model loaded successfully.\nCurrent allocated memory: {:.2f} GB".format(torch.mps.current_allocated_memory() / 1e9))
        self.ter_tokenizer = AutoTokenizer.from_pretrained("boltuix/bert-emotion")
        logger.info("TER tokenizer loaded successfully.\nCurrent allocated memory: {:.2f} GB".format(torch.mps.current_allocated_memory() / 1e9))
        # SER pipeline
        self.ser_pipeline = pipeline(
            "audio-classification",
            model="r-f/wav2vec-english-speech-emotion-recognition",
            device=torch.device(DEVICE), # gpu
            top_k=3  # store top-3 preds  
        )
        logger.info("SER pipeline initialized successfully.\nCurrent allocated memory: {:.2f} GB".format(torch.mps.current_allocated_memory() / 1e9))
        

    def _process_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.

        Optimizations from v1:
        - Removed returning raw waveform (huge memory waste)
        - Added Whisper-native loading instead of duplicating pipelines
        - Return only meaningful structured outputs
        """
        # transcribe audio
        result = self.transcriber.transcribe(file_path)
        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]
        # split audio in chunks according to segments
        audio, sr = librosa.load(file_path, sr=16000)
        audio_chunks = [audio[int(seg['start'] * sr):int(seg['end'] * sr)] for seg in segments]

        logger.info(f"Audio processed: {len(segments)} segments extracted.\nCurrent allocated memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "audio": {
                #"raw": audio,
                "audio_chunks": audio_chunks,
                "sample_rate": sr,
            }
        }
    
    def _emotion_recognition(self, data: Dict):

        segments = data["segments"]
        audio_chunks = data["audio"]["audio_chunks"]

        texts = [s["text"] for s in segments]

        inputs = self.ter_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = self.ter_model(**inputs)
        probs = outputs.logits.softmax(dim=1)
        id2label = self.ter_model.config.id2label

        text_emotions = []
        for i, seg in enumerate(segments):
            ## top-1
            # pred_id = probs[i].argmax().item()
            #top-3
            topk = torch.topk(probs[i], k=3)
            top_labels = [id2label[idx.item()] for idx in topk.indices]
            top_scores = [score.item() for score in topk.values]

            text_emotions.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                # "emotion": id2label[pred_id]
                "top3": [
                    {"label": l, "score": s}
                    for l, s in zip(top_labels, top_scores)
                ]
            })

        audio_emotions = []
        for i, chunk in enumerate(audio_chunks):
            if len(chunk) == 0:
                continue

            try:
                result = self.ser_pipeline(
                    {"array": chunk, "sampling_rate": 16000},
                    top_k=3
                )

                audio_emotions.append({
                    "start": segments[i]["start"],
                    "end": segments[i]["end"],
                    #"emotion": result
                    "top3": result
                })

            except Exception as e:
                audio_emotions.append({
                    "start": segments[i]["start"],
                    "end": segments[i]["end"],
                    "emotion": {"label": "error", "score": 0.0, "error": str(e)}
                })

        logger.info(f"Emotion recognition completed.\nCurrent allocated memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")

        return {
            "full_text": data["text"],
            "text_emotion": text_emotions,
            "audio_emotion": audio_emotions
        }

# example usage
if __name__ == "__main__":
    # audio file path
    FILE_PATH = "./data/demo1.mp3"

    er = EmotionRecognition()
    result = er._process_audio(FILE_PATH)
    emotions = er._emotion_recognition(result)
    print(emotions)