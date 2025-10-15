# EmoReA Backend

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

This repository contains the FastAPI backend API for the EmoReA (Emotion Recognition Assistant) system. It handles file processing, emotion analysis using various machine learning models, and the chatbot functionality.

## Features

-   Receives file uploads via API endpoints.
-   Processes text, audio, image, and video files for emotion analysis.
-   Integrates with machine learning libraries (e.g., librosa, deepface, litellm).
-   Provides API endpoints for the frontend to retrieve analysis results.
-   Implements a chatbot that can engage in conversations based on the analysis.

## Installation

1.  Ensure you have Python 3.8+ installed on your system.
2.  Clone the main EmoReA repository:
    ```bash
    git clone <repository-url>
    cd emorea-backend
    ```
3.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate   # On Windows
    ```
4.  Install the dependencies using Poetry (recommended):
    ```bash
    pip install poetry
    poetry install
    ```
    Or, if you prefer pip:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Backend

For local development using Uvicorn:

```bash
uvicorn app.main:app --reload

```
The backend API will be accessible at http://localhost:8000. FastAPI also provides automatic API documentation at /docs and /redoc when the server is running.

## API Endpoints
POST /analyze/: Accepts multipart/form-data with a file parameter for emotion analysis. Returns a JSON object containing the analysis results.
POST /chat/: Accepts a JSON payload with a user_input field for the chatbot. Returns a string containing the assistant's response.
GET /health/: Returns a JSON response {"status": "healthy"} if the API is running.
Deployment
For deployment to Google Cloud Platform (GCP), the recommended approach is to containerize the backend application using Docker and deploy it to Google Cloud Run.

Create a Dockerfile:

```bash
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```
Build and Push Docker Image:

```bash
docker build -t gcr.io/$PROJECT_ID/emorea-backend:latest .
docker push gcr.io/$PROJECT_ID/emorea-backend:latest
```
Replace $PROJECT_ID with your Google Cloud Project ID.

## Deploy to Cloud Run:

```bash
gcloud run deploy emorea-backend --image gcr.io/$PROJECT_ID/emorea-backend:latest --platform managed --region YOUR_REGION --port 8080 --allow-unauthenticated
```
Replace YOUR_REGION with your desired GCP region.

Refer to the GCP Cloud Run documentation for more advanced deployment options and configurations.

## Contributing
Future work.

## License
This project is licensed under the MIT License - see the https://www.google.com/search?q=../LICENSE file for details.


# Suggestions of TODOs:
Your system is well-structured and modular, nicely separating concerns into `FileProcessor`, individual emotion recognizers (text, speech, face), and a high-level `EmotionRecognitionAssistant`. This is excellent design for a multi-modal pipeline.

But you’re now reaching the complexity where **scalability, performance, and maintainability** become essential. Below is a comprehensive critique — **strengths**, **weaknesses**, **concrete suggestions**, and **improvements to LLM usage**.

---
(AI GENERATED)

### ✅ STRENGTHS

1. **Modular Architecture**
   Each recognizer (`TextEmotionRecognizer`, `SpeechEmotionRecognizer`, `FaceEmotionRecognizer`) is cleanly separated and can be reused or extended individually.

2. **Multi-modal Support**
   You handle audio, video, text, and images in an integrated yet independent way, allowing for future fusion strategies.

3. **Thoughtful Fallbacks**
   Checks like `"if 'audio' in processed_data and processed_data['audio']"` prevent crashes on edge cases.

4. **LLM Usage with Prompt Control**
   Your use of `system` roles to constrain LLM output to your emotion classes is textbook prompt engineering.

---

### ❌ WEAKNESSES & IMPROVEMENTS 

#### 🔹1. **LLM Calls Are Blocking Everything**

##### 💡 Problem:

You are **calling the LLM synchronously**, which blocks the entire program (including UI or video/audio processing) until a reply comes back.

##### ✅ Solution:

Use **concurrent execution**:

* Replace synchronous `completion()` calls with a thread or async wrapper.
* For Jupyter: use `asyncio` + `nest_asyncio`.
* Or run LLM calls inside a `ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor

self.executor = ThreadPoolExecutor(max_workers=2)

def analyze_async(self, text):
    return self.executor.submit(self.analyze, text)
```

Also, decouple your LLM calls by **deferring** them after pre-processing:

* First store preprocessed segments
* Let LLM emotion analysis happen **asynchronously** in the background
* Meanwhile return partial results (e.g., transcription)

---

#### 🔹2. **Synchronous Whisper Usage**

Whisper transcription is done inline with `transcribe(audio)`, which is blocking.

##### ✅ Suggestion:

Transcribe *first* to segments (timestamps), then process audio in chunks later or in parallel.

```python
import threading

def transcribe_async(self, audio):
    thread = threading.Thread(target=self.transcriber.transcribe, args=(audio,))
    thread.start()
```

Or break Whisper into async steps using `segments = model.transcribe(..., return_segments=True)`

---

#### 🔹3. **Lack of GPU Utilization Control**

If run on CPU, Whisper and DeepFace will be **very slow**. Consider:

* Let user choose device: `'cuda'`, `'mps'`, `'cpu'`
* Print which model uses GPU, and warn if CPU fallback occurs

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
```

---

#### 🔹4. **No Emotion Fusion Across Modalities**

Right now, each recognizer returns its own prediction. But in reality, **multi-modal fusion** boosts robustness.

##### ✅ Suggestion:

* Fuse predictions (e.g. via weighted voting or LLM aggregation)
* Add a `FusionEngine` that takes all outputs and returns a final label
* Let the chatbot reason over contradictions

---

#### 🔹5. **Video Face Detection Needs Optimization**

You detect face **on every N-th frame**, but DeepFace will still be slow.

##### ✅ Suggestions:

* Resize frames before detection: `frame = cv2.resize(frame, (320, 240))`
* Add a `max_faces` or `confidence_threshold` to reduce false positives
* Consider `face_recognizer.analyze_video_frames(frames[:10])` to limit evaluation on very long videos

---

#### 🔹6. **LLM Prompt Repetition**

You're repeating your list of emotions in multiple prompts.

##### ✅ Suggestions:

* Store emotion set in a class variable or config
* Use a template engine like `jinja2` or simple `.format()` string reuse

---

#### 🔹7. **No Handling for Overlapping Modalities**

If a video contains both text (via speech), image (face), and audio, the same content may be double-counted.

##### ✅ Suggestion:

* Implement a `SegmentAligner` that merges overlapping timestamps
* Optionally run **multimodal consensus** (e.g., if 2 of 3 say "angry", trust it)

---

### 🌟 INNOVATION IDEAS

1. **Emotion Timeline**
   Visualize emotion over time in video/audio as a graph using `matplotlib` or `plotly`.

2. **Multilingual Emotion Analysis**
   Detect language first → then send to correct LLM (or translate text).

3. **Emotion Uncertainty**
   Have your LLM or SVM models return **confidence scores** (e.g., softmax probabilities). Highlight ambiguous inputs.

4. **Prompt Optimization via Feedback**
   Store LLM inputs and outputs and allow user to flag errors → fine-tune prompt logic later.

5. **Contextual Memory in LLM**
   Keep context across user turns (e.g., chatbot remembers past emotional states and adapts responses).

---

### 🔧 Small Code Fixes

* **Bug**: In `_process_audio()`, Whisper expects a file path or file-like object, not a NumPy array. You're giving `audio` directly.

  ```python
  transcript = self.transcriber.transcribe(file_path)  # instead of raw audio
  ```

* **Optimization**: Don't load Haar cascade twice:

  ```python
  if not self.face_cascade:
      self.face_cascade = cv2.CascadeClassifier(...)
  ```

---

### 🤖 Final Suggestions for LLM Use

| Goal                            | Suggestion                                                     |
| ------------------------------- | -------------------------------------------------------------- |
| Don't block app while LLM runs  | Run in `ThreadPoolExecutor` or `asyncio`                       |
| Reduce token count (cost/speed) | Use concise prompts and limit context                          |
| Get confidence level            | Ask LLM to rate its certainty (if using natural language)      |
| Faster small models             | Use models like `phi3-mini`, `stablelm-zephyr`, or `tinyllama` |
| Response truncation prevention  | Set `max_tokens` properly and check `.finish_reason`           |

---

Would you like me to:

* Implement a threaded or async version of the LLM handler?
* Add a fusion module to combine modalities?
* Help you create visualizations (emotion over time, confidence)?

Let me know how deep you'd like to go in each area.
