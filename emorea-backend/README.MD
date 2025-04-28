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
poetry run uvicorn main:app --reload
# or
source venv/bin/activate && uvicorn main:app --reload
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