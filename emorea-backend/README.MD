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
    git clone https://github.com/sofiafernandescd/EmoReA
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
The backend API will be accessible at http://localhost:8000. FastAPI also provides automatic API documentation that can be consultedat /docs and /redoc when the server is running.

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
We appreciate all contributions! Feel free to open an issue or submit a pull request to suggest improvements. All contributors will be formally acknowledged and highlighted in this section.

## License
This project is licensed under the MIT License - see the https://www.google.com/search?q=../LICENSE file for details.

## Citation

If you use this project, code, or findings in your research, please cite the corresponding publication(s) below:

### 1. Master's Thesis
> Condesso, S. F. (2025). *Emotion recognition in multimedia content* (Master's thesis, Instituto Superior de Engenharia de Lisboa). Repositório Institucional do IPL. http://handle.net

```bibtex
@mastersthesis{thesis2025emotion,
  author       = {Condesso, Sofia Fernandes and Ferreira, Artur Jorge and Leite, Nuno Miguel da Costa de Sousa},
  title        = {Emotion recognition in multimedia content},
  school       = {Instituto Superior de Engenharia de Lisboa},
  year         = {2025},
  type         = {Master's thesis},
  url          = {http://handle.net}
}
```

### 2. Conference Papers

#### Facial Emotion Recognition (2026)
> Condesso, S., Ferreira, A. J., & Leite, N. (2026). Facial Emotion Recognition: A Comparative Study with Cross-Corpus and Multi-Corpus Training. In *Proceedings of the Conference*.

```bibtex
@inproceedings{inproceedings,
author = {Condesso, Sofia and Ferreira, Artur and Leite, Nuno},
year = {2026},
month = {03},
pages = {},
title = {Facial Emotion Recognition: A Comparative Study with Cross-Corpus and Multi-Corpus Training},
doi = {10.5220/0014639300004067}
}
```

#### User Emotion Recognition (2025)
> Condesso, S., Ferreira, A. J., & Leite, N. (2025). User Emotion Recognition from Speech and Text Messages with Machine Learning Techniques. In *Proceedings of the Conference*.

```bibtex
@inproceedings{inproceedings,
author = {Condesso, Sofia and Ferreira, Artur and Leite, Nuno},
year = {2025},
month = {10},
pages = {},
title = {User Emotion Recognition from Speech and Text Messages with Machine Learning Techniques}
}
```
