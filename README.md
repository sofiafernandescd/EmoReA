# EmoReA: Emotion Recognition Assistant

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18.0%2B-61DAFB?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688?logo=fastapi&logoColor=white)

> **Code developed for the Master's Thesis:**  
> **_Emotion Recognition in Multimedia Content_**  
> Master’s in Computer Science and Multimedia at Lisbon School of Engineering (ISEL), 2025  


EmoReA is an Emotion Recognition Assistant designed for multimodal emotion analysis. It integrates a user-friendly web interface (React frontend) with a Python backend API to process and understand emotions from various data types like text, audio, image, and video.

## Sub-Repositories

-   **`emorea-frontend/`**: Contains the React-based frontend application for user interaction and visualization. [Link to frontend README](emorea-frontend/README.md)
-   **`emorea-backend/`**: Contains the FastAPI backend API responsible for data processing, emotion analysis, and the chatbot functionality. [Link to backend README](emorea-backend/README.md)

## Installation

To get started with EmoReA, you'll need to install the dependencies for both the frontend and backend. Please follow the detailed instructions provided in the README files of each sub-repository:

-   [Frontend Installation](emorea-frontend/README.md#installation)
-   [Backend Installation](emorea-backend/README.md#installation)

## Running the Application

Instructions for running the frontend and backend development servers are provided in their respective README files:

-   [Running the Frontend](emorea-frontend/README.md#running-the-frontend)
-   [Running the Backend](emorea-backend/README.md#running-the-backend)

## Deployment

### Local Deployment

To run EmoReA on your local machine for development or testing:

1.  **Backend:** Navigate to the `emorea-backend` directory and follow the instructions in its README to install dependencies and run the FastAPI development server (typically using Uvicorn). This will usually run on `http://localhost:8000`.

2.  **Frontend:** Open a new terminal, navigate to the `emorea-frontend` directory, and follow the instructions in its README to install dependencies and start the React development server (usually using `npm start` or `yarn start`). This will typically run on `http://localhost:3000`. The frontend is configured to communicate with the backend at `http://localhost:8000`.

### Google Cloud Platform (GCP) Deployment

Deploying EmoReA to GCP involves containerizing the application (using Docker) and then deploying it using a suitable GCP service. Here's a high-level overview:

1.  **Containerization (Docker):**
    -   Create a `Dockerfile` for your backend in the `emorea-backend` directory. This file will define the environment and steps to build a Docker image for your FastAPI application.
    -   Create a `Dockerfile` for your frontend in the `emorea-frontend` directory. This will define how to build and serve your React application (e.g., using Nginx or a Node.js production server like `serve`).
    -   Build the Docker images for both the backend and frontend.
    -   Push these images to a container registry like Google Container Registry (GCR).

2.  **GCP Service Selection:**
    -   **Backend:** Consider using **Google Cloud Run**. It's a fully managed serverless platform that allows you to run stateless containers. It's cost-effective and scales automatically.
    -   **Frontend:** You can also deploy the frontend container to **Google Cloud Run** or use **Firebase Hosting** or **Cloud Storage with Cloud CDN** for serving static websites, depending on your frontend's build output.

3.  **Deployment Configuration:**
    -   **Cloud Run (Backend):** Configure a Cloud Run service with the backend Docker image from GCR. Set environment variables (if needed), configure scaling options, and expose the backend port (e.g., 8080).
    -   **Cloud Run (Frontend) / Firebase Hosting / Cloud Storage:** Configure the deployment of your frontend build artifacts. If using Cloud Run, ensure it's configured to serve the static content.

4.  **API Endpoint and CORS:**
    -   Ensure your frontend is configured to communicate with the deployed backend API endpoint (the Cloud Run URL).
    -   Update the CORS settings in your backend (`emorea-backend/main.py`) to allow requests from your deployed frontend URL.

5.  **CI/CD (Optional but Recommended):**
    -   Set up a CI/CD pipeline (e.g., using GitHub Actions and Google Cloud Build) to automate the process of building, testing, and deploying your application whenever you push changes to your repository.

## Contributing

Future work.

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.