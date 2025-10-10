'''
 # @ Author: Sofia Condesso (50308)
 # @ Create Time: 2025-04-09 15:31:30
 # @ Description: This is the main file for the FastAPI application of 
 #                  Emotion Recognition in Multimedia Content TFM.
 # @ References: 
 #       - FastAPI documentation: https://fastapi.tiangolo.com/
 #       - https://faun.pub/mastering-api-documentation-in-python-fastapi-best-practices-for-maintainable-and-readable-code-2425f9d734f7
 '''

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from src.assistant import EmotionRecognitionAssistant
import shutil
import os
from pydantic import BaseModel

class ChatInput(BaseModel):
    user_input: str

app = FastAPI(
    title="Emotion Recognition API",
    description="An API for analyzing emotions from various media types.",
    version="0.1.0",
)

# CORS middleware to allow requests from your React frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Make sure your React app's origin is here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = EmotionRecognitionAssistant()

@app.post("/analyze/", response_model=dict, summary="Analyze emotions from a file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyzes the emotions present in the provided file (text, audio, image, or video).
    Returns a dictionary containing the analysis results for each detected modality.
    The file is temporarily saved to disk for processing and then deleted.

    Args:
        file (UploadFile): The file to be analyzed.
    Returns:
        dict: A dictionary containing the analysis results for each detected modality.
    Raises:
        HTTPException: If there is an error during file processing or analysis. 
    """
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Analyze the file using the assistant
        analysis_result = assistant.analyze(file_path)
        # Clean up temporary file
        os.remove(file_path)  
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/", response_model=str, summary="Chat with the emotion analysis assistant")
async def chat_with_assistant(input_data: ChatInput):
    """
    Sends a message to the emotion analysis assistant and receives its response
    based on the previously analyzed file.
        1. The user sends a message to the assistant.
        2. The assistant processes the message and generates a response.
        3. The response is returned to the user.
        4. The user can continue the conversation by sending more messages.
        5. The assistant maintains the context of the conversation.
        6. The user can ask questions or request clarifications.
        7. The assistant provides informative and relevant responses.
        8. The user can end the conversation at any time.
  
    Args:
        input_data (ChatInput): The user input containing the message.
    Returns:
        str: The response from the emotion analysis assistant.
    Raises:
        HTTPException: If there is an error during the chat interaction.
    """
    # Process the user input and get the assistant's response
    response = assistant.chat(input_data.user_input)
    return response

@app.get("/health/", status_code=200, summary="Health check")
async def health_check():
    """
    Returns a simple message indicating that the API is healthy.
    """
    return {"status": "healthy"}