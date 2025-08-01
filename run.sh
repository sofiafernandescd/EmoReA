#!/bin/bash

# Backend
cd emorea-backend
# Create a virtual environment and install dependencies
python3.10 -m venv venv
# Activate the virtual environment
source venv/bin/activate
# Install the required packages using requirements.txt
# pip install -r requirements.txt
pip3 install --upgrade pip setuptools wheel
# Install project and required packages using setup.py
pip3 install . --use-pep517
# Run the backend server (FastAPI)
# uvicorn app.main:app --reload

# Frontend
cd ../emorea-frontend
# Make sure to have Node.js and npm installed
# Install the required packages
npm install
# Start the frontend server (React)
# npm start