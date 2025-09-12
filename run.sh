#!/bin/bash

if ! command -v brew &> /dev/null
    then
        echo "⚠️ Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi

# Backend
cd emorea-backend
# Create a virtual environment and install dependencies
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate

#python3.10 -m venv venv
#python3.8 -m venv venv
# Activate the virtual environment
#source venv/bin/activate
# Install FFMPEG
brew install ffmpeg
# Install the required packages using requirements.txt
# pip install -r requirements.txt
pip install --upgrade pip setuptools wheel
# Install project and required packages using setup.py
pip install -e . --use-pep517
# Run the backend server (FastAPI)
# uvicorn app.main:app --reload

# Frontend
cd ../emorea-frontend
# Make sure to have Node.js and npm installed
if ! command -v node &> /dev/null || ! command -v npm &> /dev/null
then
    echo "⚠️ Node.js and npm not found. Installing..."
    # If brew is not installed, install it
    brew install node

fi    

# Install the required packages
npm install
# Start the frontend server (React)
# npm start