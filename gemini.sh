#!/bin/bash

GEMINI_API_KEY=AIzaSyDOTJBWOe5K5hw4cm_rEj8FngyMcVZqFDY

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyDOTJBWOe5K5hw4cm_rEj8FngyMcVZqFDY" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "Explain how AI works"}]
    }]
   }'