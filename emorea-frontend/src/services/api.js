// src/services/api.js
const BASE_URL = "http://localhost:8000";

export async function analyzeFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${BASE_URL}/analyze/`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Analysis failed");
  }

  return response.json();
}

export async function chatWithAssistant(message) {
  const response = await fetch(`${BASE_URL}/chat/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_input: message }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Chat request failed");
  }

  return response.text();
}
