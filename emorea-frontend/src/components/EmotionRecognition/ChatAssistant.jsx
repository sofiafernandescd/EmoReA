import React, { useState } from "react";
import { chatWithAssistant } from "../../services/api";

const ChatAssistant = ({ enabled }) => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!enabled) {
      alert("Please analyze a file first!");
      return;
    }

    const reply = await chatWithAssistant(userInput);
    setMessages([
      ...messages,
      { sender: "user", text: userInput },
      { sender: "assistant", text: reply },
    ]);
    setUserInput("");
  };

  return (
    <div>
      <h3>Chat with Assistant</h3>
      <div style={{ border: "1px solid #ccc", height: 200, overflowY: "scroll" }}>
        {messages.map((m, i) => (
          <div key={i}>
            <strong>{m.sender}:</strong> {m.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Ask something..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatAssistant;
