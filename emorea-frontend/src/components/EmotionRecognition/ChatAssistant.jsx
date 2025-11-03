import React, { useState } from "react";
import { chatWithAssistant } from "../../services/api";
// you should ensure './App.css' is imported in a parent component or here if needed

// function to safely format markdown bold (e.g., **text** to <strong>text</strong>)
const formatassistanttext = (text) => {
  // warning: use a sanitizing library like dompurify in production!
  let formattedtext = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  return formattedtext;
};

// function to clean raw text response from the assistant
const cleanresponsetext = (text) => {
  if (typeof text !== 'string') {
    return "";
  }
  // removes starting/ending quotes that might come from raw json string
  let cleaned = text.trim().replace(/^"|"$/g, ''); 
  // replaces literal newline representation (\n) with a real newline character
  cleaned = cleaned.replace(/\\n/g, '\n');
  return cleaned.trim();
};

const ChatAssistant = ({ enabled }) => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!enabled) {
      alert("Please analyze a file first!");
      return;
    }

    const rawreply = await chatWithAssistant(userInput);
    const cleanedreply = cleanresponsetext(rawreply); 

    setMessages([
      ...messages,
      { sender: "user", text: userInput },
      { sender: "assistant", text: cleanedreply },
    ]);
    setUserInput("");
  };

  return (
    <div className="chat-assistant-container">
      <h3>Chat with Assistant</h3>
      
      {/* main chat messages container */}
      <div className="chat-messages-scroll">
        {messages.map((m, i) => (
          // message styling - uses dynamic classes
          <div 
            key={i} 
            className={`chat-message ${m.sender}`} // classes 'chat-message user' or 'chat-message assistant'
          >
            <strong style={{ fontWeight: 'bold' }}>{m.sender === 'user' ? 'You' : 'Assistant'}:</strong> 
            
            {/* conditional rendering for bold markdown */}
            {m.sender === "assistant" ? (
              <span dangerouslySetInnerHTML={{ __html: formatassistanttext(m.text) }} />
            ) : (
              <span>{m.text}</span>
            )}
          </div>
        ))}
      </div>
      
      {/* input form */}
      <form onSubmit={handleSubmit} className="chat-form">
        <input
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Ask something about the detected emotions..."
          className="chat-input"
        />
        <button 
          type="submit" 
          className="chat-send-button"
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatAssistant;