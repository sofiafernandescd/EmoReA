import React, { useState, useRef, useEffect } from "react";
import { chatWithAssistant } from "../../services/api";

const SUGGESTED_PROMPTS = [
  "What can I improve in my presentation?",
  "How was my body language?",
  "What can I improve in the content?",
  "Analyze my vocal tone."
];

// function to safely format markdown bold (e.g., **text** to <strong>text</strong>)
const formatassistanttext = (text) => {
  // warning: use a sanitizing library like dompurify in production!
  let formattedtext = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  return formattedtext;
};

// function to clean raw text response from the assistant
const cleanResponseText = (text) => {
  if (typeof text !== 'string') return "";

  let cleaned = text.trim()
    // remove starting/ending quotes
    .replace(/^"|"$/g, '')
    // replace literal \n with real newlines
    .replace(/\\n/g, '\n')
    // replace double backslashes \\ with a single \
    .replace(/\\\\/g, '\\')
    // optional: remove stray escape sequences like \" or \'
    .replace(/\\"/g, '"')
    .replace(/\\'/g, "'");

  return cleaned.trim();
};

const ChatAssistant = ({ enabled }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const messagesEndRef = useRef(null);

  // automatically scroll to the bottom whenever messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);


  // old function
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!enabled) {
      alert("Please analyze a file first!");
      return;
    }

    const rawreply = await chatWithAssistant(userInput);
    const cleanedreply = cleanResponseText(rawreply); 

    setMessages([
      ...messages,
      { sender: "user", text: userInput },
      { sender: "assistant", text: cleanedreply },
    ]);
    setUserInput("");
  };

  // more dynamic chatbot UI
  const handleSend = async (textToSend) => {
    if (!enabled || !textToSend.trim() || isLoading) return;

    setIsLoading(true);
    
    // add user message 
    const newMessages = [...messages, { sender: "user", text: textToSend }];
    setMessages(newMessages);
    setUserInput("");

    try {
      const rawreply = await chatWithAssistant(textToSend);
      const cleanedreply = cleanResponseText(rawreply);

      setMessages([...newMessages, { sender: "assistant", text: cleanedreply }]);
    } catch (error) {
      console.error("Chat failed", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-assistant-container">
      <h3>Chat with Assistant</h3>
      
      <div className="chat-messages-scroll">
        {messages.map((m, i) => (
          <div key={i} className={`chat-message ${m.sender} slide-up`}>
            <strong>{m.sender === 'user' ? 'You: ' : 'Assistant: '}</strong>
            <span dangerouslySetInnerHTML={{ __html: formatassistanttext(m.text) }} />
          </div>
        ))}
        {/* Invisible element to anchor the scroll */}
        <div ref={messagesEndRef} />
        {isLoading && <div className="chat-message assistant thinking">Assistant is typing...</div>}
      </div>

      {/* Suggested Options Chips */}
      <div className="suggested-prompts">
        {SUGGESTED_PROMPTS.map((prompt, idx) => (
          <button 
            key={idx} 
            className="suggestion-chip"
            onClick={() => handleSend(prompt)}
            disabled={!enabled || isLoading}
          >
            {prompt}
          </button>
        ))}
      </div>

      <form onSubmit={(e) => { e.preventDefault(); handleSend(userInput); }} className="chat-form">
        <input
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Ask something..."
          className="chat-input"
        />
        <button 
          type="submit" 
          className={`chat-send-button ${isLoading ? 'loading' : ''}`}
          disabled={isLoading || !enabled}
        >
          <span className="button-text">{isLoading ? "Thinking" : "Send"}</span>
          {isLoading && <span className="dot-loader"></span>}
        </button>
      </form>
    </div>
  );
};

export default ChatAssistant;