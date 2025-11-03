import React, { useState } from "react";
import WebcamCapture from "./WebcamCapture";
import FileUpload from "./FileUpload";
import AnalysisResult from "./AnalysisResult";
import ChatAssistant from "./ChatAssistant";
import "../../App.css";

const EmotionRecognition = () => {
  const [activeTab, setActiveTab] = useState("webcam");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [preview, setPreview] = useState(null); // { src, type }

  const renderPreview = () => {
    if (!preview) return null;

    switch (preview.type) {
      case "image":
        return <img src={preview.src} alt="Captured" className="preview-media" />;
      case "video":
        return <video controls src={preview.src} className="preview-media" />;
      case "audio":
        return <audio controls src={preview.src} className="preview-media" />;
      case "document":
        return (
          <iframe
            src={preview.src}
            title="Document preview"
            className="preview-media"
          />
        );
      default:
        return <p>Unsupported preview type.</p>;
    }
  };

  return (
    <div className="emotion-recognition-container">
      <div className="app-header">
              <h1 className="app-title">EmoReA</h1>
              <h2 className="app-subtitle">Emotion Recognition Assistant</h2>
            </div>
      <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
        <button
          onClick={() => setActiveTab("webcam")}
          className={`tab-button ${activeTab === "webcam" ? "active" : "inactive"}`}
        >
          Webcam
        </button>
        <button
          onClick={() => setActiveTab("upload")}
          className={`tab-button ${activeTab === "upload" ? "active" : "inactive"}`}
        >
          File Upload
        </button>
      </div>

      {activeTab === "webcam" ? (
        <WebcamCapture
          onAnalysisComplete={setAnalysisResult}
          onPreview={setPreview}
        />
      ) : (
        <FileUpload
          onAnalysisComplete={setAnalysisResult}
          onPreview={setPreview}
        />
      )}

      {/* Unified preview section */}
      {preview && (
        <div className="preview-section">
          <h4 className="preview-title">Preview:</h4>
          {renderPreview()}
        </div>
      )}

      <AnalysisResult result={analysisResult} />
      <ChatAssistant enabled={!!analysisResult} />
    </div>
  );
};

export default EmotionRecognition;
