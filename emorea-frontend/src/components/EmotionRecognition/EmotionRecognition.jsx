import React, { useState } from "react";
import WebcamCapture from "./WebcamCapture";
import FileUpload from "./FileUpload";
import AnalysisResult from "./AnalysisResult";
import ChatAssistant from "./ChatAssistant";
import '../../App.css'; // CSS

const EmotionRecognition = () => {
  const [activeTab, setActiveTab] = useState("webcam");
  const [analysisResult, setAnalysisResult] = useState(null);

  return (
    // main container
    <div className="emotion-recognition-container"> 
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button 
            onClick={() => setActiveTab("webcam")}
            className={`tab-button ${activeTab === 'webcam' ? 'active' : 'inactive'}`}
        >
            Webcam
        </button>
        <button 
            onClick={() => setActiveTab("upload")}
            className={`tab-button ${activeTab === 'upload' ? 'active' : 'inactive'}`}
        >
            File Upload
        </button>
      </div>

      {activeTab === "webcam" ? (
        <WebcamCapture onAnalysisComplete={setAnalysisResult} />
      ) : (
        <FileUpload onAnalysisComplete={setAnalysisResult} />
      )}

      <AnalysisResult result={analysisResult} />
      <ChatAssistant enabled={!!analysisResult} />
    </div>
  );
};

export default EmotionRecognition;