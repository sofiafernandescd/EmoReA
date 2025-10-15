import React, { useState } from "react";
import WebcamCapture from "./WebcamCapture";
import FileUpload from "./FileUpload";
import AnalysisResult from "./AnalysisResult";
import ChatAssistant from "./ChatAssistant";

const EmotionRecognition = () => {
  const [activeTab, setActiveTab] = useState("webcam");
  const [analysisResult, setAnalysisResult] = useState(null);

  return (
    <div>
      <div>
        <button onClick={() => setActiveTab("webcam")}>Webcam</button>
        <button onClick={() => setActiveTab("upload")}>File Upload</button>
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
