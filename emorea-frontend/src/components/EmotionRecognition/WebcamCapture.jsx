import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import RecordRTC from "recordrtc";
import { analyzeFile } from "../../services/api";
import "../../App.css";

const WebcamCapture = ({ onAnalysisComplete, onPreview }) => {
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);

  const capturePhoto = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    const blob = await (await fetch(imageSrc)).blob();
    const file = new File([blob], "photo.jpg", { type: "image/jpeg" });

    const result = await analyzeFile(file);
    onAnalysisComplete(result);
    onPreview({ src: imageSrc, type: "image" });
  };

  const startRecording = () => {
    const stream = webcamRef.current.stream;
    mediaRecorderRef.current = new RecordRTC(stream, {
      type: "video",
      mimeType: "video/mp4",
    });
    mediaRecorderRef.current.startRecording();
    setRecording(true);
  };

  const stopRecording = async () => {
    setRecording(false);
    mediaRecorderRef.current.stopRecording(async () => {
      const blob = mediaRecorderRef.current.getBlob();
      const file = new File([blob], "video.mp4", { type: "video/mp4" });

      const result = await analyzeFile(file);
      onAnalysisComplete(result);

      const url = URL.createObjectURL(blob);
      onPreview({ src: url, type: "video" });
    });
  };

  return (
    <div className="webcam-capture-container">
      <Webcam
        ref={webcamRef}
        audio={true}
        muted={true}
        screenshotFormat="image/jpeg"
        className="webcam-stream"
      />

      <div className="webcam-controls">
        <button
          onClick={capturePhoto}
          className="action-button record-button"
          disabled={recording}
        >
          Take Photo
        </button>

        <button
          onClick={recording ? stopRecording : startRecording}
          className={`action-button record-button ${
            recording ? "recording-active" : ""
          }`}
        >
          {recording ? "Stop" : "Record"}
        </button>
      </div>
    </div>
  );
};

export default WebcamCapture;
