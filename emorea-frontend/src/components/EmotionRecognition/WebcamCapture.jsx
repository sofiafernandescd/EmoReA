import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import RecordRTC from "recordrtc";
import { analyzeFile } from "../../services/api";
import '../../App.css'; // CSS

const WebcamCapture = ({ onAnalysisComplete }) => {
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [preview, setPreview] = useState(null);

  const capturePhoto = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    const blob = await (await fetch(imageSrc)).blob();

    const result = await analyzeFile(new File([blob], "photo.jpg"));
    onAnalysisComplete(result);
    setPreview(imageSrc);
  };

  const startRecording = () => {
    const stream = webcamRef.current.stream;
    mediaRecorderRef.current = new RecordRTC(stream, { type: "video", mimeType: 'video/mp4'});
    mediaRecorderRef.current.startRecording();
    setRecording(true);
  };

  const stopRecording = async () => {
    setRecording(false);
    mediaRecorderRef.current.stopRecording(async () => {
      const blob = mediaRecorderRef.current.getBlob();
      const result = await analyzeFile(new File([blob], "video.mp4"));
      onAnalysisComplete(result);
      setPreview(URL.createObjectURL(blob));
    });
  };

  
  return (
    <div className="webcam-capture-container"> {/* new container class for layout */}
      <Webcam ref={webcamRef} audio={true} muted={true} screenshotFormat="image/jpeg" className="webcam-stream" />
      <div className="webcam-controls">
        <button 
            onClick={capturePhoto} 
            className="action-button record-button" // class for photo
        >
            Take photo
        </button>
        <button 
            onClick={recording ? stopRecording : startRecording}
            className={`action-button record-button ${recording ? 'recording-active' : ''}`} // class for record/stop
        >
            {recording ? "stop" : "record"}
        </button>
      </div>
      {preview && (
        <div className="preview-section">
          <h4 className="preview-title">preview:</h4>
          {/* using video or img based on recording state */}
          {preview.endsWith('.mp4') ? ( 
            <video controls src={preview} className="preview-media" />
          ) : (
            <img src={preview} alt="preview" className="preview-media" />
          )}
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;
