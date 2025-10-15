import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import RecordRTC from "recordrtc";
import { analyzeFile } from "../../services/api";

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
    mediaRecorderRef.current = new RecordRTC(stream, { type: "video" });
    mediaRecorderRef.current.startRecording();
    setRecording(true);
  };

  const stopRecording = async () => {
    setRecording(false);
    mediaRecorderRef.current.stopRecording(async () => {
      const blob = mediaRecorderRef.current.getBlob();
      const result = await analyzeFile(new File([blob], "video/mp4"));
      onAnalysisComplete(result);
      setPreview(URL.createObjectURL(blob));
    });
  };

  return (
    <div>
      <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg" />
      <div>
        <button onClick={capturePhoto}>Take Photo</button>
        <button onClick={recording ? stopRecording : startRecording}>
          {recording ? "Stop" : "Record"}
        </button>
      </div>
      {preview && (
        <div>
          <h4>Preview:</h4>
          {recording ? (
            <video controls src={preview} />
          ) : (
            <img src={preview} alt="Preview" />
          )}
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;
