import React, { useState } from "react";
import { analyzeFile } from "../../services/api";
import "../../App.css";

const FileUpload = ({ onAnalysisComplete, onPreview }) => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    setFileName(selected.name);

    // Determine preview type
    let previewType = "document";
    if (selected.type.startsWith("image/")) previewType = "image";
    else if (selected.type.startsWith("video/")) previewType = "video";
    else if (selected.type.startsWith("audio/")) previewType = "audio";

    const previewSrc = URL.createObjectURL(selected);
    if (onPreview) onPreview({ src: previewSrc, type: previewType });
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const result = await analyzeFile(file);
      onAnalysisComplete(result);
    } catch (error) {
      console.error("Error analyzing file:", error);
      alert("An error occurred while analyzing the file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="file-upload-container">
      <input
        type="file"
        accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt"
        onChange={handleFileChange}
      />
      {fileName && <p>Selected: {fileName}</p>}

      <button
        onClick={handleUpload}
        className="action-button"
        disabled={!file || loading}
      >
        {loading ? "Analyzing..." : "Upload & Analyze"}
      </button>
    </div>
  );
};

export default FileUpload;
