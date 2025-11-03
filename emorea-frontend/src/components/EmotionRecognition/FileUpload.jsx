import React, { useState } from "react";
import { analyzeFile } from "../../services/api";

const FileUpload = ({ onAnalysisComplete }) => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    setFileName(selected.name);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const result = await analyzeFile(file);
      onAnalysisComplete(result);
    } catch (err) {
      console.error("Error analyzing file:", err);
      alert("An error occurred while analyzing the file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center" }}>
      <input type="file" onChange={handleFileChange} />
      {fileName && <p>Selected: {fileName}</p>}

      {file && (
        <button 
          onClick={handleAnalyze} 
          disabled={loading}
          className="action-button"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      )}
    </div>
  );
};

export default FileUpload;

