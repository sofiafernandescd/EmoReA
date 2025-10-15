import React, { useState } from "react";
import { analyzeFile } from "../../services/api";

const FileUpload = ({ onAnalysisComplete }) => {
  const [fileName, setFileName] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);
    const result = await analyzeFile(file);
    onAnalysisComplete(result);
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      {fileName && <p>Selected: {fileName}</p>}
    </div>
  );
};

export default FileUpload;
