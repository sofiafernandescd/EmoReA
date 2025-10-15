import React from "react";

const AnalysisResult = ({ result }) => {
  if (!result) return null;
  return (
    <div>
      <h3>Analysis Result:</h3>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
};

export default AnalysisResult;
