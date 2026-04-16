import React, { useState } from "react";
import {
  FileText,
  Mic,
  Smile,
} from "lucide-react";
import "../../App.css";

// Define emotion colors and emoji icons
const emotionData = {
  neutral: { color: "#9e9e9e", icon: "😐" },
  happy: { color: "#4caf50", icon: "😊" },
  sad: { color: "#2196f3", icon: "😢" },
  angry: { color: "#f44336", icon: "😠" },
  fear: { color: "#9c27b0", icon: "😨" },
  disgust: { color: "#795548", icon: "🤢" },
  surprise: { color: "#ff9800", icon: "😮" },
};

// Badge for the dominant emotion
const EmotionBadge = ({ emotion }) => {
  const emo = emotionData[emotion] || { color: "#ccc", icon: "❓" };
  return (
    <span
      className="emotion-badge"
      style={{ backgroundColor: emo.color }}
      title={emotion}
    >
      {emo.icon} {emotion}
    </span>
  );
};

// Bar chart for scored emotions (normalized to 100%)
const EmotionScores = ({ scores }) => {
  const total = Object.values(scores).reduce((sum, v) => sum + v, 0);
  const normalized = Object.fromEntries(
    Object.entries(scores).map(([emo, val]) => [emo, val / total])
  );

  return (
    <div className="emotion-scores">
      {Object.entries(normalized).map(([emo, val]) => {
        const emoInfo = emotionData[emo] || { color: "#ccc", icon: "❓" };
        return (
          <div key={emo} className="emotion-score-item">
            <span className="emotion-score-label">
              {emoInfo.icon} {emo} ({(val * 100).toFixed(1)}%)
            </span>
            <div className="emotion-score-bar-bg">
              <div
                className="emotion-score-bar"
                style={{
                  width: `${val * 100}%`,
                  backgroundColor: emoInfo.color,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

// Main analysis component
const AnalysisResult = ({ result }) => {
  if (!result) return null;

  // Render each frame
  const renderSegmentImage = (segment, idx) => {
    const emotion = segment.emotion;
    const isScores =
      typeof emotion === "object" && emotion !== null && !Array.isArray(emotion);

    const dominantEmotion =
      isScores && Object.keys(emotion).length
        ? Object.entries(emotion).reduce((a, b) => (b[1] > a[1] ? b : a))[0]
        : emotion;

    return (
      <div key={idx} className="segment-card">
        <div className="segment-header">
          <b>
            {segment.start?.toFixed(1) ?? 0}s –{" "}
            {segment.end ? `${segment.end.toFixed(1)}s` : "end"}
          </b>
          {dominantEmotion && <EmotionBadge emotion={dominantEmotion} />}
        </div>

        {segment.text && <div className="segment-text">“{segment.text}”</div>}

        {isScores && <EmotionScores scores={emotion} />}
      </div>
    );
  };

  const renderSegment = (segment, idx) => {
    // 1. Extract the actual emotion value (handles nested .label or flat string)
    const rawEmotion = segment.emotion;
    const emotionValue = rawEmotion?.label || rawEmotion; 

    // 2. Check if it's a "Scores" object (for Face or Text probabilities)
    // We explicitly check if it has 'label', if so, it's NOT a score object (it's the audio data)
    const isScores =
      typeof rawEmotion === "object" && 
      rawEmotion !== null && 
      !Array.isArray(rawEmotion) && 
      !rawEmotion.label; // Ignore the audio metrics object

    const dominantEmotion =
      isScores && Object.keys(emotionValue).length
        ? Object.entries(emotionValue).reduce((a, b) => (b[1] > a[1] ? b : a))[0]
        : emotionValue;

    return (
      <div key={idx} className="segment-card">
        <div className="segment-header">
          <b>
            {segment.start?.toFixed(1) ?? 0}s –{" "}
            {segment.end ? `${segment.end.toFixed(1)}s` : "end"}
          </b>
          {dominantEmotion && <EmotionBadge emotion={dominantEmotion} />}
        </div>

        {segment.text && <div className="segment-text">“{segment.text}”</div>}

        {/* This will now only render for Face/Text scores, ignoring Audio metrics */}
        {isScores && <EmotionScores scores={emotionValue} />}
      </div>
    );
  };

  // Expandable face emotion card
  const FaceEmotionCard = ({ face, idx }) => {
    const [open, setOpen] = useState(false);
    const toggle = () => setOpen((prev) => !prev);

    return (
      <div
        key={idx}
        className={`face-card ${open ? "open" : ""}`}
        onClick={toggle}
      >
        <div className="face-card-header">
          <h4>
            Frame {idx + 1} <EmotionBadge emotion={face.dominant_emotion} />
          </h4>
          <span className="face-card-toggle">
            {open ? "▲ Hide details" : "▼ Show details"}
          </span>
        </div>
        {open && (
          <div className="face-card-body">
            <EmotionScores scores={face.emotions} />
            {face.frame && (
              <div className="face-card-image" style={{ marginTop: '15px' }}>
                <img 
                  src={face.frame} 
                  alt={`Face analysis frame ${idx + 1}`} 
                  style={{ width: '100%', borderRadius: '8px', marginTop: '10px' }} 
                />
              </div>
            )}
          </div>
        )}
      </div>
      );
    };

  return (
    <div className="analysis-container">
      {/* Title */}
      <h2 className="analysis-title">
        Emotion Analysis Report
      </h2>

      {/* Text section */}
      {result.text_emotion && (
        <section className="analysis-section">
          <h3 className="section-title">
            <FileText className="icon-section" size={18} /> Text-based Emotions
          </h3>
          {result.text_emotion.map((s, i) => renderSegment(s, i))}
        </section>
      )}

      {/* Audio section */}
      {result.audio_emotion && (
        <section className="analysis-section">
          <h3 className="section-title">
            <Mic className="icon-section" size={18} /> Voice-based Emotions
          </h3>
          {result.audio_emotion.map((s, i) => renderSegment(s, i))}
        </section>
      )}

      {/* Facial section */}
      {result.face_emotion && (
        <section className="analysis-section">
          <h3 className="section-title">
            <Smile className="icon-section" size={18} /> Facial Emotions
          </h3>

          {Array.isArray(result.face_emotion)
            ? result.face_emotion.map((f, i) => (
                <FaceEmotionCard face={f} idx={i} key={i} />
              ))
            : (
                <FaceEmotionCard face={result.face_emotion} idx={0} />
              )}
        </section>
      )}
    </div>
  );
};

export default AnalysisResult;
