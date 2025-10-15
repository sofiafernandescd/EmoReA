import React, { useRef, useState } from 'react';
import RecordRTC from 'recordrtc';
import Webcam from 'react-webcam';



const EmotionRecognition = () => {
    const webcamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [capturedImage, setCapturedImage] = useState(null);
    const [recording, setRecording] = useState(false);
    const [recordedVideo, setRecordedVideo] = useState(null);
    const [activeTab, setActiveTab] = useState('webcam');
    const [selectedFile, setSelectedFile] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [fileType, setFileType] = useState(null);
    const [fileContent, setFileContent] = useState(null);
    const [fileName, setFileName] = useState(null);
    const allowedFileTypes = ['txt', 'pdf', 'docx', 'mp3', 'wav', 'mp4', 'webm', 'jpg', 'jpeg', 'png', 'avi'];

    const [chatMessages, setChatMessages] = useState([]);
    const [userInput, setUserInput] = useState('');
    const chatContainerRef = useRef(null);

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
    
        const fileExtension = file.name.split('.').pop().toLowerCase();
        if (!allowedFileTypes.includes(fileExtension)) {
            alert('Unsupported file type');
            return;
        }
    
        setFileType(fileExtension);
        setFileName(file.name);
        setSelectedFile(file); // Store the actual File object
        setAnalysisResult(null); // Clear previous analysis
    
        try {
            let payload = new FormData();
            payload.append('file', file);
    
            const response = await fetch('http://localhost:8000/analyze/', {
                method: 'POST',
                body: payload,
            });
    
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Error analyzing file:', errorData);
                alert(`Error analyzing file: ${errorData.detail || response.statusText}`);
                return;
            }
    
            const data = await response.json();
            setAnalysisResult(data);
            setChatMessages([]); // Clear previous chat on new analysis
        } catch (error) {
            console.error('Error processing file:', error);
            alert(`Processing failed: ${error.message}`);
        }
    };

  

    const capturePhoto = async () => {
        const imageSrc = webcamRef.current.getScreenshot();
        setCapturedImage(imageSrc);
        setAnalysisResult(null);

        try {
            const formData = new FormData();
            formData.append('file', dataURItoBlob(imageSrc), 'webcam_image.jpeg');

            const response = await fetch('http://localhost:8000/analyze/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Error analyzing image:', errorData);
                alert(`Error analyzing image: ${errorData.detail || response.statusText}`);
                return;
            }

            const data = await response.json();
            setAnalysisResult(data);
            setChatMessages([]);
        } catch (error) {
            console.error('Error uploading image:', error);
            alert(`Error uploading image: ${error.message}`);
        }
    };

    // Helper function to convert data URI to Blob
    const dataURItoBlob = (dataURI) => {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeString });
    };

    const startRecording = () => {
        setRecording(true);
        const stream = webcamRef.current.stream;
        mediaRecorderRef.current = new RecordRTC(stream, {
            type: 'video',
            mimeType: 'video/mp4',
        });
        mediaRecorderRef.current.startRecording();
    };

    const stopRecording = () => {
        setRecording(false);
        mediaRecorderRef.current.stopRecording(async () => {
            const blob = mediaRecorderRef.current.getBlob();
            setRecordedVideo(URL.createObjectURL(blob));
            setAnalysisResult(null);

            const formData = new FormData();
            formData.append('file', blob, 'webcam_video.webm');

            try {
                const response = await fetch('http://localhost:8000/analyze/', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    console.error('Error analyzing video:', errorData);
                    alert(`Error analyzing video: ${errorData.detail || response.statusText}`);
                    return;
                }

                const data = await response.json();
                setAnalysisResult(data);
                setChatMessages([]);
            } catch (error) {
                console.error('Error uploading video:', error);
                alert(`Error uploading video: ${error.message}`);
            }
        });
    };

    const handleChatSubmit = async (event) => {
        event.preventDefault();
        if (userInput && analysisResult) {
            try {
                const response = await fetch('http://localhost:8000/chat/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json', // Tell the server we're sending JSON
                    },
                    body: JSON.stringify({ user_input: userInput }), // Send user_input as JSON
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    console.error('Error chatting:', errorData);
                    alert(`Error chatting: ${errorData.detail || response.statusText}`);
                    return;
                }
                const data = await response.text();
                setChatMessages([...chatMessages, { sender: 'user', text: userInput }, { sender: 'assistant', text: data }]);
                setUserInput('');
                if (chatContainerRef.current) {
                    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
                }
            } catch (error) {
                console.error('Error chatting:', error);
                alert(`Error chatting: ${error.message}`);
            }
        } else if (!analysisResult) {
            alert('Please upload and analyze a file first to chat.');
        }
    };

    const old_handleChatSubmit = async (event) => {
        event.preventDefault();
        if (userInput && analysisResult) {
            try {
                const formData = new FormData();
                formData.append('user_input', userInput);

                const response = await fetch('http://localhost:8000/chat/', {
                    method: 'POST',
                    body: formData,
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    console.error('Error chatting:', errorData);
                    alert(`Error chatting: ${errorData.detail || response.statusText}`);
                    return;
                }
                const data = await response.text();
                setChatMessages([...chatMessages, { sender: 'user', text: userInput }, { sender: 'assistant', text: data }]);
                setUserInput('');
                if (chatContainerRef.current) {
                    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
                }
            } catch (error) {
                console.error('Error chatting:', error);
                alert(`Error chatting: ${error.message}`);
            }
        } else if (!analysisResult) {
            alert('Please upload and analyze a file first to chat.');
        }
    };

    const styles = {
        container: {
            maxWidth: '800px',
            margin: '20px auto',
            padding: '20px',
            fontFamily: 'Arial, sans-serif',
            border: '1px solid #ccc',
            borderRadius: '8px',
            backgroundColor: '#f9f9f9',
        },
        tabs: {
            marginBottom: '20px',
            display: 'flex',
            gap: '10px',
        },
        tabButton: {
            padding: '10px 20px',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            backgroundColor: '#f0f0f0',
        },
        activeTab: {
            backgroundColor: '#007bff',
            color: 'white',
        },
        webcam: {
            width: '100%',
            maxWidth: '640px',
            borderRadius: '10px',
        },
        controls: {
            margin: '20px 0',
            display: 'flex',
            gap: '10px',
            justifyContent: 'center',
        },
        button: {
            padding: '10px 20px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
        },
        recordingButton: {
            backgroundColor: '#dc3545',
        },
        previewSection: {
            marginTop: '20px',
        },
        preview: {
            maxWidth: '100%',
            borderRadius: '10px',
            marginTop: '10px',
        },
        uploadSection: {
            textAlign: 'center',
        },
        fileInput: {
            display: 'none',
        },
        uploadButton: {
            padding: '10px 20px',
            backgroundColor: '#28a745',
            color: 'white',
            borderRadius: '5px',
            cursor: 'pointer',
            display: 'inline-block',
            marginBottom: '20px',
        },
        filePreview: {
            marginTop: '20px',
        },
        analysisResultContainer: {
            marginTop: '20px',
            borderTop: '1px solid #eee',
            paddingTop: '20px',
        },
        analysisOutput: {
            backgroundColor: '#e9ecef',
            padding: '10px',
            borderRadius: '5px',
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            fontSize: '0.9em',
            marginBottom: '20px',
        },
        chatTitle: {
            marginTop: '20px',
        },
        chatContainer: {
            border: '1px solid #ccc',
            padding: '10px',
            marginBottom: '10px',
            height: '200px',
            overflowY: 'auto',
            backgroundColor: '#fff',
            borderRadius: '5px',
        },
        userMessage: {
            textAlign: 'right',
            marginBottom: '8px',
            color: '#007bff',
        },
        assistantMessage: {
            textAlign: 'left',
            marginBottom: '8px',
            color: '#28a745',
            backgroundColor: '#f8f9fa',
            padding: '8px',
            borderRadius: '5px',
            display: 'inline-block',
            maxWidth: '80%',
            wordBreak: 'break-word',
        },
        chatInputForm: {
            display: 'flex',
        },
        chatInput: {
            flexGrow: 1,
            padding: '10px',
            borderRadius: '5px',
            border: '1px solid #ccc',
            marginRight: '10px',
        },
        chatButton: {
            padding: '10px 15px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
        },
    };

    return (
        <div style={styles.container}>
            <div style={styles.tabs}>
                <button
                    style={{ ...styles.tabButton, ...(activeTab === 'webcam' && styles.activeTab) }}
                    onClick={() => setActiveTab('webcam')}
                >
                    Webcam
                </button>
                <button
                    style={{ ...styles.tabButton, ...(activeTab === 'upload' && styles.activeTab) }}
                    onClick={() => setActiveTab('upload')}
                >
                    File Upload
                </button>
            </div>

            {activeTab === 'webcam' ? (
                <div style={styles.webcamSection}>
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        style={styles.webcam}
                    />

                    <div style={styles.controls}>
                        <button style={styles.button} onClick={capturePhoto}>
                            Take Photo
                        </button>

                        <button
                            style={{ ...styles.button, ...(recording && styles.recordingButton) }}
                            onClick={recording ? stopRecording : startRecording}
                        >
                            {recording ? 'Stop Recording' : 'Start Recording'}
                        </button>
                    </div>

                    {(capturedImage || recordedVideo) && (
                        <div style={styles.previewSection}>
                            {capturedImage && (
                                <div>
                                    <h3>Captured Photo:</h3>
                                    <img src={capturedImage} alt="Captured" style={styles.preview} />
                                </div>
                            )}
                            {recordedVideo && (
                                <div>
                                    <h3>Recorded Video:</h3>
                                    <video controls src={recordedVideo} style={styles.preview} />
                                </div>
                            )}
                        </div>
                    )}
                </div>
            ) : (
                <div style={styles.uploadSection}>
                    <input
                        type="file"
                        accept="image/*,video/*, audio/*, .txt, .pdf, .docx"
                        onChange={handleFileUpload}
                        style={styles.fileInput}
                        id="fileUpload"
                    />
                    <label htmlFor="fileUpload" style={styles.uploadButton}>
                        Choose File
                    </label>
                    {selectedFile && (
                        <div style={styles.filePreview}>
                            {selectedFile.type.includes('image') ? (
                                <img src={selectedFile} alt="Uploaded" style={styles.preview} />
                            ) : selectedFile.type.includes('video') ? (
                                <video controls src={selectedFile} style={styles.preview} />
                            ) : (fileType === 'txt' || fileType === 'pdf' || fileType === 'docx') && fileContent ? (
                                <div>
                                    <h3>Conteúdo do Ficheiro:</h3>
                                    <textarea
                                        value={fileContent}
                                        style={{ width: '100%', minHeight: '200px' }}
                                        readOnly
                                    />
                                </div>
                            ) : (
                                <p>Ficheiro selecionado: {fileName}</p>
                            )}
                        </div>
                    )}
                </div>
            )}

            {analysisResult && (
                <div style={styles.analysisResultContainer}>
                    <h3>Analysis Result:</h3>
                    <pre style={styles.analysisOutput}>{JSON.stringify(analysisResult, null, 2)}</pre>

                    <h2 style={styles.chatTitle}>Chat with Assistant:</h2>
                    <div ref={chatContainerRef} style={styles.chatContainer}>
                        {chatMessages.map((msg, index) => (
                            <div key={index} style={msg.sender === 'user' ? styles.userMessage : styles.assistantMessage}>
                                <strong>{msg.sender === 'user' ? 'You' : 'Assistant'}:</strong> {msg.text}
                            </div>
                        ))}
                    </div>
                    <form onSubmit={handleChatSubmit} style={styles.chatInputForm}>
                        <input
                            type="text"
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            placeholder="Ask about the analysis"
                            style={styles.chatInput}
                        />
                        <button type="submit" style={styles.chatButton}>Send</button>
                    </form>
                </div>
            )}
        </div>
    );
};

export default EmotionRecognition;