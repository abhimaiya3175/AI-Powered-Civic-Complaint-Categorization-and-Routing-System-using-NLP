import { useState, useRef, useEffect } from 'react';
import { submitComplaint } from '../services/api';
import '../styles/RecordComplaint.css';
import { useNavigate } from 'react-router-dom';

export default function RecordComplaint() {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [liveLocation, setLiveLocation] = useState(null);
  const [locationStatus, setLocationStatus] = useState('Live location required before submission.');
  const [textNote, setTextNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [duration, setDuration] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const cameraInputRef = useRef(null);
  const galleryInputRef = useRef(null);
  const navigate = useNavigate();

  /* ── Recording Controls ──────────────────────────────────────── */
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setRecording(true);
      setDuration(0);
      timerRef.current = setInterval(() => setDuration(d => d + 1), 1000);
    } catch {
      alert('Could not access microphone. Please grant permission.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      clearInterval(timerRef.current);
    }
  };

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, []);

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

  const formatDateTime = (value) => {
    try {
      return new Date(value).toLocaleString();
    } catch {
      return value;
    }
  };

  const captureLiveLocation = () => {
    if (!navigator.geolocation) {
      setLocationStatus('Geolocation is not supported in this browser.');
      return;
    }

    setLocationStatus('Capturing live location...');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const nextLocation = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          timestamp: new Date().toISOString(),
        };
        setLiveLocation(nextLocation);
        setLocationStatus(
          `Live location captured: ${nextLocation.latitude.toFixed(6)}, ${nextLocation.longitude.toFixed(6)}`
        );
      },
      (error) => {
        setLiveLocation(null);
        setLocationStatus(`Location access failed: ${error.message}`);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  };

  const handleImagePick = (event) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;

    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }

    setImageFile(nextFile);
    setImagePreviewUrl(URL.createObjectURL(nextFile));
  };

  /* ── Submission ──────────────────────────────────────────────── */
  const handleSubmit = async () => {
    const hasAudio = Boolean(audioBlob);
    const hasText = Boolean(textNote.trim());

    if (!liveLocation) {
      alert('Live location is required. Please tap Capture Live Location first.');
      return;
    }
    if (!hasAudio && !hasText) {
      alert('Please provide either voice audio or complaint text.');
      return;
    }

    setSubmitting(true);
    try {
      const audioFile = hasAudio
        ? new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
        : null;

      const response = await submitComplaint({
        audioFile,
        imageFile,
        liveLatitude: liveLocation.latitude,
        liveLongitude: liveLocation.longitude,
        liveLocationTimestamp: liveLocation.timestamp,
        textNote,
      });

      setResult(response);
      setAudioBlob(null);
      setImageFile(null);
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
        setImagePreviewUrl('');
      }
      setTextNote('');
    } catch (err) {
      alert(err.message || 'Failed to submit complaint. Make sure the backend is running.');
    }
    setSubmitting(false);
  };

  const reset = () => {
    setResult(null);
    setAudioBlob(null);
    setImageFile(null);
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
      setImagePreviewUrl('');
    }
    setTextNote('');
    setDuration(0);
  };

  /* ── Success View ────────────────────────────────────────────── */
  if (result) {
    return (
      <div className="hero-section">
        <div className="success-card card">
          <div className="success-icon-container">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          </div>
          <h2>Complaint Submitted Successfully!</h2>
          <p className="success-subtext">Your complaint has been registered and will be reviewed shortly.</p>

          <div className="result-grid">
            <div className="result-item">
              <span className="result-label">Tracking ID</span>
              <span className="result-value mono">#{result.id}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Category</span>
              <span className="result-value">{result.category}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Location</span>
              <span className="result-value">{result.location}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Status</span>
              <span className="badge badge-pending">{result.status}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Trust Level</span>
              <span className="result-value">{result.trust_level || 'medium'}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Verification</span>
              <span className="result-value">{result.verification_mode || 'manual_review'}</span>
            </div>
          </div>

          {result.translated_text && (
            <div className="result-transcript">
              <span className="result-label">Transcript</span>
              <p className="transcript-text">{result.translated_text}</p>
            </div>
          )}

          <div className="success-actions">
            <button className="btn btn-secondary" onClick={reset}>Submit Another</button>
            <button className="btn btn-primary" onClick={() => navigate('/admin')}>View Dashboard</button>
          </div>
        </div>
      </div>
    );
  }

  /* ── Main View — Hero + Voice Capture ────────────────────────── */
  return (
    <div className="hero-section">
      {/* Hero Content */}
      <div className="hero-content">
        <div className="hero-badge">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" x2="12" y1="19" y2="22"/>
          </svg>
          Voice-Powered Complaints
        </div>
        <h1 className="hero-title">Report Civic Issues<br/>in Your Language</h1>
        <p className="hero-subtitle">
          Speak in <strong>Kannada</strong>, <strong>Hindi</strong>, or <strong>English</strong>
          — our AI will transcribe, translate, and classify your complaint automatically.
        </p>
        <p className="hero-subtitle-multilingual">
          <span className="lang-tag">ಕನ್ನಡ</span>
          <span className="lang-divider">•</span>
          <span className="lang-tag">हिन्दी</span>
          <span className="lang-divider">•</span>
          <span className="lang-tag">English</span>
        </p>
      </div>

      {/* Voice Capture Card */}
      <div className="voice-capture-card card">
        <div className="location-block">
          <div className="location-header">
            <h3>Live Location</h3>
            <button className="btn btn-secondary" type="button" onClick={captureLiveLocation}>
              Capture Live Location
            </button>
          </div>
          <p className="location-status">{locationStatus}</p>
          {liveLocation && (
            <p className="location-coords mono">
              {liveLocation.latitude.toFixed(6)}, {liveLocation.longitude.toFixed(6)}
              {' · '}
              {formatDateTime(liveLocation.timestamp)}
            </p>
          )}
        </div>

        <div className="image-block">
          <h3>Image Evidence</h3>
          <p className="image-hint">Choose one option. Both buttons are always visible.</p>
          <div className="image-actions">
            <button
              className="btn btn-secondary"
              type="button"
              onClick={() => cameraInputRef.current?.click()}
              id="btn-capture-photo"
            >
              Capture Photo
            </button>
            <button
              className="btn btn-secondary"
              type="button"
              onClick={() => galleryInputRef.current?.click()}
              id="btn-upload-gallery"
            >
              Upload from Gallery
            </button>
          </div>
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png"
            capture="environment"
            onChange={handleImagePick}
            className="hidden-file-input"
          />
          <input
            ref={galleryInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png"
            onChange={handleImagePick}
            className="hidden-file-input"
          />
          {imageFile && (
            <div className="image-preview-card">
              <img src={imagePreviewUrl} alt="Evidence preview" className="evidence-preview" />
              <p className="image-file-name">{imageFile.name}</p>
            </div>
          )}
        </div>

        <div className="text-note-block">
          <label htmlFor="complaint-text" className="result-label">Additional Complaint Text (Optional)</label>
          <textarea
            id="complaint-text"
            rows={2}
            className="complaint-textarea"
            value={textNote}
            onChange={(e) => setTextNote(e.target.value)}
            placeholder="Describe the issue in text if needed..."
          />
        </div>

        {!audioBlob && !recording && (
          <div className="mic-area">
            <button className="mic-button" onClick={startRecording} id="btn-start-recording" aria-label="Start recording">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" x2="12" y1="19" y2="22"/>
              </svg>
            </button>
            <p className="mic-label">Tap to start recording</p>
            <p className="mic-hint">Supported: Kannada, Hindi, English</p>
          </div>
        )}

        {recording && (
          <div className="recording-area">
            {/* Waveform Visualization */}
            <div className="waveform" aria-label="Recording in progress">
              {Array.from({ length: 12 }).map((_, i) => (
                <div
                  key={i}
                  className="waveform-bar"
                  style={{ animationDelay: `${i * 0.08}s` }}
                />
              ))}
            </div>
            <div className="recording-info">
              <span className="recording-dot"></span>
              <span className="recording-time mono">{formatTime(duration)}</span>
              <span className="recording-label">Recording…</span>
            </div>
            <button className="btn btn-danger btn-lg" onClick={stopRecording} id="btn-stop-recording">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <rect x="4" y="4" width="16" height="16" rx="2"/>
              </svg>
              Stop Recording
            </button>
          </div>
        )}

        {audioBlob && !recording && (
          <div className="preview-area">
            <div className="audio-preview-wrapper">
              <audio src={URL.createObjectURL(audioBlob)} controls className="audio-preview-player" />
            </div>
            <div className="preview-actions">
              <button className="btn btn-secondary" onClick={() => { setAudioBlob(null); setDuration(0); }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="1 4 1 10 7 10"/>
                  <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
                </svg>
                Re-record
              </button>
              <button className="btn btn-primary btn-lg" onClick={handleSubmit} disabled={submitting} id="btn-submit-complaint">
                {submitting ? (
                  <>
                    <span className="spinner"></span>
                    Processing…
                  </>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="22" x2="11" y1="2" y2="13"/>
                      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                    </svg>
                    Submit Complaint
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {!audioBlob && !recording && (
          <div className="submit-only-actions">
            <button className="btn btn-primary btn-lg" onClick={handleSubmit} disabled={submitting} id="btn-submit-complaint-no-audio">
              {submitting ? 'Processing…' : 'Submit with Current Details'}
            </button>
          </div>
        )}
      </div>

      {/* How It Works */}
      <div className="how-it-works">
        <div className="step-card">
          <div className="step-number">1</div>
          <h4>Record</h4>
          <p>Speak your complaint in any supported language</p>
        </div>
        <div className="step-connector">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#CBD5E1" strokeWidth="2" strokeLinecap="round">
            <line x1="5" y1="12" x2="19" y2="12"/>
            <polyline points="12 5 19 12 12 19"/>
          </svg>
        </div>
        <div className="step-card">
          <div className="step-number">2</div>
          <h4>AI Processes</h4>
          <p>Transcribed, translated, and auto-classified</p>
        </div>
        <div className="step-connector">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#CBD5E1" strokeWidth="2" strokeLinecap="round">
            <line x1="5" y1="12" x2="19" y2="12"/>
            <polyline points="12 5 19 12 12 19"/>
          </svg>
        </div>
        <div className="step-card">
          <div className="step-number">3</div>
          <h4>Track Status</h4>
          <p>Get a tracking ID and follow up on your complaint</p>
        </div>
      </div>
    </div>
  );
}
