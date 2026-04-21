import { useState, useRef, useEffect } from 'react';
import piexif from 'piexifjs';
import { submitComplaint, getVoterFingerprint } from '../services/api';
import '../styles/RecordComplaint.css';
import { useNavigate } from 'react-router-dom';

const LANGUAGE_OPTIONS = [
  { value: 'kn', label: 'ಕನ್ನಡ' },
  { value: 'hi', label: 'हिंदी' },
  { value: 'en', label: 'English' },
];

const TARGET_LANGUAGE_OPTIONS = [
  { value: 'en', label: 'English' },
  { value: 'kn', label: 'ಕನ್ನಡ' },
  { value: 'hi', label: 'हिंदी' },
];

const getLanguageLabel = (languageCode) => {
  const option = LANGUAGE_OPTIONS.find((item) => item.value === languageCode);
  return option?.label || (languageCode || 'Unknown');
};

const formatConfidence = (value) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A';
  }
  return `${(value * 100).toFixed(1)}%`;
};

const buildHighlightRanges = (text, terms) => {
  if (!text || !Array.isArray(terms) || terms.length === 0) {
    return [];
  }

  const normalizedTerms = [...new Set(
    terms
      .map((term) => String(term || '').trim().toLowerCase())
      .filter(Boolean)
  )].sort((a, b) => b.length - a.length);

  const loweredText = text.toLowerCase();
  const ranges = [];

  const isWordChar = (char) => /[a-z0-9]/i.test(char);

  normalizedTerms.forEach((term) => {
    let searchStart = 0;
    while (searchStart < loweredText.length) {
      const start = loweredText.indexOf(term, searchStart);
      if (start === -1) {
        break;
      }

      const end = start + term.length;
      const previousChar = start > 0 ? loweredText[start - 1] : ' ';
      const nextChar = end < loweredText.length ? loweredText[end] : ' ';
      const isBoundaryMatch = term.includes(' ') || (!isWordChar(previousChar) && !isWordChar(nextChar));

      if (isBoundaryMatch) {
        const overlaps = ranges.some((range) => !(end <= range.start || start >= range.end));
        if (!overlaps) {
          ranges.push({ start, end });
        }
      }

      searchStart = start + 1;
    }
  });

  return ranges.sort((a, b) => a.start - b.start);
};

const renderHighlightedText = (text, terms) => {
  if (!text) {
    return '';
  }

  const ranges = buildHighlightRanges(text, terms);
  if (ranges.length === 0) {
    return text;
  }

  const nodes = [];
  let cursor = 0;

  ranges.forEach((range, index) => {
    if (range.start > cursor) {
      nodes.push(<span key={`plain-${index}-${cursor}`}>{text.slice(cursor, range.start)}</span>);
    }

    nodes.push(
      <mark key={`highlight-${index}-${range.start}`} className="reason-highlight">
        {text.slice(range.start, range.end)}
      </mark>
    );

    cursor = range.end;
  });

  if (cursor < text.length) {
    nodes.push(<span key={`tail-${cursor}`}>{text.slice(cursor)}</span>);
  }

  return nodes;
};

export default function RecordComplaint() {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [liveLocation, setLiveLocation] = useState(null);
  const [locationStatus, setLocationStatus] = useState('GPS location will be captured automatically.');
  const [textNote, setTextNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [duplicateResult, setDuplicateResult] = useState(null);
  const [rejectionMessage, setRejectionMessage] = useState('');
  const [duration, setDuration] = useState(0);
  const [cameraActive, setCameraActive] = useState(false);
  const [recordingLanguage, setRecordingLanguage] = useState('kn');
  const [targetLanguage, setTargetLanguage] = useState('en');

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const cameraInputRef = useRef(null);
  const galleryInputRef = useRef(null);
  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const canvasRef = useRef(null);
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
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach(track => track.stop());
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

  const toExifDateTime = (date) => {
    const d = date instanceof Date ? date : new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    return `${year}:${month}:${day} ${hours}:${minutes}:${seconds}`;
  };

  const decimalToDmsRational = (coordinate) => {
    const absolute = Math.abs(coordinate);
    const degrees = Math.floor(absolute);
    const minutesFloat = (absolute - degrees) * 60;
    const minutes = Math.floor(minutesFloat);
    const secondsFloat = (minutesFloat - minutes) * 60;
    const seconds = Math.round(secondsFloat * 10000);

    return [
      [degrees, 1],
      [minutes, 1],
      [seconds, 10000],
    ];
  };

  const injectGeotagIntoJpeg = (jpegDataUrl, latitude, longitude, capturedAt) => {
    const exifTime = toExifDateTime(capturedAt);
    const exifObj = {
      '0th': {
        [piexif.ImageIFD.DateTime]: exifTime,
      },
      Exif: {
        [piexif.ExifIFD.DateTimeOriginal]: exifTime,
        [piexif.ExifIFD.DateTimeDigitized]: exifTime,
      },
      GPS: {
        [piexif.GPSIFD.GPSLatitudeRef]: latitude >= 0 ? 'N' : 'S',
        [piexif.GPSIFD.GPSLatitude]: decimalToDmsRational(latitude),
        [piexif.GPSIFD.GPSLongitudeRef]: longitude >= 0 ? 'E' : 'W',
        [piexif.GPSIFD.GPSLongitude]: decimalToDmsRational(longitude),
      },
      Interop: {},
      '1st': {},
      thumbnail: null,
    };

    const exifBytes = piexif.dump(exifObj);
    return piexif.insert(exifBytes, jpegDataUrl);
  };

  const getCurrentLocation = () =>
    new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported in this browser.'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            timestamp: new Date().toISOString(),
          });
        },
        (error) => {
          reject(new Error(error.message || 'Unable to capture location.'));
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
      );
    });

  const ensureLiveLocation = async ({ forceFresh = false } = {}) => {
    if (!forceFresh && liveLocation) {
      return liveLocation;
    }

    setLocationStatus('Capturing live location...');
    const nextLocation = await getCurrentLocation();
    setLiveLocation(nextLocation);
    setLocationStatus(
      `Live location captured: ${nextLocation.latitude.toFixed(6)}, ${nextLocation.longitude.toFixed(6)}`
    );
    return nextLocation;
  };

  const drawPhotoStamp = (ctx, width, height, latitude, longitude, capturedAt) => {
    const capturedDate = new Date(capturedAt);
    const datePart = `${capturedDate.getFullYear()}-${String(capturedDate.getMonth() + 1).padStart(2, '0')}-${String(capturedDate.getDate()).padStart(2, '0')}`;
    const timePart = `${String(capturedDate.getHours()).padStart(2, '0')}:${String(capturedDate.getMinutes()).padStart(2, '0')}:${String(capturedDate.getSeconds()).padStart(2, '0')}`;
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'Local';

    const stampLines = [
      `Lat: ${latitude.toFixed(6)}`,
      `Lng: ${longitude.toFixed(6)}`,
      `Date: ${datePart}`,
      `Time: ${timePart} (${timeZone})`,
    ];

    const fontSize = Math.max(18, Math.round(width * 0.022));
    const lineHeight = Math.round(fontSize * 1.35);
    const horizontalPadding = Math.round(fontSize * 0.7);
    const verticalPadding = Math.round(fontSize * 0.55);

    ctx.font = `600 ${fontSize}px Arial`;
    const maxTextWidth = Math.max(...stampLines.map((line) => ctx.measureText(line).width));

    const boxWidth = Math.ceil(maxTextWidth + horizontalPadding * 2);
    const boxHeight = Math.ceil(stampLines.length * lineHeight + verticalPadding * 2);
    const boxX = Math.max(12, width - boxWidth - 14);
    const boxY = Math.max(12, height - boxHeight - 14);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
    ctx.fillRect(boxX, boxY, boxWidth, boxHeight);

    ctx.fillStyle = '#ffffff';
    stampLines.forEach((line, index) => {
      const textY = boxY + verticalPadding + lineHeight * (index + 0.82);
      ctx.fillText(line, boxX + horizontalPadding, textY);
    });
  };

  const captureLiveLocation = () => {
    ensureLiveLocation({ forceFresh: true }).catch((error) => {
      setLiveLocation(null);
      setLocationStatus(`Location access failed: ${error.message}`);
    });
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

  /* ── Camera Capture (Desktop Webcam) ───────────────────────── */
  const startCamera = async () => {
    try {
      ensureLiveLocation({ forceFresh: true }).catch((error) => {
        setLocationStatus(`Location access failed: ${error.message}`);
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      cameraStreamRef.current = stream;
      setCameraActive(true);
      // Wait for the video element to mount, then attach the stream
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      }, 100);
    } catch (err) {
      alert('Could not access camera. Please grant camera permission or use "Upload from Gallery" instead.');
    }
  };

  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach(track => track.stop());
      cameraStreamRef.current = null;
    }
    setCameraActive(false);
  };

  const takeSnapshot = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    let locationForPhoto = liveLocation;
    try {
      locationForPhoto = await ensureLiveLocation({ forceFresh: true });
    } catch (error) {
      alert(`Unable to capture GPS location automatically: ${error.message}`);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    const capturedAt = new Date().toISOString();
    ctx.drawImage(video, 0, 0);
    drawPhotoStamp(
      ctx,
      canvas.width,
      canvas.height,
      locationForPhoto.latitude,
      locationForPhoto.longitude,
      capturedAt
    );

    try {
      const jpegDataUrl = canvas.toDataURL('image/jpeg', 0.9);
      const geotaggedDataUrl = injectGeotagIntoJpeg(
        jpegDataUrl,
        locationForPhoto.latitude,
        locationForPhoto.longitude,
        capturedAt
      );

      fetch(geotaggedDataUrl)
        .then((response) => response.blob())
        .then((blob) => {
          const file = new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' });
          if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
          setImageFile(file);
          setImagePreviewUrl(URL.createObjectURL(blob));
          stopCamera();
        })
        .catch(() => {
          alert('Failed to encode geotag metadata in photo. Please retake the photo.');
        });
    } catch {
      alert('Could not attach geotag metadata to this photo. Please retry.');
    }
  };

  /* ── Submission ──────────────────────────────────────────────── */
  const handleSubmit = async () => {
    const hasAudio = Boolean(audioBlob);
    const hasText = Boolean(textNote.trim());

    if (!hasAudio && !hasText) {
      alert('Please provide either voice audio or complaint text.');
      return;
    }

    let locationForSubmission = liveLocation;
    if (!locationForSubmission) {
      try {
        locationForSubmission = await ensureLiveLocation({ forceFresh: true });
      } catch (error) {
        alert(`Unable to capture GPS location automatically: ${error.message}`);
        return;
      }
    }

    setSubmitting(true);
    try {
      const audioFile = hasAudio
        ? new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
        : null;

      const response = await submitComplaint({
        audioFile,
        imageFile,
        liveLatitude: locationForSubmission.latitude,
        liveLongitude: locationForSubmission.longitude,
        liveLocationTimestamp: locationForSubmission.timestamp,
        textNote,
        language: recordingLanguage,
        targetLanguage,
        voterFingerprint: getVoterFingerprint(),
      });

      if (response.duplicate) {
        setDuplicateResult(response);
        setAudioBlob(null);
        setTextNote('');
        setDuration(0);
      } else {
        setResult(response);
      }
      setImageFile(null);
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
        setImagePreviewUrl('');
      }
    } catch (err) {
      const msg = err.message || 'Failed to submit complaint. Make sure the backend is running.';
      if (msg.toLowerCase().includes('irrelevant')) {
        setRejectionMessage(msg);
        setAudioBlob(null);
        setTextNote('');
        setDuration(0);
      } else {
        alert(msg);
      }
    }
    setSubmitting(false);
  };

  const reset = () => {
    setResult(null);
    setRejectionMessage('');
    setDuplicateResult(null);
    setAudioBlob(null);
    setImageFile(null);
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
      setImagePreviewUrl('');
    }
    setTextNote('');
    setDuration(0);
  };

  /* ── Duplicate Detected View ────────────────────────────────── */
  if (duplicateResult) {
    return (
      <div className="record-complaint-page gravless-container">
        <div className="success-card card" style={{ borderColor: '#F59E0B', boxShadow: '0 0 0 1px #F59E0B' }}>
          <div className="success-icon-container">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
          </div>
          <h2 style={{ color: '#D97706' }}>Duplicate Issue Detected</h2>
          <p className="success-subtext">{duplicateResult.message}</p>
          <div className="result-grid">
            <div className="result-item">
              <span className="result-label">Existing Issue ID</span>
              <span className="result-value mono">#{duplicateResult.id}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Category</span>
              <span className="result-value">{duplicateResult.category}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Status</span>
              <span className="result-value">{duplicateResult.status}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Total Votes</span>
              <span className="result-value">👍 {duplicateResult.votes}</span>
            </div>
          </div>
          <div className="success-actions">
            <button className="btn btn-secondary" onClick={reset}>Submit Another</button>
            <button className="btn btn-primary" onClick={() => window.location.href = '/complaints'}>View All Issues</button>
          </div>
        </div>
      </div>
    );
  }

  /* ── Rejection View ─────────────────────────────────────────── */
  if (rejectionMessage) {
    return (
      <div className="record-complaint-page gravless-container">
        <div className="success-card card" style={{ borderColor: 'var(--color-danger, #ef4444)', boxShadow: '0 0 0 1px var(--color-danger, #ef4444)' }}>
          <div className="success-icon-container">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="15" y1="9" x2="9" y2="15"/>
              <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
          </div>
          <h2 style={{ color: '#ef4444' }}>Submission Discarded</h2>
          <p className="success-subtext">
            Your recording does not appear to be a valid civic complaint and has been <strong>discarded automatically</strong> by the AI classifier.
          </p>
          <div className="result-transcript" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: '8px', padding: '1rem', marginTop: '1rem' }}>
            <span className="result-label">Reason</span>
            <p className="transcript-text" style={{ marginTop: '0.4rem' }}>{rejectionMessage}</p>
          </div>
          <p style={{ marginTop: '1.2rem', fontSize: '0.9rem', opacity: 0.7 }}>
            Please submit a genuine civic complaint such as a road pothole, drainage overflow, garbage issue, street light problem, etc.
          </p>
          <div className="success-actions">
            <button className="btn btn-primary" onClick={reset}>Try Again</button>
          </div>
        </div>
      </div>
    );
  }

  /* ── Success View ────────────────────────────────────────────── */
  if (result) {
    return (
      <div className="record-complaint-page gravless-container">
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
            <div className="result-item">
              <span className="result-label">Detected Language</span>
              <span className="result-value">{getLanguageLabel(result.detected_language)}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Translated To</span>
              <span className="result-value">{getLanguageLabel(result.target_language)}</span>
            </div>
          </div>

          {result.transcribed_text && (
            <div className="result-transcript">
              <span className="result-label">Original Transcribed Text</span>
              <p className="transcript-text">{result.transcribed_text}</p>
            </div>
          )}

          {result.translated_text && (
            <div className="result-transcript">
              <span className="result-label">Translated Text</span>
              <p className="transcript-text">{result.translated_text}</p>
            </div>
          )}

          {result.category_explanation && (
            <div className="result-transcript explanation-card">
              <span className="result-label">Why This Category (NLP Evidence)</span>
              <p className="explanation-meta">
                TF-IDF + Naive Bayes model · Confidence: {formatConfidence(result.category_explanation?.confidence)}
              </p>
              <p className="transcript-text">
                {renderHighlightedText(
                  result.classification_text || result.translated_text || result.transcribed_text || '',
                  result.category_explanation?.highlight_terms || []
                )}
              </p>
              {Array.isArray(result.category_explanation?.top_features) && result.category_explanation.top_features.length > 0 && (
                <div className="reason-tags" aria-label="Top words influencing category prediction">
                  {result.category_explanation.top_features.map((feature, index) => (
                    <span key={`${feature.term}-${index}`} className="reason-tag">
                      {feature.term} ({feature.importance_percent}%)
                    </span>
                  ))}
                </div>
              )}
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
    <div className="hero-section gravless-container">
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
              Refresh GPS Location
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
          <p className="image-hint">Camera photos are stamped with location and capture time. You can also upload from gallery.</p>
          <div className="image-actions">
            <button
              className="btn btn-secondary"
              type="button"
              onClick={startCamera}
              disabled={cameraActive}
              id="btn-capture-photo"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
                <circle cx="12" cy="13" r="3"/>
              </svg>
              {cameraActive ? 'Camera Open' : 'Capture Photo'}
            </button>
            <button
              className="btn btn-secondary"
              type="button"
              onClick={() => galleryInputRef.current?.click()}
              id="btn-upload-gallery"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                <circle cx="9" cy="9" r="2"/>
                <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
              </svg>
              Upload from Gallery
            </button>
          </div>

          {/* Live Camera Preview */}
          {cameraActive && (
            <div className="camera-preview-card">
              <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
              <div className="camera-controls">
                <button className="btn btn-primary" type="button" onClick={takeSnapshot}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                  </svg>
                  Take Snapshot
                </button>
                <button className="btn btn-ghost" type="button" onClick={stopCamera}>
                  Cancel
                </button>
              </div>
            </div>
          )}

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
              <button className="btn btn-ghost btn-sm" type="button" onClick={() => { setImageFile(null); if (imagePreviewUrl) { URL.revokeObjectURL(imagePreviewUrl); setImagePreviewUrl(''); } }}>
                Remove
              </button>
            </div>
          )}
        </div>

        <div className="text-note-block">
          <label htmlFor="recording-language" className="result-label">Recording Language</label>
          <select
            id="recording-language"
            className="input"
            value={recordingLanguage}
            onChange={(e) => setRecordingLanguage(e.target.value)}
          >
            {LANGUAGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>

          <label htmlFor="target-language" className="result-label">Target Translation Language</label>
          <select
            id="target-language"
            className="input"
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
          >
            {TARGET_LANGUAGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
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
