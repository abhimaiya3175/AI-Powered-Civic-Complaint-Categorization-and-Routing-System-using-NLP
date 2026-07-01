import os
import pytest
import random
from PIL import Image
from datetime import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app, get_db
from image_features import (
    load_yolo_model,
    _run_yolo_inference_sync,
    analyze_image,
)

# Use test client
client = TestClient(app)

def create_image_with_exif(filename, lat=12.9716, lon=77.5946):
    img = Image.new('RGB', (100, 100), color='blue')
    exif = img.getexif()
    gps_ifd = exif.get_ifd(34853)
    
    lat_deg = int(lat)
    lat_min = int((lat - lat_deg) * 60)
    lat_sec = round(((lat - lat_deg) * 60 - lat_min) * 60, 4)
    gps_ifd[1] = 'N'
    gps_ifd[2] = (float(lat_deg), float(lat_min), float(lat_sec))
    
    lon_deg = int(lon)
    lon_min = int((lon - lon_deg) * 60)
    lon_sec = round(((lon - lon_deg) * 60 - lon_min) * 60, 4)
    gps_ifd[3] = 'E'
    gps_ifd[4] = (float(lon_deg), float(lon_min), float(lon_sec))
    
    current_utc = datetime.utcnow().strftime("%Y:%m:%d %H:%M:%S")
    exif[36867] = current_utc
    
    img.save(filename, exif=exif)

# ==================== FIXTURE MOCKS ====================

@pytest.fixture
def mock_yolo():
    """Mock the YOLO model to simulate a loaded checkpoint without weights."""
    with patch("image_features.yolo_model") as mock_model:
        yield mock_model

@pytest.fixture
def mock_analyze_image():
    """Bypass the actual image analysis during full endpoint tests."""
    with patch("main.analyze_image") as mock_analyze:
        mock_analyze.return_value = {
            "detections": [
                {
                    "class": "pothole",
                    "confidence": 0.85,
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "mask_polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                    "mask_area_ratio": 0.08,
                    "severity": "High"
                }
            ],
            "severity": "High",
            "annotated_image_path": None,
        }
        yield mock_analyze

@pytest.fixture
def mock_analyze_image_clear():
    """Simulate image analysis finding no potholes."""
    with patch("main.analyze_image") as mock_analyze:
        mock_analyze.return_value = {
            "detections": [],
            "severity": "Clear",
            "annotated_image_path": None,
        }
        yield mock_analyze

# ==================== UNIT TESTS ====================

@pytest.mark.asyncio
async def test_analyze_image_model_missing():
    """If YOLO model is not loaded, it should return gracefully with severity=None."""
    # Ensure yolo_model is None
    with patch("image_features.yolo_model", None):
        result = await analyze_image("dummy_path.jpg")
        
        assert result["detections"] == []
        assert result["severity"] is None
        assert result["annotated_image_path"] is None

def test_run_yolo_inference_sync_logic(mock_yolo):
    """Test the synchronous inference extraction logic with mocked Ultralytics results."""
    
    # Create a mock result object that mimics Ultralytics YOLO predictions
    mock_result = MagicMock()
    
    # Mock boxes
    mock_result.boxes = MagicMock()
    mock_result.boxes.__len__.return_value = 1
    mock_result.boxes.cls = [MagicMock(item=lambda: 0)]
    mock_result.boxes.conf = [MagicMock(item=lambda: 0.92)]
    mock_result.boxes.xyxy = [MagicMock(tolist=lambda: [100.0, 100.0, 200.0, 200.0])]
    mock_result.names = {0: "pothole"}
    
    # Mock masks
    mock_result.masks = MagicMock()
    mock_result.masks.xy = [MagicMock(tolist=lambda: [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])]
    
    mock_yolo.predict.return_value = [mock_result]
    
    # We need to mock PIL Image to avoid reading from disk
    with patch("image_features.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.size = (800, 600)  # Original size
        
        # When resized to 640 max dim, it will be 640x480
        mock_img.resize.return_value = mock_img
        mock_open.return_value = mock_img
        
        result = _run_yolo_inference_sync("dummy.jpg")
        
        assert len(result["detections"]) == 1
        d = result["detections"][0]
        
        assert d["class"] == "pothole"
        assert d["confidence"] == 0.92
        
        # BBox coordinates should be normalized relative to 640x480 inference size
        assert d["bbox"] == [round(100/640, 6), round(100/480, 6), round(200/640, 6), round(200/480, 6)]
        
        # Area = 100x100 = 10000. Image area = 640x480 = 307200. Ratio = 10000/307200 ≈ 0.0325
        assert abs(d["mask_area_ratio"] - (10000 / 307200)) < 0.001
        
        # 0.0325 is >= 0.02 (Medium) but < 0.07 (High) -> Medium
        assert d["severity"] == "Medium"
        
        assert result["image_width"] == 800
        assert result["image_height"] == 600

# ==================== ENDPOINT TESTS ====================

def test_submit_complaint_with_image_damage(mock_analyze_image):
    """Test complaint submission pipeline when image analysis detects damage."""
    
    with patch("main.whisper_model") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": "Huge pothole here", "language": "en"}
        
        # We need an image to trigger the stage.
        # We'll use a dummy text file but tell FastAPI it's an image.
        test_lat = round(random.uniform(12.75, 13.10), 4)
        test_file_path = "test_dummy.jpg"
        create_image_with_exif(test_file_path, lat=test_lat, lon=77.5946)
            
        try:
            with open(test_file_path, "rb") as f:
                response = client.post(
                    "/submit-complaint",
                    data={
                        "live_latitude": test_lat,
                        "live_longitude": 77.5946,
                        "live_location_timestamp": "2024-01-01T10:00:00Z",
                    },
                    files={"file": ("test.wav", b"dummy_audio", "audio/wav"), "image": ("test.jpg", f, "image/jpeg")},
                )
            
            if response.status_code != 200:
                print(response.json())
            if response.status_code != 200:
                print(response.json())
            assert response.status_code == 200
            data = response.json()
            
            # Verify image analysis results made it to the response
            assert data["pothole_severity"] == "High"
            assert "detected_objects" in data
            assert len(data["detected_objects"]) == 1
            assert data["detected_objects"][0]["class"] == "pothole"
            
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

def test_submit_complaint_with_clear_image(mock_analyze_image_clear):
    """Test complaint submission pipeline when image analysis finds no damage (Clear)."""
    
    with patch("main.whisper_model") as mock_whisper:
        mock_whisper.transcribe.return_value = {"text": "Road looks fine", "language": "en"}
        test_lat = round(random.uniform(12.75, 13.10), 4)
        test_file_path = "test_dummy_clear.jpg"
        create_image_with_exif(test_file_path, lat=test_lat, lon=77.5946)
            
        try:
            with open(test_file_path, "rb") as f:
                response = client.post(
                    "/submit-complaint",
                    data={
                        "live_latitude": test_lat,
                        "live_longitude": 77.5946,
                        "live_location_timestamp": "2024-01-01T10:00:00Z",
                    },
                    files={"file": ("test.wav", b"dummy_audio", "audio/wav"), "image": ("test.jpg", f, "image/jpeg")},
                )
            
            if response.status_code != 200:
                print(response.json())
            assert response.status_code == 200
            data = response.json()
            
            # Severity should be "Clear"
            assert data["pothole_severity"] == "Clear"
            assert data["detected_objects"] == []
            
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

def test_submit_complaint_category_mismatch():
    """Test that a high-confidence visual mismatch downgrades trust but preserves original text category."""
    
    with patch("main.whisper_model") as mock_whisper, \
         patch("main.analyze_image") as mock_analyze:
        
        # Text model will predict "Road Repair" (based on "pothole")
        mock_whisper.transcribe.return_value = {"text": "Huge pothole here", "language": "en"}
        
        # Image model finds "garbage_pile" with high confidence (0.85 > 0.6 threshold)
        mock_analyze.return_value = {
            "detections": [
                {
                    "class": "garbage_pile",
                    "confidence": 0.85,
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "mask_polygon": [],
                    "mask_area_ratio": 0.08,
                    "severity": "High"
                }
            ],
            "severity": "High",
            "annotated_image_path": None,
        }
        
        test_lat = round(random.uniform(12.75, 13.10), 4)
        test_file_path = "test_dummy_mismatch.jpg"
        create_image_with_exif(test_file_path, lat=test_lat, lon=77.5946)
            
        try:
            with open(test_file_path, "rb") as f:
                response = client.post(
                    "/submit-complaint",
                    data={
                        "live_latitude": test_lat,
                        "live_longitude": 77.5946,
                        "live_location_timestamp": "2024-01-01T10:00:00Z",
                    },
                    files={"file": ("test.wav", b"dummy_audio", "audio/wav"), "image": ("test.jpg", f, "image/jpeg")},
                )
            
            if response.status_code != 200:
                print(response.json())
            assert response.status_code == 200
            data = response.json()
            
            # The text classification MUST remain Road Repair
            assert data.get("category") == "Road Repair", f"Expected Road Repair, got {data}"
            # The mismatch flag should be true
            assert "category_mismatch" in data, f"Missing category_mismatch in {data}"
            assert data["category_mismatch"] is True
            assert data["image_suggested_category"] == "Garbage / Sanitation"
            # Trust level should be downgraded to manual_review
            assert data["trust_level"] == "manual_review"
            
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

def test_submit_complaint_category_match():
    """Test that if image and text categories agree, trust_level is NOT downgraded and mismatch is False."""
    
    with patch("main.whisper_model") as mock_whisper, \
         patch("main.analyze_image") as mock_analyze:
        
        # Text model predicts "Road Repair"
        mock_whisper.transcribe.return_value = {"text": "Huge pothole here", "language": "en"}
        
        # Image model finds "pothole" -> matches "Road Repair"
        mock_analyze.return_value = {
            "detections": [
                {
                    "class": "pothole",
                    "confidence": 0.85,
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "mask_polygon": [],
                    "mask_area_ratio": 0.08,
                    "severity": "High"
                }
            ],
            "severity": "High",
            "annotated_image_path": None,
        }
        
        test_lat = round(random.uniform(12.75, 13.10), 4)
        test_file_path = "test_dummy_match.jpg"
        create_image_with_exif(test_file_path, lat=test_lat, lon=77.5946)
            
        try:
            with open(test_file_path, "rb") as f:
                response = client.post(
                    "/submit-complaint",
                    data={
                        "live_latitude": test_lat,
                        "live_longitude": 77.5946,
                        "live_location_timestamp": "2024-01-01T10:00:00Z",
                    },
                    files={"file": ("test.wav", b"dummy_audio", "audio/wav"), "image": ("test.jpg", f, "image/jpeg")},
                )
            
            if response.status_code != 200:
                print(response.json())
            assert response.status_code == 200
            data = response.json()
            
            assert data["category"] == "Road Repair"
            assert data["category_mismatch"] is False
            assert data["image_suggested_category"] == "Road Repair"
            # Trust level should remain "high" (or whatever EXIF granted it, since they matched)
            # We don't have real EXIF in this test dummy file so it falls back to medium or fails EXIF parsing?
            # Wait, the dummy image won't have EXIF so it'll get "medium" and stay "medium", NOT manual_review
            assert data["trust_level"] != "manual_review"
            
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
