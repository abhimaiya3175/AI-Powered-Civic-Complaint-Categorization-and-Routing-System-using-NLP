# Image Fixtures Directory

This directory contains test images for the YOLOv8n-seg pothole and road-damage detection pipeline stage.

In a real CI environment, you would place sample `.jpg` files here representing different severities of road damage, as well as clear roads, for integration testing.

- `clear_road.jpg` - Image of a road with no potholes (expected severity: Clear)
- `minor_damage.jpg` - Image of a road with small potholes (expected severity: Low/Medium)
- `severe_pothole.jpg` - Image of a road with a massive crater (expected severity: High/Severe)

The tests in `test_image_detection.py` are currently structured to pass even if the YOLO model checkpoint or these image files are missing, testing the graceful degradation paths.
