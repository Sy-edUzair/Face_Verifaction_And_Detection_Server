# Face Verification API

A FastAPI-based application that performs face verification across multiple video files using advanced facial recognition and comparison techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Face Verification API performs facial verification tasks by analyzing video files to determine whether the same person appears across multiple videos. The first uploaded video is treated as the reference, and all subsequent videos are compared against it to provide verification results.

## ✨ Features

- **Multi-video Face Verification**: Compare faces across multiple video files
- **Reference-based Matching**: First video serves as the reference for comparison
- **Face Detection & Extraction**: Automatically detects and extracts faces from videos
- **Detailed Verification Results**: Provides per-video verification details
- **Face Crops Storage**: Saves face images to local storage for review
- **RESTful API**: Easy-to-use API endpoints for integration
- **Health Monitoring**: Built-in health check endpoint
- **Logging & Error Handling**: Comprehensive logging and error management

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Server**: Uvicorn
- **Production**: Gunicorn
- **Face Detection**: MTCNN, DeepFace
- **Computer Vision**: OpenCV
- **ML/DL**: TensorFlow, Keras
- **Language**: Python 3.x

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup Steps

1. **Clone/Navigate to the project directory**:
   ```bash
   cd Face_Verifacation
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python run.py
   ```

   Or with uvicorn directly:
   ```bash
   uvicorn app.run:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
Face_Verifacation/
├── app/
│   ├── __init__.py              # Package initialization
│   ├── app_logging.py           # Logging configuration
│   ├── config.py                # Application configuration
│   ├── pipeline.py              # Verification pipeline orchestration
│   ├── routes.py                # API route definitions
│   ├── run.py                   # FastAPI app creation
│   ├── schemas.py               # Pydantic models for request/response
│   ├── utils.py                 # Utility functions
│   └── services/                # Core business logic services
│       ├── __init__.py
│       ├── face_detector.py     # Face detection service
│       ├── face_verifier.py     # Face verification logic
│       ├── processor.py         # Video/frame processing
│       └── storage.py           # File storage management
├── face_video/                  # Directory for input videos
├── outputs/                     # Directory for output files
│   └── faces/
│       ├── matched/             # Matched face crops
│       ├── reference/           # Reference face crops
│       └── unmatched/           # Unmatched face crops
├── requirements.txt             # Python dependencies
└── run.py                       # Application entry point
```

## ⚙️ Configuration

Configuration is managed in [app/config.py](app/config.py). Key settings include:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `OUTPUT_DIR`: Directory for storing outputs
- `LOG_LEVEL`: Logging level

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns the server health status and API version.

**Response**:
```json
{
  "status": "ok",
  "message": "Face Verification API is running."
}
```

### Face Verification
```
POST /verify
```
Performs face verification across multiple video files.

**Parameters**:
- `videos` (File, required): Multiple video files. The first file is treated as the reference.

**Response**:
```json
{
  "verified": true,
  "match_count": 2,
  "message": "Same person verified across all videos",
  "details": [
    {
      "video_name": "video1.mp4",
      "is_reference": true,
      "faces_detected": 1,
      "confidence": 0.95
    },
    {
      "video_name": "video2.mp4",
      "is_reference": false,
      "faces_detected": 1,
      "confidence": 0.92,
      "match_score": 0.89
    }
  ],
  "saved_faces": {
    "reference": ["face_1.jpg"],
    "matched": ["face_1.jpg", "face_1.jpg"],
    "unmatched": []
  }
}
```

## 📚 Usage Examples

### Using cURL

```bash
# Health check
curl -X GET http://localhost:8000/health

# Face verification with two videos
curl -X POST http://localhost:8000/verify \
  -F "videos=@path/to/reference_video.mp4" \
  -F "videos=@path/to/comparison_video.mp4"
```

### Using Python Requests

```python
import requests

# Verify faces from multiple videos
with open('reference_video.mp4', 'rb') as ref_video, \
     open('comparison_video.mp4', 'rb') as comp_video:
    
    files = [
        ('videos', ref_video),
        ('videos', comp_video)
    ]
    
    response = requests.post(
        'http://localhost:8000/verify',
        files=files
    )
    
    print(response.json())
```

## 📖 API Documentation

For comprehensive API documentation with interactive examples and request/response samples, please visit:

**[Postman API Documentation](https://documenter.getpostman.com/view/53747524/2sBXqCQ4EJ)**

You can also access the interactive Swagger documentation at:
```
http://localhost:8000/docs
```

And the ReDoc documentation at:
```
http://localhost:8000/redoc
```

## 📤 Output Structure

Output files are organized in the `outputs/` directory:

```
outputs/
├── faces/
│   ├── reference/          # Face crops from the reference video
│   │   └── *.jpg
│   ├── matched/            # Face crops that matched the reference
│   │   └── *.jpg
│   └── unmatched/          # Face crops that didn't match
│       └── *.jpg
```

All face crops are accessible via the API at: `http://localhost:8000/outputs/faces/`

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Issues**:
- Reduce video resolution or frame sampling rate
- Process videos sequentially instead of in batches

**Face Detection Failures**:
- Ensure videos have good lighting and clear facial visibility
- Verify video format is compatible (MP4, AVI, MOV recommended)

**Port Already in Use**:
```bash
# Use a different port
uvicorn app.run:app --port 8001
```

**Module Import Errors**:
- Verify virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## 📝 License

This project is developed by Tezeract.

## 📞 Support

For issues, questions, or feature requests, please contact the development team or refer to the Postman documentation for API-specific queries.

---

**Version**: 1.0.0  
**Last Updated**: April 2026
