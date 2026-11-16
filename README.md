# Face Recognition

A face recognition and emotion detection application using ArcFace and Vision Transformer models.

## Features

- Face detection and extraction using OpenCV Haar Cascade
- Identity verification using ArcFace (IRetNet-50)
- Emotion detection for 7 basic emotions
- 1:1 face verification
- 1:N face identification against a database

## Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- Flask

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download model weights (if not included):
   ```
   python download_model.py
   ```

## Usage

Run the application:
```
python web_app.py
```

The web interface will be available at `http://localhost:4444`

## API Endpoints

- `GET /api/status` - System status
- `POST /api/verify` - 1:1 face verification
- `POST /api/emotion` - Emotion detection
- `POST /api/database/register` - Register face to database
- `GET /api/database/list` - List registered faces
- `POST /api/database/identify` - 1:N identification
- `POST /api/database/clear` - Clear database
- `POST /api/inspect` - Extract face embeddings

## Models

- **Identity**: ArcFace IResNet-50 (FP32)
- **Emotion**: Facial Emotion ViT (FER-2013)
- **Detector**: OpenCV Haar Cascade

## License

MIT License
