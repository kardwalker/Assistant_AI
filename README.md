# 🎬 Assistant AI - Video Separator & Analysis Tool

A powerful Python-based video processing tool that separates video files into audio and video components, with additional capabilities for video analysis and chat-based interactions.

## 🚀 Features

### 📹 Video Separation
- **Extract Audio**: Convert video files to MP3 audio format
- **Extract Video**: Create video files without audio tracks
- **Batch Processing**: Process multiple videos at once
- **Smart Naming**: Automatic file naming with descriptive suffixes

### 🧠 Video Analysis
- Event detection and timestamping
- Guideline compliance checking
- Confidence scoring for detected events
- Comprehensive video summaries

### 💬 Chat Interface
- Interactive chat system for video discussions
- Context-aware conversations
- Session management
- Message history tracking

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/kardwalker/Assistant_AI.git
cd Assistant_AI
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Video Separation

```python
from Project.separator import process_video, extract_audio_only, extract_video_only

# Extract both audio and video separately
result = process_video("input_video.mp4")
if result['success']:
    print(f"Audio: {result['audio']}")
    print(f"Video: {result['video']}")

# Extract only audio
audio_file = extract_audio_only("input_video.mp4")

# Extract only video (without audio)
video_file = extract_video_only("input_video.mp4")
```

### Video Analysis

```python
from Project.schema import VideoAnalysis, VideoEvent

# Create video analysis
analysis = VideoAnalysis(
    video_id="unique_id",
    duration=120.5,
    events=[],
    summary="Video analysis summary",
    guideline_violations=[],
    processed_at=datetime.now()
)
```

## 📁 Project Structure

```
Assistant_AI/
├── Project/
│   ├── separator.py      # Video separation functionality
│   └── schema.py         # Data models and schemas
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## 🎯 Supported Formats

### Input Video Formats
- MP4, AVI, MOV, MKV, WMV
- Most common video codecs

### Output Formats
- **Audio**: MP3 (high quality)
- **Video**: MP4 (H.264 codec)

## 🔧 Dependencies

- **moviepy**: Video processing and manipulation
- **pydantic**: Data validation and serialization
- **typing**: Type hints support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MoviePy team for excellent video processing capabilities
- Pydantic team for robust data validation
- OpenAI for inspiration and AI capabilities

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**Made with ❤️ for the Vuecode AI Hackathon**
