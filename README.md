# VocEx - Advanced Voice Authentication System

A cutting-edge voice authentication system with real-time speaker verification and ML-powered spoof detection. 

## Features

- **Real-time Speaker Verification**: Instantly confirm user identity using SpeechBrain's ECAPA model
- **ML-Based Spoof Detection**: Advanced machine learning algorithms detect replay and synthetic voice attacks
- **Secure Embedding Storage**: Only mathematical representations stored - never raw audio for maximum privacy
- **No-Password Authentication**: Frictionless voice-based access control
- **Cross-Platform Support**: Works on Windows, macOS, and Linux
- **Web & CLI Interfaces**: Both browser-based and command-line clients available

## Technology Stack

- **Backend**: FastAPI + WebSocket + SpeechBrain + PyTorch
- **Frontend**: HTML5 + JavaScript + Web Audio API
- **ML Framework**: Scikit-learn Random Forest for spoof detection
- **Audio Processing**: NumPy + SoundDevice
- **Deployment**: Uvicorn ASGI server

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Microphone access
- Modern web browser (for web interface)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd voice-authentication-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the SpeechBrain model:
   ```bash
   python setup_model.py
   ```

4. Train the spoof detection model:
   ```bash
   python train_spoof_detector.py
   ```

### Running the System

1. Start the server:
   ```bash
   python run_server.py
   ```
   The server will start on `http://localhost:8000` with WebSocket endpoint at `ws://localhost:8000/ws`.

2. Use the system:
   - **Web Interface**: Open `index.html` in your browser
   - **CLI Interface**: Run `python client.py` in a new terminal

## How It Works

### First Run (Registration)
1. Click "Start Recording" in the web interface or run the CLI client
2. Speak naturally for 3 seconds when prompted
3. The system registers your voice and responds with "REGISTERED"

### Subsequent Runs (Authentication)
1. Click "Start Recording" again or run the CLI client
2. Speak naturally when prompted
3. The system compares your voice to the registered sample
4. Possible responses:
   - **ACCESS_GRANTED**: Your voice matches (similarity ≥ 0.80)
   - **ACCESS_DENIED**: Your voice doesn't match (similarity < 0.80)
   - **SPOOF_DETECTED**: Potential fake voice detected

## Security Features

### Speaker Verification
- Uses SpeechBrain's state-of-the-art ECAPA model
- Generates 192-dimensional voice embeddings
- Cosine similarity comparison with configurable thresholds

### Spoof Detection
- Random Forest classifier trained on 17 audio features
- Detects replay attacks, text-to-speech, and voice conversion
- Real-time analysis with heuristic fallback

### Privacy Protection
- Only stores mathematical embeddings, never raw audio
- Embeddings are irreversible - cannot reconstruct original voice
- In-memory storage for demo (can be extended to encrypted database)

## Project Structure

demo/
├── .gitignore              # Git ignore file
├── index.html              # Professional UI Web Client
├── main.py                 # FastAPI WebSocket Server (Logic)
├── client.py               # Interactive CLI Client
├── test_suite.py           # Automated Test Suite (10 samples)
├── spoof_detector.py       # ML-based Spoof Detection Logic
├── deepfake_detector.py    # Deepfake Detection Logic
├── liveness_check.py       # Liveness Challenge Logic
├── log_manager.py          # Logging System
├── alert_system.py         # Security Alert System
├── auth_utils.py           # Authentication Utilities
├── setup_model.py          # SpeechBrain Model Downloader
├── train_spoof_detector.py # Spoof Detection Model Trainer
├── run_server.py           # Server Startup Script
├── requirements.txt        # Python Dependencies
├── models/                 # Trained ML Models
└── pretrained_models/      # SpeechBrain Model Files
```

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Microphone access (optional, simulator included)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the SpeechBrain model:
   ```bash
   python setup_model.py
   ```

3. Train the spoof detection model:
   ```bash
   python train_spoof_detector.py
   ```

### Running the System

1. **Start the Server**:
   ```bash
   python run_server.py
   ```
   The server will start on `http://localhost:8001` with WebSocket endpoint at `ws://localhost:8001/ws`.

2. **Run the Client**:
   Open a new terminal and run:
   ```bash
   python client.py
   ```
   - Press **'r'** to register/authenticate (using simulated human voice).
   - Press **'m'** to authenticate using your microphone.
   - Press **'n'** (Noise) or **'t'** (Tone) to test Deepfake Detection.

3. **Run Automated Tests**:
   To verify the system with 10 different scenarios:
   ```bash
   python test_suite.py
   ```

## Ngrok Integration (Optional)

To access the system remotely:

1. Install ngrok: https://ngrok.com/download
2. Start ngrok: `ngrok http 8000`
3. Update the WebSocket URL in `index.html` or `client.py` to use the ngrok HTTPS URL with `wss://` protocol

## Troubleshooting

### Common Issues

1. **Server won't start**: Ensure port 8000 is not in use
2. **Microphone access denied**: Check browser permissions
3. **Model loading errors**: Run `setup_model.py` again
4. **Connection refused**: Ensure server is running before starting client

### Windows-Specific Notes

- SpeechBrain symlink issues are handled with torch.hub fallback
- Use dummy model if real model fails to load
- For production, consider Linux or Docker deployment

## Future Enhancements

- Multi-user support with database storage
- Enhanced ML models with more training data
- Mobile app development
- Integration with existing authentication systems
- Advanced encryption for embedding storage

## License

This project is for demonstration purposes only. 
- SpeechBrain model is licensed under CC-BY-NC 4.0
- Project code is available under MIT License

## Contact

For questions or feedback, please open an issue on the GitHub repository.
