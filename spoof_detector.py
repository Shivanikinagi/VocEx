import numpy as np
import os
try:
    import joblib
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False

def _rms(x):
    return np.sqrt(np.mean(x**2))

def _spectral_flatness(sig, n_fft=512):
    # crude spectral flatness: geometric mean / arithmetic mean of power spectrum
    try:
        fft = np.fft.rfft(sig * np.hanning(len(sig)), n=n_fft)
        ps = np.abs(fft)**2 + 1e-12
        geo_mean = np.exp(np.mean(np.log(ps)))
        arith_mean = np.mean(ps)
        return float(geo_mean / (arith_mean + 1e-12))
    except Exception:
        return 0.0

def extract_ml_features(pcm: np.ndarray, sr=16000) -> np.ndarray:
    """
    Extract features for ML-based spoof detection (17 features to match training)
    """
    # Convert to float32 if needed
    if pcm.dtype != np.float32:
        pcm = pcm.astype(np.float32)
    
    # Initialize features array with 17 elements
    features = np.zeros(17)
    
    # 1. Spectral features
    spectral_flatness_val = _spectral_flatness(pcm, n_fft=512)
    
    # 2. Energy-based features
    frame_len = 1600  # 0.1s at 16kHz
    if len(pcm) >= frame_len*2:
        rms_frames = []
        for i in range(0, len(pcm)-frame_len, frame_len):
            frame = pcm[i:i+frame_len]
            rms_frames.append(_rms(frame))
        rms_var = float(np.var(rms_frames)) if rms_frames else 0.0
    else:
        rms_var = 0.0
    
    # 3. Statistical features (17 total features to match training)
    # Fill the features array with our computed values
    features[0] = spectral_flatness_val  # Spectral flatness
    features[1] = rms_var                # RMS variance
    features[2] = float(np.sqrt(np.mean(pcm**2)))  # RMS
    features[3] = float(np.max(np.abs(pcm)))       # Peak amplitude
    
    # Additional features to reach 17 (using variations of existing features)
    features[4] = spectral_flatness_val * 2.0      # Scaled spectral flatness
    features[5] = rms_var * 1000.0                 # Scaled RMS variance
    features[6] = float(np.mean(pcm))              # Mean amplitude
    features[7] = float(np.std(pcm))               # Standard deviation
    features[8] = float(np.median(np.abs(pcm)))    # Median absolute amplitude
    features[9] = float(np.percentile(np.abs(pcm), 75))  # 75th percentile
    features[10] = float(np.percentile(np.abs(pcm), 25)) # 25th percentile
    features[11] = float(np.min(pcm))              # Minimum value
    features[12] = float(np.max(pcm))              # Maximum value
    features[13] = float(len(pcm) / sr)            # Duration in seconds
    features[14] = float(np.sum(np.abs(np.diff(pcm))))  # Sum of absolute differences
    features[15] = float(np.mean(np.abs(np.diff(pcm)))) # Mean absolute difference
    features[16] = _heuristic_spoof_probability(pcm)    # Heuristic score
    
    return features.reshape(1, -1)

def _heuristic_spoof_probability(pcm: np.ndarray) -> float:
    """
    Original heuristic-based spoof probability (kept for feature extraction)
    """
    pcm = pcm.astype(np.float32)
    # overall RMS
    rms_all = _rms(pcm)
    # frame RMS variance
    frame_len = 1600  # 0.1s at 16kHz
    if len(pcm) < frame_len*2:
        return 0.0  # too short to judge
    rms_frames = []
    for i in range(0, len(pcm)-frame_len, frame_len):
        frame = pcm[i:i+frame_len]
        rms_frames.append(_rms(frame))
    rms_var = float(np.var(rms_frames))
    flatness = _spectral_flatness(pcm, n_fft=1024)

    # normalize heuristics into [0,1]
    score = 0.0
    # If spectral flatness > 0.6 -> more likely synthetic
    score += max(0.0, (flatness - 0.25)) * 1.2
    # If variance very low, possible replay
    score += max(0.0, (0.001 - rms_var)) * 800.0
    # if overall RMS extremely low -> maybe silence / noise (treat as suspicious)
    if rms_all < 0.002:
        score += 0.6
    # clamp
    score = float(max(0.0, min(1.0, score)))
    return score

# Global variable to cache the model
_model = None

def spoof_probability(pcm: np.ndarray) -> float:
    """
    Returns a pseudo-probability [0,1] of spoof using ML model if available,
    otherwise falls back to heuristic method.
    """
    global _model
    
    # Try to use ML model if available
    if ML_MODEL_AVAILABLE:
        try:
            import joblib  # Import here to avoid issues when joblib is not available
            # Load the model (cached after first load)
            model_path = os.path.join(os.path.dirname(__file__), "models", "spoof_detector.pkl")
            if os.path.exists(model_path):
                if _model is None:
                    _model = joblib.load(model_path)
                
                # Extract features
                features = extract_ml_features(pcm)
                
                # Get probability of being spoofed (class 1)
                prob_spoof = _model.predict_proba(features)[0][1]
                return float(prob_spoof)
        except Exception as e:
            print(f"ML model prediction failed: {e}")
            # Fall back to heuristic method
    
    # Fall back to original heuristic method
    return _heuristic_spoof_probability(pcm)