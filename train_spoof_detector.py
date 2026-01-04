import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from spoof_detector import spoof_probability as heuristic_spoof_probability

def extract_features(audio_file, sr=16000):
    """
    Extract features from audio file for spoof detection
    """
    # Load audio
    y, _ = librosa.load(audio_file, sr=sr)
    
    # Basic features
    features = []
    
    # 1. Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # 2. MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 3. Statistical features
    features.extend([
        np.mean(spectral_centroids), np.std(spectral_centroids),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
        np.mean(mfccs[1]), np.std(mfccs[1]),  # MFCC 1
        np.mean(mfccs[2]), np.std(mfccs[2]),  # MFCC 2
        np.mean(mfccs[3]), np.std(mfccs[3]),  # MFCC 3
        np.mean(mfccs[4]), np.std(mfccs[4]),  # MFCC 4
        np.sqrt(np.mean(y**2)),  # RMS
        np.max(np.abs(y)),       # Peak amplitude
    ])
    
    # 4. Heuristic features (from existing implementation)
    heuristic_score = heuristic_spoof_probability(y)
    features.append(heuristic_score)
    
    return np.array(features)

def train_spoof_detector():
    """
    Train a machine learning model for spoof detection
    """
    print("Training ML-based spoof detector...")
    
    # In a real scenario, you would have directories with genuine and spoofed audio files
    # For this demo, we'll create synthetic training data based on our heuristic
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Features for genuine speech (lower heuristic scores)
    genuine_features = []
    genuine_labels = []
    for i in range(n_samples):
        # Generate features that would be typical for genuine speech
        features = np.random.normal(0.3, 0.1, 17)  # 17 features + 1 heuristic
        features = np.clip(features, 0, 1)  # Keep in [0,1] range
        # Heuristic score for genuine speech (typically lower)
        features[-1] = np.random.beta(2, 5)  # Lower heuristic scores
        genuine_features.append(features)
        genuine_labels.append(0)  # 0 = genuine
    
    # Features for spoofed speech (higher heuristic scores)
    spoofed_features = []
    spoofed_labels = []
    for i in range(n_samples):
        # Generate features that would be typical for spoofed speech
        features = np.random.normal(0.6, 0.15, 17)  # 17 features + 1 heuristic
        features = np.clip(features, 0, 1)  # Keep in [0,1] range
        # Heuristic score for spoofed speech (typically higher)
        features[-1] = np.random.beta(5, 2)  # Higher heuristic scores
        spoofed_features.append(features)
        spoofed_labels.append(1)  # 1 = spoofed
    
    # Combine data
    X = np.vstack([genuine_features, spoofed_features])
    y = np.hstack([genuine_labels, spoofed_labels])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Genuine', 'Spoofed']))
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "spoof_detector.pkl"))
    print(f"Model saved to {os.path.join(model_dir, 'spoof_detector.pkl')}")
    
    return model

if __name__ == "__main__":
    train_spoof_detector()