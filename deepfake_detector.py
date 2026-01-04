import numpy as np
from scipy import signal
import librosa
import warnings
warnings.filterwarnings('ignore')

class DeepfakeDetector:
    """
    Deepfake voice detection using acoustic feature analysis
    """
    
    def __init__(self):
        # Pre-trained model parameters (simplified for demo)
        # In a real implementation, these would come from actual training
        self.thresholds = {
            'spectral_flatness': 0.15,
            'zero_crossing_rate': 0.08,
            'spectral_centroid_var': 1000000,
            'mfcc_var': 50,
            'pitch_var': 200,
            'energy_var': 0.01
        }
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Extract acoustic features for deepfake detection
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        features = {}
        
        # 1. Spectral Flatness (noise-like vs tonal)
        # Human speech has more tonal components, AI often sounds flatter
        spectrum = np.abs(np.fft.fft(audio_data))
        spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / np.mean(spectrum)
        features['spectral_flatness'] = spectral_flatness
        
        # 2. Zero Crossing Rate (how often signal crosses zero)
        # AI voices may have different ZCR patterns
        zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
        zcr = len(zero_crossings) / len(audio_data)
        features['zero_crossing_rate'] = zcr
        
        # 3. Spectral Centroid Variance (brightness changes)
        # Human speech has more dynamic spectral changes
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features['spectral_centroid_var'] = np.var(spectral_centroids)
        
        # 4. MFCC Variance (timbral characteristics)
        # AI voices may have less natural MFCC variations
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features['mfcc_var'] = np.mean(np.var(mfccs, axis=1))
        
        # 5. Pitch Variance (fundamental frequency changes)
        # Human speech has more natural pitch variations
        try:
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_vals = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                if pitches[index, i] > 0:
                    pitch_vals.append(pitches[index, i])
            
            if len(pitch_vals) > 0:
                features['pitch_var'] = np.var(pitch_vals)
            else:
                features['pitch_var'] = 0
        except:
            features['pitch_var'] = 0
        
        # 6. Energy Variance (amplitude changes)
        # AI voices may have more uniform energy
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hops
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0) / frame_length
        features['energy_var'] = np.var(energy)
        
        return features
    
    def detect_deepfake(self, audio_data: np.ndarray, sample_rate: int = 16000) -> tuple:
        """
        Detect if audio is deepfake with confidence score
        Returns (is_deepfake, confidence_score, feature_details)
        """
        features = self.extract_features(audio_data, sample_rate)
        
        # Calculate anomaly scores for each feature
        anomaly_scores = {}
        
        # Lower spectral flatness = more tonal (human-like)
        # Higher values suggest flatter spectrum (AI-like)
        anomaly_scores['spectral_flatness'] = min(1.0, features['spectral_flatness'] / self.thresholds['spectral_flatness'])
        
        # Zero crossing rate comparison
        anomaly_scores['zero_crossing_rate'] = abs(features['zero_crossing_rate'] - self.thresholds['zero_crossing_rate']) / self.thresholds['zero_crossing_rate']
        
        # Spectral centroid variance (human speech has more variation)
        normalized_sc_var = features['spectral_centroid_var'] / self.thresholds['spectral_centroid_var']
        anomaly_scores['spectral_centroid_var'] = max(0, 1 - normalized_sc_var)
        
        # MFCC variance (human speech has more variation)
        normalized_mfcc_var = features['mfcc_var'] / self.thresholds['mfcc_var']
        anomaly_scores['mfcc_var'] = max(0, 1 - normalized_mfcc_var)
        
        # Pitch variance (human speech has more variation)
        normalized_pitch_var = features['pitch_var'] / self.thresholds['pitch_var']
        anomaly_scores['pitch_var'] = max(0, 1 - normalized_pitch_var)
        
        # Energy variance (human speech has more variation)
        normalized_energy_var = features['energy_var'] / self.thresholds['energy_var']
        anomaly_scores['energy_var'] = max(0, 1 - normalized_energy_var)
        
        # Weighted average of anomaly scores
        # Features more indicative of deepfakes get higher weights
        weights = {
            'spectral_flatness': 0.2,
            'zero_crossing_rate': 0.15,
            'spectral_centroid_var': 0.2,
            'mfcc_var': 0.15,
            'pitch_var': 0.2,
            'energy_var': 0.1
        }
        
        weighted_score = sum(anomaly_scores[key] * weights[key] for key in weights)
        confidence_score = min(1.0, weighted_score)
        
        # Threshold for deepfake detection
        is_deepfake = confidence_score > 0.6
        
        return is_deepfake, confidence_score, {
            'features': features,
            'anomaly_scores': anomaly_scores,
            'weighted_score': weighted_score
        }

# Global instance
detector = DeepfakeDetector()

def detect_deepfake_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> tuple:
    """
    Detect if audio is deepfake with confidence score
    Returns (is_deepfake, confidence_score, details)
    """
    return detector.detect_deepfake(audio_data, sample_rate)

# Demo function
if __name__ == "__main__":
    # Test with dummy audio data
    # In real usage, this would be actual audio from the WebSocket
    dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second of random audio
    is_deepfake, confidence, details = detect_deepfake_audio(dummy_audio)
    
    print(f"Deepfake Detection Results:")
    print(f"  Is Deepfake: {is_deepfake}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Features:")
    for key, value in details['features'].items():
        print(f"    {key}: {value:.6f}")