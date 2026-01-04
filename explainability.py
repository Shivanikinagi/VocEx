import numpy as np
import librosa

class SpoofExplanation:
    """
    Explainability module for voice spoof detection
    """
    
    def __init__(self):
        pass
    
    def extract_acoustic_features(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Extract acoustic features from audio data for explainability
        """
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Extract features
        features = {}
        
        # 1. Pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:  # Only consider non-zero pitches
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_variation'] = features['pitch_std'] / (features['pitch_mean'] + 1e-8)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_variation'] = 0
        
        # 2. Energy and energy variation
        # Frame-wise energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        # Use librosa's built-in RMS function for better compatibility
        rms_energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        features['energy_mean'] = np.mean(rms_energy)
        features['energy_std'] = np.std(rms_energy)
        features['energy_variation'] = features['energy_std'] / (features['energy_mean'] + 1e-8)
        
        # 3. Spectral features
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 5. MFCCs (first 13)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 6. Spectral flatness (for deepfake detection)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        features['spectral_flatness'] = np.mean(spectral_flatness)
        
        return features
    
    def analyze_spoof_indicators(self, features: dict, spoof_probability: float):
        """
        Analyze acoustic features to explain why spoof was detected
        """
        explanations = []
        confidence = 0
        
        # Thresholds for normal human speech patterns (these would be learned from training data)
        normal_pitch_variation = 0.15  # Human speech has pitch variation
        normal_energy_variation = 0.3   # Human speech has energy variation
        normal_zcr_mean = 0.1          # Normal for human speech
        
        # Check pitch variation
        if 'pitch_variation' in features:
            if features['pitch_variation'] < normal_pitch_variation * 0.5:
                explanations.append({
                    "indicator": "Abnormal pitch pattern",
                    "description": "Detected abnormally low pitch variation typical of synthetic speech",
                    "confidence": min(1.0, (normal_pitch_variation * 0.5 - features['pitch_variation']) / (normal_pitch_variation * 0.5))
                })
                confidence += explanations[-1]["confidence"]
        
        # Check energy variation
        if 'energy_variation' in features:
            if features['energy_variation'] < normal_energy_variation * 0.5:
                explanations.append({
                    "indicator": "Low energy variation",
                    "description": "Detected unusually consistent energy levels typical of replayed audio",
                    "confidence": min(1.0, (normal_energy_variation * 0.5 - features['energy_variation']) / (normal_energy_variation * 0.5))
                })
                confidence += explanations[-1]["confidence"]
        
        # Check zero crossing rate
        if 'zcr_mean' in features:
            # Replay attacks often have different ZCR than natural speech
            if abs(features['zcr_mean'] - normal_zcr_mean) > normal_zcr_mean * 0.8:
                explanations.append({
                    "indicator": "Unusual zero crossing rate",
                    "description": "Detected abnormal zero crossing rate pattern indicative of audio processing",
                    "confidence": min(1.0, abs(features['zcr_mean'] - normal_zcr_mean) / (normal_zcr_mean * 0.8))
                })
                confidence += explanations[-1]["confidence"]
        
        # Check spectral features for signs of processing
        if 'spectral_centroid_mean' in features and 'spectral_bandwidth_mean' in features:
            # Processed audio often has different spectral characteristics
            if features['spectral_bandwidth_mean'] < 1000:  # Narrow bandwidth can indicate processed audio
                explanations.append({
                    "indicator": "Narrow spectral bandwidth",
                    "description": "Detected narrow frequency range typical of processed or filtered audio",
                    "confidence": min(1.0, (1000 - features['spectral_bandwidth_mean']) / 1000)
                })
                confidence += explanations[-1]["confidence"]
        
        # If we have no specific explanations but high spoof probability, provide a general explanation
        if not explanations and spoof_probability > 0.7:
            explanations.append({
                "indicator": "General anomaly detection",
                "description": "Multiple subtle acoustic anomalies detected that don't match natural human speech patterns",
                "confidence": spoof_probability
            })
            confidence = spoof_probability
        
        # Normalize confidence
        if explanations:
            confidence = min(1.0, confidence / len(explanations))
        else:
            confidence = 0.0
        
        return {
            "explanations": explanations,
            "overall_confidence": confidence,
            "spoof_probability": spoof_probability
        }
    
    def explain_deepfake_indicators(self, features: dict, deepfake_confidence: float):
        """
        Analyze acoustic features to explain why deepfake was detected
        """
        explanations = []
        confidence = 0
        
        # Deepfake-specific indicators
        # 1. Spectral flatness (AI-generated speech often has flatter spectrum)
        if 'spectral_flatness' in features:
            # High spectral flatness indicates more tonal (AI-like) characteristics
            if features['spectral_flatness'] > 0.8:
                explanations.append({
                    "indicator": "High spectral flatness",
                    "description": "Detected unusually flat frequency spectrum typical of AI-generated speech",
                    "confidence": min(1.0, (features['spectral_flatness'] - 0.8) / 0.2)
                })
                confidence += explanations[-1]["confidence"]
        
        # 2. MFCC variation (AI speech has less natural variation)
        mfcc_variation_indicators = []
        for i in range(13):
            if f'mfcc_{i}_std' in features:
                # AI-generated speech often has less MFCC variation
                if features[f'mfcc_{i}_std'] < 0.1:
                    mfcc_variation_indicators.append(features[f'mfcc_{i}_std'])
        
        if mfcc_variation_indicators:
            avg_mfcc_variation = np.mean(mfcc_variation_indicators)
            if avg_mfcc_variation < 0.05:
                explanations.append({
                    "indicator": "Low MFCC variation",
                    "description": "Detected unusually consistent mel-frequency patterns typical of synthetic speech",
                    "confidence": min(1.0, (0.05 - avg_mfcc_variation) / 0.05)
                })
                confidence += explanations[-1]["confidence"]
        
        # 3. Pitch consistency (AI speech often has unnaturally stable pitch)
        if 'pitch_std' in features and 'pitch_mean' in features:
            if features['pitch_mean'] > 0:
                pitch_cv = features['pitch_std'] / features['pitch_mean']
                if pitch_cv < 0.05:  # Very stable pitch
                    explanations.append({
                        "indicator": "Abnormally stable pitch",
                        "description": "Detected unnaturally consistent pitch patterns typical of AI-generated voices",
                        "confidence": min(1.0, (0.05 - pitch_cv) / 0.05)
                    })
                    confidence += explanations[-1]["confidence"]
        
        # If we have no specific explanations but high deepfake confidence, provide a general explanation
        if not explanations and deepfake_confidence > 0.7:
            explanations.append({
                "indicator": "AI voice signature detected",
                "description": "Multiple acoustic features match known patterns of AI-generated speech",
                "confidence": deepfake_confidence
            })
            confidence = deepfake_confidence
        
        # Normalize confidence
        if explanations:
            confidence = min(1.0, confidence / len(explanations))
        else:
            confidence = 0.0
        
        return {
            "explanations": explanations,
            "overall_confidence": confidence,
            "deepfake_confidence": deepfake_confidence
        }

# Global instance
spoof_explainer = SpoofExplanation()

def get_spoof_explanation(audio_data: np.ndarray, spoof_probability: float, sample_rate: int = 16000):
    """
    Get explanation for why spoof was detected
    """
    features = spoof_explainer.extract_acoustic_features(audio_data, sample_rate)
    return spoof_explainer.analyze_spoof_indicators(features, spoof_probability)

def get_deepfake_explanation(audio_data: np.ndarray, deepfake_confidence: float, sample_rate: int = 16000):
    """
    Get explanation for why deepfake was detected
    """
    features = spoof_explainer.extract_acoustic_features(audio_data, sample_rate)
    # Add spectral flatness for deepfake detection
    features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=audio_data))
    return spoof_explainer.explain_deepfake_indicators(features, deepfake_confidence)

# Demo function
if __name__ == "__main__":
    # Create test audio data (silence for demo)
    test_audio = np.zeros(16000)  # 1 second of silence
    
    # Test spoof explanation
    spoof_result = get_spoof_explanation(test_audio, 0.85)
    print("Spoof Explanation:")
    print(f"Overall confidence: {spoof_result['overall_confidence']:.2f}")
    for explanation in spoof_result['explanations']:
        print(f"  - {explanation['indicator']}: {explanation['description']} (confidence: {explanation['confidence']:.2f})")
    
    # Test deepfake explanation
    deepfake_result = get_deepfake_explanation(test_audio, 0.92)
    print("\nDeepfake Explanation:")
    print(f"Overall confidence: {deepfake_result['overall_confidence']:.2f}")
    for explanation in deepfake_result['explanations']:
        print(f"  - {explanation['indicator']}: {explanation['description']} (confidence: {explanation['confidence']:.2f})")