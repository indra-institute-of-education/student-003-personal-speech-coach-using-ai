"""
Feature Extraction Module
Extracts audio features using librosa for speech analysis
Enhanced: Robust multi-format support, extended feature set
"""

import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract comprehensive audio features for speech analysis"""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def extract_features(self, audio_file_path):
        """
        Extract all speech features from audio file.

        Returns:
            tuple: (features_dict, audio_data, sample_rate) or (None, None, None) on error
        """
        try:
            # Load audio — librosa handles WAV, FLAC, OGG natively; pydub handles the rest
            y, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)

            # Guard: audio must be non-trivial
            if len(y) < sr * 0.5:
                print("Audio too short (< 0.5 s). Skipping feature extraction.")
                return None, None, None

            features = {}

            # 1. MFCC Features (13 coefficients × mean + std = 26)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
                features[f'mfcc_{i+1}_std']  = float(np.std(mfcc[i]))

            # 2. Pitch (Fundamental Frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 50:          # filter sub-50 Hz artefacts
                    pitch_values.append(p)

            features['pitch_mean'] = float(np.mean(pitch_values))  if pitch_values else 0.0
            features['pitch_std']  = float(np.std(pitch_values))   if pitch_values else 0.0
            features['pitch_max']  = float(np.max(pitch_values))   if pitch_values else 0.0
            features['pitch_min']  = float(np.min(pitch_values))   if pitch_values else 0.0

            # 3. Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std']  = float(np.std(rms))
            features['energy_max']  = float(np.max(rms))

            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std']  = float(np.std(zcr))

            # 5. Spectral Centroid
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spec_cent))
            features['spectral_centroid_std']  = float(np.std(spec_cent))

            # 6. Speech Rate (onset-based tempo)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            features['speech_rate'] = float(tempo)

            # 7. Pause / Silence
            intervals = librosa.effects.split(y, top_db=20)
            total_dur = len(y) / sr
            if len(intervals) > 0:
                speech_dur = sum(iv[1] - iv[0] for iv in intervals) / sr
                silence_dur = total_dur - speech_dur
            else:
                silence_dur = 0.0

            features['pause_duration'] = float(silence_dur)
            features['silence_ratio']  = float(silence_dur / total_dur) if total_dur > 0 else 0.0
            features['num_pauses']     = float(max(0, len(intervals) - 1))

            # 8. Spectral Rolloff
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spec_roll))
            features['spectral_rolloff_std']  = float(np.std(spec_roll))

            # 9. Spectral Bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
            features['spectral_bandwidth_std']  = float(np.std(spec_bw))

            # 10. Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std']  = float(np.std(chroma))

            # 11. Duration
            features['duration'] = float(total_dur)

            # 12. Noise Level (spectral flatness proxy)
            flatness = librosa.feature.spectral_flatness(y=y)
            features['noise_level'] = float(np.mean(flatness))

            return features, y, sr

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None, None, None

    # ------------------------------------------------------------------
    def get_feature_names(self):
        """Return ordered list of all feature names (must match training)."""
        names = []

        # MFCC
        for i in range(13):
            names.append(f'mfcc_{i+1}_mean')
            names.append(f'mfcc_{i+1}_std')

        # All other features in the same order as extract_features
        names.extend([
            'pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min',
            'energy_mean', 'energy_std', 'energy_max',
            'zcr_mean', 'zcr_std',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'speech_rate',
            'pause_duration', 'silence_ratio', 'num_pauses',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'chroma_mean', 'chroma_std',
            'duration', 'noise_level',
        ])

        return names
