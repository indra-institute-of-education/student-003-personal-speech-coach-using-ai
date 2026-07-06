# Dataset Structure and Information

## Synthetic Dataset Generation

The application generates synthetic data for demonstration purposes. In production, this would be replaced with real labeled speech data.

### Dataset Composition

**Total Samples**: 300 (configurable)
**Distribution**:
- Fluent: 100 samples (33.3%)
- Average: 100 samples (33.3%)
- Needs Improvement: 100 samples (33.3%)

### Feature Set (40+ features)

#### 1. MFCC Features (26 features)
- mfcc_1_mean through mfcc_13_mean
- mfcc_1_std through mfcc_13_std
- Captures voice timbre and characteristics

#### 2. Pitch Features (4 features)
- pitch_mean: Average fundamental frequency
- pitch_std: Pitch variation (expressiveness)
- pitch_max: Maximum pitch
- pitch_min: Minimum pitch

#### 3. Energy Features (3 features)
- energy_mean: Average volume/projection
- energy_std: Energy variation
- energy_max: Maximum energy

#### 4. Zero Crossing Rate (2 features)
- zcr_mean: Voice quality indicator
- zcr_std: Variation in voice quality

#### 5. Spectral Features (6 features)
- spectral_centroid_mean/std: Brightness of sound
- spectral_rolloff_mean/std: High-frequency content
- spectral_bandwidth_mean/std: Frequency range

#### 6. Speech Temporal Features (4 features)
- speech_rate: Tempo in BPM
- pause_duration: Total silence time
- silence_ratio: Percentage of silence
- num_pauses: Count of pauses

#### 7. Chroma Features (2 features)
- chroma_mean: Pitch class profile
- chroma_std: Variation in pitch classes

#### 8. Other Features (2 features)
- duration: Total speech length
- noise_level: Background noise estimate

### Fluency Category Characteristics

#### Fluent Speakers
- Speech Rate: ~110 BPM (moderate, clear)
- Silence Ratio: ~15% (appropriate pauses)
- Energy Mean: ~0.08 (good projection)
- Pitch Std: ~40 Hz (expressive)
- Noise Level: ~0.05 (clear audio)

#### Average Speakers
- Speech Rate: ~95 BPM (slightly slower)
- Silence Ratio: ~28% (more pauses)
- Energy Mean: ~0.05 (moderate volume)
- Pitch Std: ~25 Hz (less expressive)
- Noise Level: ~0.08 (some noise)

#### Needs Improvement
- Speech Rate: ~70 or ~150 BPM (too slow/fast)
- Silence Ratio: ~45% (excessive pauses)
- Energy Mean: ~0.03 (low volume)
- Pitch Std: ~15 Hz (monotone)
- Noise Level: ~0.12 (noisy environment)

## Using Real Data

To use your own labeled data:

1. **Prepare Audio Files**
   - Collect .wav files of speeches
   - Label each as: Fluent, Average, or Needs Improvement
   - Organize in folders by category

2. **Extract Features**
   ```python
   from feature_extraction import AudioFeatureExtractor
   
   extractor = AudioFeatureExtractor()
   features, _, _ = extractor.extract_features('path/to/audio.wav')
   ```

3. **Create DataFrame**
   ```python
   import pandas as pd
   
   data = []
   for audio_file, label in labeled_data:
       features, _, _ = extractor.extract_features(audio_file)
       features['fluency_label'] = label
       data.append(features)
   
   df = pd.DataFrame(data)
   ```

4. **Train Models**
   ```python
   from model_training import SpeechModelTrainer
   
   trainer = SpeechModelTrainer()
   results = trainer.train_models(df)
   trainer.save_models()
   ```

## Data Validation

### Audio File Requirements
- **Format**: WAV (PCM)
- **Sample Rate**: 22050 Hz (recommended)
- **Bit Depth**: 16-bit or 24-bit
- **Channels**: Mono or Stereo
- **Duration**: 5-60 seconds
- **Quality**: Clear speech, minimal noise

### Feature Value Ranges

| Feature | Typical Range | Unit |
|---------|---------------|------|
| Speech Rate | 60-160 | BPM |
| Silence Ratio | 0.05-0.50 | Ratio |
| Energy Mean | 0.01-0.15 | Amplitude |
| Pitch Mean | 80-300 | Hz |
| Pitch Std | 10-80 | Hz |
| Noise Level | 0.01-0.20 | Ratio |

## Model Performance

Expected performance on synthetic data:
- **Logistic Regression**: 85-90% accuracy
- **Random Forest**: 90-95% accuracy
- **SVM**: 88-93% accuracy

Real-world performance depends on:
- Data quality
- Dataset size
- Feature relevance
- Label accuracy

## Data Privacy

- All processing is local
- No data sent to external servers
- Audio files stored only in local data/ directory
- History saved in user_history.csv (local)
- Models saved locally in models/ directory

## Future Enhancements

Potential data improvements:
- [ ] Collect real speech corpus
- [ ] Add more fluency categories
- [ ] Include age/gender demographics
- [ ] Support multiple languages
- [ ] Add emotion labels
- [ ] Include speaking context (presentation, conversation, etc.)
- [ ] Expand feature set with prosody features
- [ ] Add speaker identification
