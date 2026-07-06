"""
Test Script - Personal Speech Coach
Verify installation and basic functionality of all modules.
Enhanced: covers pydub, scipy, audio conversion, and extended feature checks.
"""

import sys
import importlib
import os


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_imports() -> bool:
    print_header("TESTING PACKAGE IMPORTS")

    packages = [
        ('streamlit',  'Streamlit'),
        ('librosa',    'Librosa'),
        ('numpy',      'NumPy'),
        ('pandas',     'Pandas'),
        ('sklearn',    'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('plotly',     'Plotly'),
        ('joblib',     'Joblib'),
        ('scipy',      'SciPy'),
        ('soundfile',  'SoundFile'),
        ('pydub',      'pydub'),
    ]

    all_ok = True
    for pkg, label in packages:
        try:
            importlib.import_module(pkg)
            print(f"✅ {label:<20s} OK")
        except ImportError:
            print(f"❌ {label:<20s} NOT FOUND")
            all_ok = False

    if all_ok:
        print("\n✅ All packages installed successfully!")
    else:
        print("\n❌ Some packages missing. Run: pip install -r requirements.txt")
    return all_ok


def test_modules() -> bool:
    print_header("TESTING CUSTOM MODULES")
    ok = True

    try:
        from feature_extraction import AudioFeatureExtractor
        print("✅ feature_extraction      OK")
    except Exception as e:
        print(f"❌ feature_extraction      ERROR: {e}")
        ok = False

    try:
        from feedback_engine import FeedbackEngine
        print("✅ feedback_engine         OK")
    except Exception as e:
        print(f"❌ feedback_engine         ERROR: {e}")
        ok = False

    try:
        from model_training import SpeechModelTrainer
        print("✅ model_training          OK")
    except Exception as e:
        print(f"❌ model_training          ERROR: {e}")
        ok = False

    if ok:
        print("\n✅ All custom modules loaded successfully!")
    return ok


def test_feature_extraction() -> bool:
    print_header("TESTING FEATURE EXTRACTION")
    try:
        from feature_extraction import AudioFeatureExtractor
        extractor = AudioFeatureExtractor()
        names = extractor.get_feature_names()
        print(f"✅ Feature extractor initialised")
        print(f"✅ Feature count: {len(names)}")
        print(f"✅ First 5 features: {names[:5]}")
        assert len(names) >= 40, "Expected 40+ features"
        return True
    except Exception as e:
        print(f"❌ Feature extraction test FAILED: {e}")
        return False


def test_model_training() -> bool:
    print_header("TESTING MODEL TRAINING (small dataset)")
    try:
        from model_training import SpeechModelTrainer
        trainer = SpeechModelTrainer()
        df = trainer.create_synthetic_dataset(n_samples=60)
        print(f"✅ Dataset created: {df.shape}")
        print(f"✅ Labels: {sorted(df['fluency_label'].unique())}")
        assert set(df['fluency_label'].unique()) == {'Fluent', 'Average', 'Needs Improvement'}
        return True
    except Exception as e:
        print(f"❌ Model training test FAILED: {e}")
        return False


def test_feedback_engine() -> bool:
    print_header("TESTING FEEDBACK ENGINE")
    try:
        from feedback_engine import FeedbackEngine
        engine = FeedbackEngine()

        sample = {
            'speech_rate':   115.0,
            'silence_ratio':   0.20,
            'energy_mean':     0.065,
            'pitch_std':      35.0,
            'noise_level':     0.05,
            'duration':       15.0,
            'num_pauses':      4.0,
        }
        fb = engine.generate_feedback(sample, 'Fluent', 87.0)
        print(f"✅ Feedback generated")
        print(f"✅ Summary snippet: {fb['summary'][:60]}…")
        print(f"✅ Strengths : {len(fb['strengths'])}")
        print(f"✅ Improvements: {len(fb['improvements'])}")

        metrics = engine.get_metrics_summary(sample)
        print(f"✅ Metrics keys: {list(metrics.keys())}")
        return True
    except Exception as e:
        print(f"❌ Feedback engine test FAILED: {e}")
        return False


def test_audio_conversion() -> bool:
    """Quick smoke-test of pydub round-trip WAV → WAV."""
    print_header("TESTING AUDIO CONVERSION (pydub)")
    try:
        from pydub import AudioSegment
        import tempfile, numpy as np
        from scipy.io import wavfile

        # Create a tiny WAV in memory
        sr = 22050
        tone = (np.sin(2 * np.pi * 440 * np.arange(sr * 1) / sr) * 0.5 * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wavfile.write(f.name, sr, tone)
            src = f.name

        seg = AudioSegment.from_file(src, format='wav')
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f2:
            seg.export(f2.name, format='wav')
            dst = f2.name

        assert os.path.exists(dst) and os.path.getsize(dst) > 0
        os.unlink(src)
        os.unlink(dst)
        print("✅ pydub WAV round-trip OK")
        return True
    except Exception as e:
        print(f"❌ Audio conversion test FAILED: {e}")
        return False


def test_sample_files() -> bool:
    print_header("CHECKING SAMPLE AUDIO FILES")
    audio_dir = os.path.join("data", "audio")
    expected = ["sample_fluent.wav", "sample_average.wav", "sample_needs_improvement.wav"]
    all_found = True
    for fname in expected:
        path = os.path.join(audio_dir, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"✅ {fname:<40s} ({size_kb:.1f} KB)")
        else:
            print(f"⚠️  {fname:<40s} not found — run generate_samples.py")
            all_found = False
    return True   # non-critical


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print_header("PERSONAL SPEECH COACH — SYSTEM TEST")

    results = []
    results.append(("Package Imports",    test_imports()))

    if results[0][1]:
        results.append(("Custom Modules",    test_modules()))
        results.append(("Feature Extraction",test_feature_extraction()))
        results.append(("Model Training",    test_model_training()))
        results.append(("Feedback Engine",   test_feedback_engine()))
        results.append(("Audio Conversion",  test_audio_conversion()))
        results.append(("Sample Files",      test_sample_files()))

    print_header("TEST SUMMARY")
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:<25s} {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED — ready to run:")
        print("   streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED — fix issues before running the app.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
