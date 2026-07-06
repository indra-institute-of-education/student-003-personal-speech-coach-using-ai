"""
Sample Audio Generator
Creates synthetic .wav audio samples for testing the application.
Enhanced: richer speech simulation, harmonic overtones, realistic envelope.
"""

import numpy as np
from scipy.io import wavfile
import os


def _apply_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply a gentle attack-sustain-release envelope to avoid clicks."""
    fade_samples = min(int(0.05 * sample_rate), len(audio) // 10)
    envelope = np.ones(len(audio))
    envelope[:fade_samples]  = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return audio * envelope


def generate_sample_audio(
    filename: str,
    duration: int = 10,
    sample_rate: int = 22050,
    speech_type: str = 'fluent',
) -> None:
    """
    Generate a synthetic audio sample that mimics speech characteristics.

    Args:
        filename:    Output .wav path.
        duration:    Length in seconds.
        sample_rate: Sampling rate in Hz.
        speech_type: 'fluent' | 'average' | 'needs_improvement'
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    rng = np.random.default_rng(seed=42)

    if speech_type == 'fluent':
        # ── Fluent: varied pitch, good energy, low silence ──────────────
        pitch_base = 200 + 35 * np.sin(2 * np.pi * 0.5 * t) \
                        + 10 * np.sin(2 * np.pi * 1.3 * t)
        audio = 0.30 * np.sin(2 * np.pi * pitch_base * t)
        # Harmonics
        audio += 0.10 * np.sin(2 * np.pi * 2 * pitch_base * t)
        audio += 0.05 * np.sin(2 * np.pi * 3 * pitch_base * t)
        # Pauses ~15 %
        pause_mask = rng.random(len(audio)) < 0.15
        audio[pause_mask] = 0.0
        # Minimal noise
        audio += 0.003 * rng.standard_normal(len(audio))

    elif speech_type == 'average':
        # ── Average: moderate variation, moderate pauses ─────────────────
        pitch_base = 180 + 18 * np.sin(2 * np.pi * 0.3 * t)
        audio = 0.20 * np.sin(2 * np.pi * pitch_base * t)
        audio += 0.07 * np.sin(2 * np.pi * 2 * pitch_base * t)
        # Pauses ~28 %
        pause_mask = rng.random(len(audio)) < 0.28
        audio[pause_mask] = 0.0
        # Some noise
        audio += 0.008 * rng.standard_normal(len(audio))

    else:  # needs_improvement
        # ── Needs Improvement: monotone, low energy, many pauses, noisy ──
        pitch_base = 160 * np.ones_like(t)
        audio = 0.10 * np.sin(2 * np.pi * pitch_base * t)
        audio += 0.03 * np.sin(2 * np.pi * 2 * pitch_base * t)
        # Pauses ~45 %
        pause_mask = rng.random(len(audio)) < 0.45
        audio[pause_mask] = 0.0
        # Higher noise floor
        audio += 0.030 * rng.standard_normal(len(audio))

    # Apply smooth envelope
    audio = _apply_envelope(audio, sample_rate)

    # Normalise to ±0.8
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.80

    # Convert to int16 and save
    audio_int16 = np.int16(audio * 32767)
    wavfile.write(filename, sample_rate, audio_int16)
    print(f"✅ Generated: {filename}")


def main() -> None:
    print("\n" + "=" * 60)
    print("SAMPLE AUDIO GENERATOR")
    print("=" * 60 + "\n")

    audio_dir = os.path.join("data", "audio")
    os.makedirs(audio_dir, exist_ok=True)

    samples = [
        ("sample_fluent.wav",            15, 'fluent'),
        ("sample_average.wav",           18, 'average'),
        ("sample_needs_improvement.wav", 20, 'needs_improvement'),
    ]

    print("Generating sample audio files…\n")
    for fname, dur, stype in samples:
        generate_sample_audio(
            os.path.join(audio_dir, fname),
            duration=dur,
            speech_type=stype,
        )

    print("\n" + "=" * 60)
    print("✅ Sample audio files created successfully!")
    print("=" * 60)
    print(f"\nFiles saved to: {audio_dir}/")
    for fname, _, _ in samples:
        print(f"  • {fname}")
    print("\nUse these files to test the application!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
