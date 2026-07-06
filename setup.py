#!/usr/bin/env python3
"""
Automated Setup Script for Personal Speech Coach
Installs dependencies, trains models, verifies installation, and launches the app.
Enhanced: checks ffmpeg, handles pre-trained models, colour output.
"""

import subprocess
import sys
import os
import shutil


# ── Helpers ────────────────────────────────────────────────────────────────────
def cprint(msg: str, symbol: str = "ℹ️") -> None:
    print(f"{symbol}  {msg}")


def print_header(title: str) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62 + "\n")


def run_command(cmd: str, description: str) -> bool:
    cprint(f"{description}…", "⏳")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            capture_output=True, text=True,
        )
        cprint(f"{description} — SUCCESS", "✅")
        return True
    except subprocess.CalledProcessError as e:
        cprint(f"{description} — FAILED", "❌")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()[:300]}")
        return False


def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        cprint(f"Python {major}.{minor} detected — Python 3.8+ required.", "❌")
        return False
    cprint(f"Python {major}.{minor} — OK", "✅")
    return True


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        cprint("ffmpeg found — multi-format audio conversion enabled.", "✅")
    else:
        cprint(
            "ffmpeg NOT found in PATH. Non-WAV formats (MP3, M4A, etc.) won't convert.\n"
            "   Install: https://ffmpeg.org/download.html  (Linux: sudo apt install ffmpeg)",
            "⚠️ ",
        )


def create_directories() -> None:
    for d in ["models", os.path.join("data", "audio"), "exports"]:
        os.makedirs(d, exist_ok=True)
    cprint("Project directories ready.", "✅")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print_header("🎙️  PERSONAL SPEECH COACH — SETUP")

    print("This script will:")
    print("  1. Check Python version")
    print("  2. Check for ffmpeg (multi-format audio)")
    print("  3. Create project directories")
    print("  4. Install Python dependencies")
    print("  5. Generate sample audio files")
    print("  6. Train ML models (or verify existing)")
    print("  7. Run installation tests")
    print("  8. Print launch instructions\n")

    input("Press Enter to continue…\n")

    # Step 1 — Python version
    print_header("STEP 1 · Python Version Check")
    if not check_python_version():
        sys.exit(1)

    # Step 2 — ffmpeg
    print_header("STEP 2 · ffmpeg Check")
    check_ffmpeg()

    # Step 3 — Directories
    print_header("STEP 3 · Creating Directories")
    create_directories()

    # Step 4 — Dependencies
    print_header("STEP 4 · Installing Dependencies")
    if not run_command(
        f'"{sys.executable}" -m pip install -r requirements.txt --quiet',
        "Installing required packages",
    ):
        cprint("Installation failed. Check your internet connection.", "❌")
        cprint("Manual fix: pip install -r requirements.txt", "💡")
        sys.exit(1)

    # Step 5 — Sample audio
    print_header("STEP 5 · Generating Sample Audio Files")
    audio_dir = os.path.join("data", "audio")
    sample_files = [
        "sample_fluent.wav",
        "sample_average.wav",
        "sample_needs_improvement.wav",
    ]
    all_exist = all(os.path.exists(os.path.join(audio_dir, f)) for f in sample_files)
    if all_exist:
        cprint("Sample audio files already exist — skipping.", "✅")
    else:
        run_command(f'"{sys.executable}" generate_samples.py', "Generating sample audio")

    # Step 6 — Models
    print_header("STEP 6 · ML Model Training")
    if os.path.exists(os.path.join("models", "best_model.pkl")):
        cprint("Pre-trained models found — skipping training.", "✅")
    else:
        cprint("No pre-trained models found — training now…", "ℹ️")
        if not run_command(
            f'"{sys.executable}" model_training.py',
            "Training ML models",
        ):
            cprint("Model training failed. Run model_training.py manually.", "❌")
            sys.exit(1)

    # Step 7 — Tests
    print_header("STEP 7 · Running Installation Tests")
    run_command(f'"{sys.executable}" test_installation.py', "Installation tests")

    # Step 8 — Done
    print_header("✅  SETUP COMPLETE")
    cprint("Personal Speech Coach is ready!\n", "🎉")
    print("  Launch the app:")
    print(f"    {sys.executable} -m streamlit run app.py")
    print("  — or —")
    print("    streamlit run app.py\n")
    print("  Documentation:")
    print("    README.md · QUICKSTART.md · EXAMPLES.md · INSTALL.md\n")
    print("=" * 62)


if __name__ == "__main__":
    main()
