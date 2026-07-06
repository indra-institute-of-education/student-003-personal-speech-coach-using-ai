"""
Model Training Module
Trains and compares multiple ML models for fluency classification.
Enhanced: cleaner synthetic data generation, reproducible splits, richer reporting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class SpeechModelTrainer:
    """Train and evaluate multiple ML models for speech fluency classification."""

    LABELS = ['Fluent', 'Average', 'Needs Improvement']

    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42),
            'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
        }
        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model      = None
        self.best_model_name = None
        self.results         = {}

    # ── Synthetic dataset ──────────────────────────────────────────────────────
    def create_synthetic_dataset(self, n_samples: int = 300) -> pd.DataFrame:
        """
        Create a balanced synthetic dataset for demonstration.
        Replace with real labelled data for production use.
        """
        np.random.seed(42)
        n_per_class = n_samples // 3
        data, labels = [], []

        # ── Fluent speakers ──────────────────────────────────────────────
        for _ in range(n_per_class):
            row = {
                **{f'mfcc_{i}_mean': np.random.normal(0,   1.0) for i in range(1, 14)},
                **{f'mfcc_{i}_std':  np.random.normal(1,   0.3) for i in range(1, 14)},
                'pitch_mean':              np.random.normal(180,  30),
                'pitch_std':               np.random.normal(40,   10),
                'pitch_max':               np.random.normal(250,  40),
                'pitch_min':               np.random.normal(120,  20),
                'energy_mean':             np.random.normal(0.08, 0.02),
                'energy_std':              np.random.normal(0.03, 0.01),
                'energy_max':              np.random.normal(0.15, 0.03),
                'zcr_mean':                np.random.normal(0.08, 0.02),
                'zcr_std':                 np.random.normal(0.03, 0.01),
                'spectral_centroid_mean':  np.random.normal(2000, 300),
                'spectral_centroid_std':   np.random.normal(500,  100),
                'speech_rate':             np.random.normal(110,  15),
                'pause_duration':          np.random.normal(2,    0.5),
                'silence_ratio':           np.random.normal(0.15, 0.05),
                'num_pauses':              float(np.random.randint(3, 8)),
                'spectral_rolloff_mean':   np.random.normal(3500, 500),
                'spectral_rolloff_std':    np.random.normal(800,  200),
                'spectral_bandwidth_mean': np.random.normal(2500, 400),
                'spectral_bandwidth_std':  np.random.normal(600,  150),
                'chroma_mean':             np.random.normal(0.40, 0.10),
                'chroma_std':              np.random.normal(0.15, 0.05),
                'duration':                np.random.normal(15,   5),
                'noise_level':             np.random.normal(0.05, 0.02),
            }
            data.append(row)
            labels.append('Fluent')

        # ── Average speakers ─────────────────────────────────────────────
        for _ in range(n_per_class):
            row = {
                **{f'mfcc_{i}_mean': np.random.normal(0,    1.2) for i in range(1, 14)},
                **{f'mfcc_{i}_std':  np.random.normal(1,    0.4) for i in range(1, 14)},
                'pitch_mean':              np.random.normal(170,  35),
                'pitch_std':               np.random.normal(25,   8),
                'pitch_max':               np.random.normal(220,  50),
                'pitch_min':               np.random.normal(110,  25),
                'energy_mean':             np.random.normal(0.05, 0.02),
                'energy_std':              np.random.normal(0.025,0.01),
                'energy_max':              np.random.normal(0.10, 0.03),
                'zcr_mean':                np.random.normal(0.10, 0.03),
                'zcr_std':                 np.random.normal(0.04, 0.01),
                'spectral_centroid_mean':  np.random.normal(1800, 400),
                'spectral_centroid_std':   np.random.normal(450,  120),
                'speech_rate':             np.random.normal(95,   20),
                'pause_duration':          np.random.normal(4,    1.0),
                'silence_ratio':           np.random.normal(0.28, 0.08),
                'num_pauses':              float(np.random.randint(5, 12)),
                'spectral_rolloff_mean':   np.random.normal(3200, 600),
                'spectral_rolloff_std':    np.random.normal(700,  250),
                'spectral_bandwidth_mean': np.random.normal(2200, 500),
                'spectral_bandwidth_std':  np.random.normal(550,  180),
                'chroma_mean':             np.random.normal(0.35, 0.12),
                'chroma_std':              np.random.normal(0.18, 0.06),
                'duration':                np.random.normal(18,   7),
                'noise_level':             np.random.normal(0.08, 0.03),
            }
            data.append(row)
            labels.append('Average')

        # ── Needs Improvement ────────────────────────────────────────────
        for _ in range(n_per_class):
            # Randomly either too fast or too slow
            rate = np.random.choice([
                np.random.normal(150, 20),   # too fast
                np.random.normal(68,  15),   # too slow
            ])
            row = {
                **{f'mfcc_{i}_mean': np.random.normal(0,    1.5) for i in range(1, 14)},
                **{f'mfcc_{i}_std':  np.random.normal(1,    0.5) for i in range(1, 14)},
                'pitch_mean':              np.random.normal(160,  40),
                'pitch_std':               np.random.normal(15,   5),   # monotone
                'pitch_max':               np.random.normal(200,  60),
                'pitch_min':               np.random.normal(100,  30),
                'energy_mean':             np.random.normal(0.03, 0.015),  # low energy
                'energy_std':              np.random.normal(0.02, 0.008),
                'energy_max':              np.random.normal(0.07, 0.02),
                'zcr_mean':                np.random.normal(0.12, 0.04),
                'zcr_std':                 np.random.normal(0.05, 0.015),
                'spectral_centroid_mean':  np.random.normal(1600, 500),
                'spectral_centroid_std':   np.random.normal(400,  150),
                'speech_rate':             float(rate),
                'pause_duration':          np.random.normal(6,    2.0),   # many pauses
                'silence_ratio':           np.random.normal(0.45, 0.10),
                'num_pauses':              float(np.random.randint(10, 20)),
                'spectral_rolloff_mean':   np.random.normal(3000, 700),
                'spectral_rolloff_std':    np.random.normal(650,  280),
                'spectral_bandwidth_mean': np.random.normal(2000, 600),
                'spectral_bandwidth_std':  np.random.normal(500,  200),
                'chroma_mean':             np.random.normal(0.30, 0.15),
                'chroma_std':              np.random.normal(0.20, 0.08),
                'duration':                np.random.normal(20,   10),
                'noise_level':             np.random.normal(0.12, 0.04),  # noisy
            }
            data.append(row)
            labels.append('Needs Improvement')

        df = pd.DataFrame(data)
        df['fluency_label'] = labels
        return df

    # ── Train all models ───────────────────────────────────────────────────────
    def train_models(self, df: pd.DataFrame) -> dict:
        """
        Train all models, evaluate them, and store the best one.

        Args:
            df: DataFrame with features + 'fluency_label' column.

        Returns:
            dict: {model_name: {accuracy, precision, recall, f1_score,
                                confusion_matrix, cv_mean, cv_std, model}}
        """
        X = df.drop('fluency_label', axis=1).values
        y = df['fluency_label'].values

        y_enc = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        best_acc = -1.0

        for name, model in self.models.items():
            print(f"Training {name}…")
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm   = confusion_matrix(y_test, y_pred)

            # 5-fold cross-validation score
            cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')

            self.results[name] = {
                'accuracy':         acc,
                'precision':        prec,
                'recall':           rec,
                'f1_score':         f1,
                'confusion_matrix': cm,
                'cv_mean':          float(cv_scores.mean()),
                'cv_std':           float(cv_scores.std()),
                'model':            model,
            }

            if acc > best_acc:
                best_acc          = acc
                self.best_model      = model
                self.best_model_name = name

        print(f"\nBest model: {self.best_model_name}  (accuracy={best_acc:.4f})")
        return self.results

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_models(self, models_dir: str = 'models'):
        """Persist best model, scaler, label encoder, and full results."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.best_model,      os.path.join(models_dir, 'best_model.pkl'))
        joblib.dump(self.scaler,          os.path.join(models_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder,   os.path.join(models_dir, 'label_encoder.pkl'))
        joblib.dump(self.results,         os.path.join(models_dir, 'model_results.pkl'))
        print(f"Models saved → {models_dir}/")

    def load_models(self, models_dir: str = 'models') -> bool:
        """Load saved models and preprocessing objects."""
        try:
            self.best_model    = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
            self.scaler        = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
            self.results       = joblib.load(os.path.join(models_dir, 'model_results.pkl'))
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = SpeechModelTrainer()

    print("Generating synthetic dataset (300 samples)…")
    df = trainer.create_synthetic_dataset(n_samples=300)
    print(f"Dataset shape: {df.shape}")
    print(df['fluency_label'].value_counts().to_string())

    print("\nTraining models…")
    results = trainer.train_models(df)

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)
    for name, m in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:  {m['f1_score']:.4f}")
        print(f"  CV Score:  {m['cv_mean']:.4f} ± {m['cv_std']:.4f}")

    trainer.save_models('models')
    print("\nTraining complete — models saved to models/")
