"""
Feedback Engine Module
Generates personalised improvement feedback based on speech features.
Enhanced: richer rule set, communication solutions, action-item grading.
"""

import numpy as np


class FeedbackEngine:
    """Generate AI-powered personalised feedback for speech improvement."""

    # ── Thresholds ─────────────────────────────────────────────────────────────
    PACE_FAST    = 140   # BPM
    PACE_SLOW    = 80
    SILENCE_HIGH = 0.40
    SILENCE_LOW  = 0.10
    ENERGY_LOW   = 0.02
    ENERGY_GOOD  = 0.06
    PITCH_MONO   = 20    # Hz std dev
    NOISE_HIGH   = 0.10
    DUR_SHORT    = 3     # seconds
    DUR_LONG     = 60

    def __init__(self):
        self.feedback_rules = {
            'pace': {
                'too_fast': "Your speech rate is quite fast ({:.1f} BPM). Slow down and enunciate clearly — take brief pauses between key points.",
                'too_slow': "Your speech rate is slow ({:.1f} BPM). Try speaking a bit faster to maintain listener engagement.",
                'optimal':  "Your speech pace ({:.1f} BPM) is excellent — comfortable and clear.",
            },
            'pauses': {
                'too_many': "You have {:.0f} pause segments with {:.1f}% silence. Reduce filler words and unnecessary hesitations, and practise smooth transitions.",
                'too_few':  "Very few pauses detected. Remember to breathe and pause for emphasis — it helps listeners absorb information.",
                'optimal':  "Great use of pauses! Your silence ratio of {:.1f}% lets listeners process each idea effectively.",
            },
            'energy': {
                'low':     "Your voice energy is low ({:.3f}). Speak louder and with more enthusiasm to engage your audience.",
                'high':    "Excellent vocal energy ({:.3f})! You sound confident and engaging.",
                'optimal': "Perfect energy level ({:.3f}) — your voice is clear and engaging.",
            },
            'pitch': {
                'monotone': "Your pitch variation is limited (std: {:.1f} Hz). Add more vocal variety to sound dynamic and interesting.",
                'varied':   "Great pitch variation (std: {:.1f} Hz)! Your speech is expressive and engaging.",
                'optimal':  "Excellent pitch modulation — you are using vocal variety effectively.",
            },
            'noise': {
                'high':    "Background noise detected ({:.3f}). Record in a quieter environment for better clarity.",
                'low':     "Excellent recording quality with minimal noise ({:.3f})!",
                'optimal': "Perfect audio quality — clear recording environment.",
            },
            'duration': {
                'too_short': "Your speech is very brief ({:.1f}s). Provide more detail and elaboration for a thorough analysis.",
                'too_long':  "Your speech is lengthy ({:.1f}s). Try to be more concise and focus on key points.",
                'optimal':   "Good speech duration ({:.1f}s) — well-balanced length.",
            },
        }

    # ── Main entry point ───────────────────────────────────────────────────────
    def generate_feedback(self, features: dict, fluency_score: str, confidence: float) -> dict:
        """
        Generate comprehensive feedback based on extracted features.

        Args:
            features:      Dictionary of extracted audio features.
            fluency_score: Predicted fluency category string.
            confidence:    Prediction confidence percentage.

        Returns:
            dict with keys: overall_score, confidence, strengths, improvements,
                            detailed_feedback, action_items, summary, priority_focus.
        """
        fb = {
            'overall_score':    fluency_score,
            'confidence':       confidence,
            'strengths':        [],
            'improvements':     [],
            'detailed_feedback':[],
            'action_items':     [],
            'priority_focus':   [],
        }

        # ── Speech Rate ──────────────────────────────────────────────────
        sr = features.get('speech_rate', 100)
        if sr > self.PACE_FAST:
            fb['improvements'].append("Speech pace too fast")
            fb['detailed_feedback'].append(self.feedback_rules['pace']['too_fast'].format(sr))
            fb['action_items'].append("Practise speaking slower with deliberate pauses between sentences")
            fb['priority_focus'].append("pace")
        elif sr < self.PACE_SLOW:
            fb['improvements'].append("Speech pace too slow")
            fb['detailed_feedback'].append(self.feedback_rules['pace']['too_slow'].format(sr))
            fb['action_items'].append("Increase speaking speed slightly to maintain listener engagement")
            fb['priority_focus'].append("pace")
        else:
            fb['strengths'].append("Optimal speech pace")
            fb['detailed_feedback'].append(self.feedback_rules['pace']['optimal'].format(sr))

        # ── Pauses / Silence ─────────────────────────────────────────────
        silence_ratio = features.get('silence_ratio', 0.2)
        num_pauses    = features.get('num_pauses', 0)

        if silence_ratio > self.SILENCE_HIGH:
            fb['improvements'].append("Excessive pauses / hesitations")
            fb['detailed_feedback'].append(
                self.feedback_rules['pauses']['too_many'].format(num_pauses, silence_ratio * 100))
            fb['action_items'].append("Eliminate filler words (um, uh, like) — replace with silence")
            fb['priority_focus'].append("pauses")
        elif silence_ratio < self.SILENCE_LOW:
            fb['improvements'].append("Need strategic pauses")
            fb['detailed_feedback'].append(self.feedback_rules['pauses']['too_few'])
            fb['action_items'].append("Add deliberate pauses for emphasis and natural breathing")
            fb['priority_focus'].append("pauses")
        else:
            fb['strengths'].append("Good pause management")
            fb['detailed_feedback'].append(
                self.feedback_rules['pauses']['optimal'].format(silence_ratio * 100))

        # ── Energy ───────────────────────────────────────────────────────
        energy = features.get('energy_mean', 0.05)
        if energy < self.ENERGY_LOW:
            fb['improvements'].append("Low vocal energy")
            fb['detailed_feedback'].append(self.feedback_rules['energy']['low'].format(energy))
            fb['action_items'].append("Stand while speaking and project your voice to the back of the room")
            fb['priority_focus'].append("energy")
        elif energy >= self.ENERGY_GOOD:
            fb['strengths'].append("Strong vocal energy")
            fb['detailed_feedback'].append(self.feedback_rules['energy']['high'].format(energy))
        else:
            fb['strengths'].append("Adequate vocal energy")
            fb['detailed_feedback'].append(self.feedback_rules['energy']['optimal'].format(energy))

        # ── Pitch Variation ──────────────────────────────────────────────
        pitch_std = features.get('pitch_std', 25)
        if pitch_std < self.PITCH_MONO:
            fb['improvements'].append("Limited pitch variation (monotone)")
            fb['detailed_feedback'].append(
                self.feedback_rules['pitch']['monotone'].format(pitch_std))
            fb['action_items'].append("Read children's stories aloud, exaggerating character voices to widen pitch range")
            fb['priority_focus'].append("pitch")
        else:
            fb['strengths'].append("Expressive pitch variation")
            fb['detailed_feedback'].append(
                self.feedback_rules['pitch']['varied'].format(pitch_std))

        # ── Noise Level ──────────────────────────────────────────────────
        noise = features.get('noise_level', 0.05)
        if noise > self.NOISE_HIGH:
            fb['improvements'].append("Background noise")
            fb['detailed_feedback'].append(self.feedback_rules['noise']['high'].format(noise))
            fb['action_items'].append("Record in a quieter room — a closet full of clothes is a great free sound booth")
            fb['priority_focus'].append("noise")
        else:
            fb['strengths'].append("Clear audio quality")
            fb['detailed_feedback'].append(self.feedback_rules['noise']['low'].format(noise))

        # ── Duration ─────────────────────────────────────────────────────
        duration = features.get('duration', 15)
        if duration < self.DUR_SHORT:
            fb['improvements'].append("Very brief speech")
            fb['detailed_feedback'].append(
                self.feedback_rules['duration']['too_short'].format(duration))
        elif duration > self.DUR_LONG:
            fb['improvements'].append("Lengthy speech")
            fb['detailed_feedback'].append(
                self.feedback_rules['duration']['too_long'].format(duration))
        else:
            fb['strengths'].append("Appropriate duration")
            fb['detailed_feedback'].append(
                self.feedback_rules['duration']['optimal'].format(duration))

        # ── Overall summary ──────────────────────────────────────────────
        fb['summary'] = self._generate_summary(fb, fluency_score)

        return fb

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _generate_summary(self, fb: dict, fluency_score: str) -> str:
        """Build a concise overall summary string."""
        if fluency_score == "Fluent":
            summary = "🎉 Excellent work! Your speech demonstrates strong fluency and clarity. "
        elif fluency_score == "Average":
            summary = "👍 Good job! Your speech is generally clear with room for targeted improvement. "
        else:
            summary = "💪 Keep practising! Focus on the improvement areas below to enhance your fluency. "

        if fb['strengths']:
            summary += f"You excel at: {', '.join(fb['strengths'][:3])}. "

        if fb['improvements']:
            summary += f"Focus on improving: {', '.join(fb['improvements'][:3])}."

        return summary

    def get_metrics_summary(self, features: dict) -> dict:
        """Return simplified metric strings for dashboard display."""
        return {
            'Speech Rate':     f"{features.get('speech_rate', 0):.1f} BPM",
            'Silence Ratio':   f"{features.get('silence_ratio', 0) * 100:.1f}%",
            'Vocal Energy':    f"{features.get('energy_mean', 0):.3f}",
            'Pitch Variation': f"{features.get('pitch_std', 0):.1f} Hz",
            'Duration':        f"{features.get('duration', 0):.1f}s",
            'Noise Level':     f"{features.get('noise_level', 0):.3f}",
        }

    def score_to_numeric(self, fluency_label: str) -> int:
        """Convert fluency label to numeric score for charting."""
        return {"Fluent": 85, "Average": 58, "Needs Improvement": 32}.get(fluency_label, 50)
