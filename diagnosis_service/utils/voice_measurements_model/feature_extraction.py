from typing import Any

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call


class FeatureExtraction:
    """Feature extraction class for extracting features from each voice sample."""

    def __init__(self) -> None:
        self.acoustic_features: list[dict[str, float]] = []
        self.mfcc: list[np.ndarray[Any, Any]] = []

    def extract_acoustic_features(
        self,
        voice_sample: str,
        f0_min: int = 75,
        f0_max: int = 500,
        unit: str = "Hertz",
    ) -> dict[str, float] | None:
        """
        Extract acoustic features from a single .wav file.

        Extracts fundamental frequency (F0) statistics, harmonicity-to-noise ratio,
        jitter measurements (local, absolute, RAP, PPQ5), and shimmer measurements
        (local, local_dB, APQ3, APQ5) using Praat's acoustic analysis algorithms.

        Parameters:
            voice_sample: Path to the .wav file to analyze.
            f0_min: Minimum F0 value for pitch analysis in Hz.
            f0_max: Maximum F0 value for pitch analysis in Hz.
            unit: Unit for F0 measurements.

        Returns:
            Dictionary containing acoustic features or None if extraction fails.
        """
        try:
            sound = parselmouth.Sound(voice_sample)
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            f0_mean = call(pitch, "Get mean", 0, 0, unit)
            f0_std_deviation = call(pitch, "Get standard deviation", 0, 0, unit)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            point_process = call(
                sound,
                "To PointProcess (periodic, cc)",
                f0_min,
                f0_max,
            )
            jitter_relative = call(
                point_process,
                "Get jitter (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
            )
            jitter_absolute = call(
                point_process,
                "Get jitter (local, absolute)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
            )
            jitter_rap = call(
                point_process,
                "Get jitter (rap)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
            )
            jitter_ppq5 = call(
                point_process,
                "Get jitter (ppq5)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
            )
            shimmer_relative = call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            shimmer_local_db = call(
                [sound, point_process],
                "Get shimmer (local_dB)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            shimmer_apq3 = call(
                [sound, point_process],
                "Get shimmer (apq3)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            shimmer_apq5 = call(
                [sound, point_process],
                "Get shimmer (apq5)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )

            return {
                "f0_mean": f0_mean,
                "f0_std_deviation": f0_std_deviation,
                "hnr": hnr,
                "jitter_relative": jitter_relative,
                "jitter_absolute": jitter_absolute,
                "jitter_rap": jitter_rap,
                "jitter_ppq5": jitter_ppq5,
                "shimmer_relative": shimmer_relative,
                "shimmer_local_db": shimmer_local_db,
                "shimmer_apq3": shimmer_apq3,
                "shimmer_apq5": shimmer_apq5,
            }

        except Exception:
            return None

    def extract_mfcc(self, voice_sample: str) -> np.ndarray[Any, Any] | None:
        """Extract MFCC from a single .wav file."""
        try:
            sound = parselmouth.Sound(voice_sample)
            mfcc_object = sound.to_mfcc(number_of_coefficients=12)  # Extract 12 MFCCs
            mfcc = mfcc_object.to_array()
            return np.mean(mfcc.T, axis=0)

        except Exception:
            return None

    def process_single_file(self, file_path: str) -> pd.DataFrame | None:
        """Process a single .wav file and extract both acoustic and MFCC features."""
        try:
            from pathlib import Path

            if not Path(file_path).exists():
                return None

            # Extract acoustic features
            acoustic_features = self.extract_acoustic_features(file_path)

            # Extract MFCC features
            mfcc_features = self.extract_mfcc(file_path)

            # Combine into a DataFrame
            if acoustic_features and mfcc_features is not None:
                data = {
                    **acoustic_features,
                    **{f"mfcc_{i}": mfcc for i, mfcc in enumerate(mfcc_features)},
                }
                return pd.DataFrame([data])
            return None

        except Exception:
            return None
