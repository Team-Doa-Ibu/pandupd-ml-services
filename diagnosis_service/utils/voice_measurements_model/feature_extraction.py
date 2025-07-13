import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import os

class Feature_Extraction:
    """
    Feature extraction class containing the methods to extract features for each voice sample.
    """

    def __init__(self):
        self.acoustic_features = []
        self.mfcc = []

    def extract_acoustic_features(self, voice_sample, f0_min=75, f0_max=500, unit="Hertz"):
        """
        Extract acoustic features like jitters, shimmers, and HNR from a single .wav file.

        Parameters:
        voice_sample : .wav file path
            Path to the voice sample we want to extract the features from
        f0_min: int
            Minimum fundamental frequency (default 75 Hz)
        f0_max: int
            Maximum fundamental frequency (default 500 Hz)
        """
        try:
            sound = parselmouth.Sound(voice_sample)
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            f0_mean = call(pitch, "Get mean", 0, 0, unit)
            f0_std_deviation = call(pitch, "Get standard deviation", 0, 0, unit)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
            jitter_relative = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_absolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_relative = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_localDb = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            return {
                "f0_mean": f0_mean,
                "f0_std_deviation": f0_std_deviation,
                "hnr": hnr,
                "jitter_relative": jitter_relative,
                "jitter_absolute": jitter_absolute,
                "jitter_rap": jitter_rap,
                "jitter_ppq5": jitter_ppq5,
                "shimmer_relative": shimmer_relative,
                "shimmer_localDb": shimmer_localDb,
                "shimmer_apq3": shimmer_apq3,
                "shimmer_apq5": shimmer_apq5
            }

        except Exception as e:
            print(f"Error processing acoustic features for {voice_sample}: {e}")
            return None

    def extract_mfcc(self, voice_sample):
        """
        Extract MFCC from a single .wav file.

        Parameters:
        voice_sample : .wav file path
            Path to the voice sample we want to extract the features from.
        """
        try:
            sound = parselmouth.Sound(voice_sample)
            mfcc_object = sound.to_mfcc(number_of_coefficients=12)  # Extract 12 MFCCs
            mfcc = mfcc_object.to_array()
            mfcc_mean = np.mean(mfcc.T, axis=0)
            return mfcc_mean

        except Exception as e:
            print(f"Error processing MFCC for {voice_sample}: {e}")
            return None

    def process_single_file(self, file_path):
        """
        Process a single .wav file and extract both acoustic and MFCC features.

        Parameters:
        file_path : str
            Path to the .wav file to be processed.

        Returns:
        pd.DataFrame : DataFrame containing the extracted acoustic and MFCC features.
        """
        try:
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                return None

            # Extract acoustic features
            acoustic_features = self.extract_acoustic_features(file_path)

            # Extract MFCC features
            mfcc_features = self.extract_mfcc(file_path)

            # Combine into a DataFrame
            if acoustic_features and mfcc_features is not None:
                data = {**acoustic_features, **{f"mfcc_{i}": mfcc for i, mfcc in enumerate(mfcc_features)}}
                df = pd.DataFrame([data])
                return df
            else:
                print("Failed to extract features.")
                return None

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
