from pathlib import Path

import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence

from diagnosis_service.utils.voice_measurements_model.feature_extraction import (
    FeatureExtraction,
)

f = FeatureExtraction()


def process_single_file_for_prediction(file_path: str) -> pd.DataFrame | None:
    """
    Process a single .wav file for prediction by extracting acoustic and MFCC features.

    Parameters:
        file_path : str
            The file path to the raw .wav file that needs to be processed.

    Returns:
        pd.DataFrame : Extracted acoustic and MFCC features with renamed
            acoustic feature columns.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None

        sound_file = AudioSegment.from_wav(file_path)
        audio_chunks = split_on_silence(
            sound_file,
            min_silence_len=1000,
            silence_thresh=-40,
        )

        if len(audio_chunks) == 0:
            return None

        chunk_folder = path.with_suffix("").__str__() + "_chunks"
        chunk_folder_path = Path(chunk_folder)
        chunk_folder_path.mkdir(parents=True, exist_ok=True)

        chunk_files = []
        for i, chunk in enumerate(audio_chunks):
            out_file = chunk_folder_path / f"chunk{i}.wav"
            chunk.export(str(out_file), format="wav")
            chunk_files.append(str(out_file))

        if len(chunk_files) == 0:
            return None

        all_features = []
        for chunk_file in chunk_files:
            df = f.process_single_file(chunk_file)
            if df is not None:
                all_features.append(df)

        if len(all_features) > 0:
            df_all_features = pd.concat(all_features)

            acoustic_feature_names = [
                "meanF0Hz",
                "stdevF0Hz",
                "HNR",
                "localJitter",
                "localabsoluteJitter",
                "rapJitter",
                "ppq5Jitter",
                "localShimmer",
                "localdbShimmer",
                "apq3Shimmer",
                "apq5Shimmer",
            ]

            current_columns = df_all_features.columns.tolist()
            new_column_names = acoustic_feature_names + current_columns[11:]
            df_all_features.columns = new_column_names

            return df_all_features

        return None

    except Exception:
        return None
