from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from diagnosis_service.utils.voice_measurements_model.feature_extraction import Feature_Extraction
import pandas as pd

# Initialize the feature extraction object
f = Feature_Extraction()

def process_single_file_for_prediction(file_path):
    """
    Splits a single .wav file into chunks and extracts acoustic and MFCC features for prediction.
    
    Parameters:
    file_path : str
        The file path to the raw .wav file that needs to be processed.
        
    Returns:
    pd.DataFrame : Extracted acoustic and MFCC features with renamed acoustic feature columns.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None

        # Load the .wav file
        sound_file = AudioSegment.from_wav(file_path)
        
        # Split the file into chunks
        audio_chunks = split_on_silence(sound_file, min_silence_len=1000, silence_thresh=-40)

        if len(audio_chunks) == 0:
            print("No chunks detected, unable to process the file.")
            return None

        # Create a temporary folder for chunks
        chunk_folder = os.path.splitext(file_path)[0] + "_chunks"
        os.makedirs(chunk_folder, exist_ok=True)
        
        # Export each chunk
        chunk_files = []
        for i, chunk in enumerate(audio_chunks):
            out_file = os.path.join(chunk_folder, f"chunk{i}.wav")
            chunk.export(out_file, format="wav")
            chunk_files.append(out_file)

        if len(chunk_files) == 0:
            print("No valid chunks generated.")
            return None

        # Process the chunks using the updated Feature_Extraction methods
        all_features = []
        for chunk_file in chunk_files:
            df = f.process_single_file(chunk_file)  # Extract features for each chunk
            if df is not None:
                all_features.append(df)

        # Combine features from all chunks
        if len(all_features) > 0:
            df_all_features = pd.concat(all_features)
            
            # Rename only the acoustic feature columns
            acoustic_feature_names = [
                'meanF0Hz',
                'stdevF0Hz',
                'HNR',
                'localJitter',
                'localabsoluteJitter',
                'rapJitter',
                'ppq5Jitter',
                'localShimmer',
                'localdbShimmer',
                'apq3Shimmer',
                'apq5Shimmer'
            ]

            # Get current column names
            current_columns = df_all_features.columns.tolist()

            # Map the acoustic feature names 
            new_column_names = acoustic_feature_names + current_columns[11:]

            # Apply the new column names
            df_all_features.columns = new_column_names

            return df_all_features
        else:
            print("No features extracted from chunks.")
            return None

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
