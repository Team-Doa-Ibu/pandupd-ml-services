import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from diagnosis_service.settings import settings
from diagnosis_service.utils.voice_measurements_model.mdvr_extraction import (
    process_single_file_for_prediction,
)


class VoiceMeasurementService:
    """Service for voice measurement prediction using machine learning model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = settings.vm_model_path
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> Any:

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with model_path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    def preprocess_audio(self, audio_bytes: bytes) -> Any:
        """Preprocess audio bytes for prediction."""
        import os
        import tempfile

        # Create a temporary file with the audio bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        try:
            processed_data = process_single_file_for_prediction(temp_file_path)
            if processed_data is None:
                raise ValueError("Failed to process the audio file")
            return processed_data
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def predict(
        self,
        preprocessed_data: Any,
    ) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
        """Predict voice measurement from preprocessed data."""
        try:
            probabilities = self.model.predict_proba(preprocessed_data)
            total_probabilities = probabilities.sum(axis=0)
            final_prediction = bool(total_probabilities.argmax())
            confidence = (
                round(
                    float(np.max(total_probabilities) / np.sum(total_probabilities))
                    * 100,
                )
                if np.sum(total_probabilities) > 0
                else None
            )
            return (
                final_prediction,
                f"{int(confidence)}%" if confidence is not None else None,
                None,
            )
        except Exception as e:
            return None, None, str(e)

    def predict_from_file(self, file_path: str) -> Dict[str, Any]:
        """Predict from a given audio file path."""
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            preprocessed_data = self.preprocess_audio(audio_bytes)
            prediction, confidence, error = self.predict(preprocessed_data)
            return {
                "vm_prediction": prediction,
                "vm_confidence": confidence,
                "error": error,
            }
        except Exception as e:
            return {
                "vm_prediction": None,
                "vm_confidence": None,
                "error": str(e),
            }

    def cleanup_temp_files(self, file_path: str) -> None:
        """Cleanup temporary chunk files."""
        chunk_folder = Path(file_path).with_suffix("").__str__() + "_chunks"
        chunk_folder_path = Path(chunk_folder)
        if chunk_folder_path.exists():
            for file in chunk_folder_path.iterdir():
                file.unlink()
            chunk_folder_path.rmdir()

    def predict_from_file_content(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Predict from audio file content."""
        if not audio_bytes:
            return {
                "vm_prediction": None,
                "vm_confidence": None,
                "error": "No audio data provided",
            }

        try:
            preprocessed_data = self.preprocess_audio(audio_bytes)
            prediction, confidence, error = self.predict(preprocessed_data)
            return {
                "vm_prediction": prediction,
                "vm_confidence": confidence,
                "error": error,
            }
        except Exception as e:
            logger.error(f"Voice measurement prediction error: {e}")
            return {
                "vm_prediction": None,
                "vm_confidence": None,
                "error": str(e),
            }