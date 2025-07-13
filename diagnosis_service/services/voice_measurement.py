import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from diagnosis_service.utils.voice_measurements_model.mdvr_extraction import (
    process_single_file_for_prediction,
)


class VoiceMeasurementService:
    """Service for voice measurement prediction using machine learning model."""

    def __init__(self) -> None:
        self.model = self._load_model()
        self.scaler = MinMaxScaler()

    def _load_model(self) -> Any:

        model_path = (
            Path(__file__).parent.parent / "models" / "model_vm_mdvr-kcl_knn.bin"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with model_path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    def preprocess_audio(self, file_path: str) -> np.ndarray:
        """Preprocess audio file for prediction."""
        processed_data = process_single_file_for_prediction(file_path)
        if processed_data is None:
            raise ValueError("Failed to process the audio file")
        x_test = processed_data.values
        return self.scaler.fit_transform(x_test)

    def predict(
        self,
        preprocessed_data: np.ndarray,
    ) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
        """Predict voice measurement from preprocessed data."""
        try:
            probabilities = self.model.predict_proba(preprocessed_data)
            total_probabilities = probabilities.sum(axis=0)
            final_prediction = bool(total_probabilities.argmax())
            confidence = (
                float(np.max(total_probabilities) / np.sum(total_probabilities))
                if np.sum(total_probabilities) > 0
                else None
            )
            return final_prediction, confidence, None
        except Exception as e:
            return None, None, str(e)

    def predict_from_file(self, file_path: str) -> Dict[str, Any]:
        """Predict from a given audio file path."""
        try:
            preprocessed_data = self.preprocess_audio(file_path)
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

    def predict_from_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Predict from request containing vm_url."""
        vm_url = request.get("vm_url")
        temp_file_path = None
        result = {
            "vm_prediction": None,
            "vm_confidence": None,
            "error": "Missing vm_url",
        }
        if not vm_url:
            return result
        try:
            # NOTE: This may need to be extracted to a separate utility function
            import requests

            response = requests.get(str(vm_url), timeout=600)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            result = self.predict_from_file(temp_file_path)
            self.cleanup_temp_files(temp_file_path)
        except Exception as e:
            result["error"] = str(e)
        finally:
            if temp_file_path and Path(temp_file_path).exists():
                from contextlib import suppress

                with suppress(Exception):
                    Path(temp_file_path).unlink()
        return result
