import pickle
import tempfile
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from diagnosis_service.utils.voice_measurements_model.mdvr_extraction import process_single_file_for_prediction

class VoiceMeasurementService():
    """Service for voice measurement prediction using machine learning model."""

    def __init__(self):
        self.model = self._load_model()
        self.scaler = MinMaxScaler()

    def _load_model(self):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models",
            "model_vm_mdvr-kcl_knn.bin"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def preprocess_audio(self, file_path: str):
        processed_data = process_single_file_for_prediction(file_path)
        if processed_data is None:
            raise ValueError("Failed to process the audio file")
        X_test = processed_data.values
        X_test_scaled = self.scaler.fit_transform(X_test)
        return X_test_scaled

    def predict(self, preprocessed_data: np.ndarray) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
        try:
            probabilities = self.model.predict_proba(preprocessed_data)
            total_probabilities = probabilities.sum(axis=0)
            final_prediction = bool(total_probabilities.argmax())
            confidence = float(np.max(total_probabilities) / np.sum(total_probabilities)) if np.sum(total_probabilities) > 0 else None
            return final_prediction, confidence, None
        except Exception as e:
            return None, None, str(e)

    def predict_from_file(self, file_path: str) -> Dict[str, Any]:
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
        try:
            chunk_folder = os.path.splitext(file_path)[0] + "_chunks"
            if os.path.exists(chunk_folder):
                for file in os.listdir(chunk_folder):
                    os.remove(os.path.join(chunk_folder, file))
                os.rmdir(chunk_folder)
        except Exception:
            pass

    def predict_from_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
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
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass
        return result
