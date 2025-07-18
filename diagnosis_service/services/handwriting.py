import io
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

from diagnosis_service.settings import settings


class HandwritingService:
    """Service for handwriting-based diagnosis using ONNX model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the handwriting service with the ONNX model."""
        if model_path is None:
            model_path = settings.hw_model_path
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, image_data: bytes) -> np.ndarray[Any, Any]:
        """
        Preprocess image bytes for ONNX model inference.

        Args:
            image_data: Image bytes or file-like object.
        Returns:
            np.ndarray: Preprocessed image tensor (NCHW, float32).
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_size = (299, 299)

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize(input_size, Image.Resampling.BILINEAR)
        image_np = np.asarray(image).astype(np.float32) / 255.0
        image_np = (image_np - mean) / std
        image_np = np.transpose(image_np, (2, 0, 1))  # HWC to CHW
        return np.expand_dims(image_np, axis=0).astype(np.float32)  # NCHW

    def _softmax(self, x: np.ndarray[Any, Any], axis: int = 1) -> np.ndarray[Any, Any]:
        """Compute softmax along the specified axis."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def predict_from_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Predict handwriting diagnosis from request data."""

        hw_url = request.get("hw_url")
        if not hw_url:
            return {
                "hw_prediction": None,
                "hw_confidence": None,
                "hw_error": "Missing hw_url",
            }

        try:
            resp = requests.get(hw_url, timeout=10)
            resp.raise_for_status()
            image_bytes = resp.content
            input_tensor = self._preprocess(image_bytes)
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor},
            )
            probs = self._softmax(outputs[0], axis=1)
            pred_class = int(np.argmax(probs, axis=1)[0])
            class_names = ["healthy", "parkinson"]
            hw_prediction = class_names[pred_class] == "parkinson"
            hw_confidence = float(probs[0, pred_class])
            return {
                "hw_prediction": hw_prediction,
                "hw_confidence": hw_confidence,
                "hw_error": None,
            }
        except Exception as e:

            # Logging can be added here if needed
            return {
                "hw_prediction": None,
                "hw_confidence": None,
                "hw_error": str(e),
            }
