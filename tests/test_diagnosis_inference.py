from pathlib import Path

from diagnosis_service.services.handwriting import HandwritingService
from diagnosis_service.services.voice_measurement import VoiceMeasurementService


def test_handwriting_service_predict_from_request() -> None:
    """Test HandwritingService with a sample image."""
    with Path("tests/assets/sample.png").open("rb") as f:
        image_bytes = f.read()
    service = HandwritingService()
    result = service.predict_from_request({"hw_image": image_bytes})
    assert isinstance(result, dict)
    assert "hw_prediction" in result or "prediction" in result
    assert result.get("error") is None or result.get("error") == ""


def test_voice_measurement_service_predict_from_file() -> None:
    """Test VoiceMeasurementService with a sample audio file."""
    audio_path = "tests/assets/sample.wav"
    service = VoiceMeasurementService()
    result = service.predict_from_file(audio_path)
    assert isinstance(result, dict)
    assert "vm_prediction" in result
    assert result.get("error") is None or result.get("error") == ""
    # Cleanup any chunk folders created during test
    service.cleanup_temp_files(audio_path)
