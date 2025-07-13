from fastapi import APIRouter, HTTPException
from loguru import logger

from diagnosis_service.services.handwriting import HandwritingService
from diagnosis_service.services.voice_measurement import VoiceMeasurementService
from diagnosis_service.web.api.diagnosis.schema import (
    DiagnosisRequest,
    DiagnosisResponse,
)

router = APIRouter()

vm_service = VoiceMeasurementService()
hw_service = HandwritingService()


@router.post("/diagnosis", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest) -> DiagnosisResponse:
    """Performs diagnosis based on the provided URLs."""
    if not request.vm_url and not request.hw_url:
        raise HTTPException(status_code=400, detail="No diagnosis URL provided.")

    results = {}
    errors = []

    # Voice Measurement Diagnosis
    if request.vm_url:
        try:
            vm_result = vm_service.predict_from_request(request.dict())
            results.update(
                {
                    "vm_prediction": vm_result.get("vm_prediction"),
                    "vm_confidence": vm_result.get("vm_confidence"),
                    "vm_error": vm_result.get("error"),
                },
            )
            if vm_result.get("error"):
                errors.append("Voice measurement failed.")
        except Exception as e:
            logger.error(f"Voice measurement error: {e}")
            results["vm_error"] = str(e)
            errors.append("Voice measurement failed unexpectedly.")

    # Handwriting Diagnosis
    if request.hw_url:
        try:
            hw_result = hw_service.predict_from_request(request.dict())
            results.update(
                {
                    "hw_prediction": hw_result.get("hw_prediction"),
                    "hw_confidence": hw_result.get("hw_confidence"),
                    "hw_error": hw_result.get("error"),
                },
            )
            if hw_result.get("error"):
                errors.append("Handwriting diagnosis failed.")
        except Exception as e:
            logger.error(f"Handwriting error: {e}")
            results["hw_error"] = str(e)
            errors.append("Handwriting diagnosis failed unexpectedly.")

    success = not errors
    message = "Diagnosis completed."
    if errors:
        message = "Diagnosis completed with errors: " + " ".join(errors)

    return DiagnosisResponse(success=success, message=message, **results)
