from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from diagnosis_service.services.handwriting import HandwritingService
from diagnosis_service.services.voice_measurement import VoiceMeasurementService
from diagnosis_service.web.api.diagnosis.schema import DiagnosisResponse

router = APIRouter()

vm_service = VoiceMeasurementService()
hw_service = HandwritingService()


@router.post("/diagnosis", response_model=DiagnosisResponse)
async def diagnose(
    vm_file: UploadFile = File(None, description="Voice measurement audio file"),
    hw_file: UploadFile = File(None, description="Handwriting image file"),
) -> DiagnosisResponse:
    """Performs diagnosis based on the provided files."""
    if not vm_file and not hw_file:
        raise HTTPException(status_code=400, detail="No diagnosis files provided.")

    results = {}
    errors = []

    # Voice Measurement Diagnosis
    if vm_file:
        try:
            # Validate file type
            if not vm_file.content_type or not vm_file.content_type.startswith(
                "audio/",
            ):
                raise HTTPException(
                    status_code=400, detail="Invalid audio file format.",
                )

            vm_result = vm_service.predict_from_file_content(await vm_file.read())
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
    if hw_file:
        try:
            # Validate file type
            if not hw_file.content_type or not hw_file.content_type.startswith(
                "image/",
            ):
                raise HTTPException(
                    status_code=400, detail="Invalid image file format.",
                )

            hw_result = hw_service.predict_from_file_content(await hw_file.read())
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

    # Check if both predictions are missing/null
    both_missing = (
        results.get("hw_prediction") is None and results.get("vm_prediction") is None
    )

    if both_missing:
        success = False
        message = "No diagnosis could be made from the provided data."
    else:
        success = not errors
        message = "Diagnosis completed."
        if errors:
            message = "Diagnosis completed with errors: " + " ".join(errors)

    return DiagnosisResponse(success=success, message=message, **results)
