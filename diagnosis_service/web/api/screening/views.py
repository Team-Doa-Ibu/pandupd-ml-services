import base64

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from loguru import logger
from diagnosis_service.services.agent import create_agent
from diagnosis_service.web.api.screening.schema import ScreeningResponse

router = APIRouter()

@router.post("/screening", response_model=ScreeningResponse)
async def screen(
    vm_file: UploadFile = File(None, description="Voice measurement audio file"),
    hw_file: UploadFile = File(None, description="Handwriting image file"),
    symptoms: str = Form(None),
) -> ScreeningResponse:
    """Performs screening based on the provided files."""
    if not vm_file and not hw_file:
        raise HTTPException(status_code=400, detail="No screening files provided.")

    vm_file_content = b""
    hw_file_content = b""
    hw_file_mime_type = None

    if vm_file:
        if not vm_file.content_type or not vm_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid audio file format.")
        vm_file_content = await vm_file.read()

    if hw_file:
        if not hw_file.content_type or not hw_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file format.")
        hw_file_content = await hw_file.read()
        hw_file_mime_type = hw_file.content_type

    symptoms_content = (
        symptoms.strip()
        if symptoms and symptoms.strip()
        else "Pasien tidak memberikan informasi gejala penyakit yang relevan."
    )
    logger.info(f"Received symptoms: {symptoms_content}")
    
    agent = create_agent()

    message_content = [{"type": "text", "text": symptoms_content}]
    
    if hw_file_content:
        base64_image = base64.b64encode(hw_file_content).decode("utf-8")

        message_content.append({
            "type": "media", 
            "mime_type": hw_file_mime_type or "image/png",
            "data": base64_image,
        })

    messages = [HumanMessage(content=message_content)]

    response = await agent.ainvoke(
        {
            "messages": messages,
            "symptoms": symptoms_content,
        },
        context={
            "image_binary": hw_file_content, 
            "audio_binary": vm_file_content
        },
    )

    return response["structured_response"]