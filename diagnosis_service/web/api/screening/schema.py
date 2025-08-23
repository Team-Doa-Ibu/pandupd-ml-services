from typing import Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field


class ScreeningRequest(BaseModel):
    """Request schema for multi-modal screening endpoint."""

    vm_file: Optional[UploadFile] = Field(
        None,
        description=("Voice measurement audio file (WAV format)"),
    )
    hw_file: Optional[UploadFile] = Field(
        None,
        description=("Handwriting image file (PNG format)"),
    )


class ScreeningResponse(BaseModel):
    """Response schema for multi-modal screening."""

    success: bool = Field(
        ...,
        description=("Overall status of the screening process"),
    )
    vm_prediction: Optional[bool] = Field(
        None,
        description=("Final voice measurement prediction result"),
    )
    vm_confidence: Optional[str] = Field(
        None,
        description=("Final voice measurement confidence score in percentage"),
    )
    vm_error: Optional[str] = Field(
        None,
        description=("Error message if voice measurement prediction failed"),
    )
    hw_prediction: Optional[bool] = Field(
        None,
        description=("Final handwriting image classification result"),
    )
    hw_confidence: Optional[str] = Field(
        None,
        description=(
            "Final handwriting image classification confidence score in percentage"
        ),
    )
    hw_error: Optional[str] = Field(
        None,
        description=("Error message if handwriting image classification failed"),
    )
    message: str = Field(
        ...,
        description=("Human-readable message summarising the results"),
    )
