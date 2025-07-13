from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class DiagnosisRequest(BaseModel):
    """Request schema for multi-modal diagnosis endpoint."""
    vm_url: Optional[HttpUrl] = Field(None, description="URL to the voice measurement audio file")
    hw_url: Optional[HttpUrl] = Field(None, description="URL to the handwriting image file")

class DiagnosisResponse(BaseModel):
    """Response schema for multi-modal diagnosis endpoint."""
    success: bool = Field(..., description="Overall status of the diagnosis request")
    vm_prediction: Optional[bool] = Field(None, description="Voice measurement prediction result")
    vm_confidence: Optional[float] = Field(None, description="Voice measurement confidence score")
    vm_error: Optional[str] = Field(None, description="Error message if voice measurement diagnosis failed")
    hw_prediction: Optional[bool] = Field(None, description="Handwriting prediction result")
    hw_confidence: Optional[float] = Field(None, description="Handwriting confidence score")
    hw_error: Optional[str] = Field(None, description="Error message if handwriting diagnosis failed")
    message: str = Field(..., description="Human-readable message summarising the results")
