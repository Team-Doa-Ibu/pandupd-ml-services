# %%
import asyncio
import base64
from dataclasses import field
from typing import Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import get_runtime
from pydantic import BaseModel, Field

from diagnosis_service.services.handwriting import HandwritingService
from diagnosis_service.services.voice_measurement import VoiceMeasurementService

from diagnosis_service.settings import settings

load_dotenv()

vm_service = VoiceMeasurementService()
hw_service = HandwritingService()


def create_agent():
    """Create a fresh agent with a new model instance to avoid event loop issues."""

    model = init_chat_model(
        "gemini-2.5-flash", 
        model_provider="google_genai", 
        temperature=0,
        api_key=settings.google_api_key,
    )

    agent = create_react_agent(
        model=model,
        tools=[
            mini_handwriting_image_classification_model,
            voice_measurement_prediction_model,
        ],
        prompt=prompt,
        response_format=ScreeningResponse,
        context_schema=InputSchema,
        state_schema=State,
    )

    return agent


class ScreeningResponse(BaseModel):
    """Response schema for multi-modal screening."""

    success: bool = Field(
        ...,
        description=(
            "Overall status of the screening process. If all avaibale model tool calls failed then this should be False. If all avaibale model tool calls calls manage to return prediction then this should be True"
        ),
    )
    vm_prediction: Optional[bool] = Field(
        None,
        description=("Final verdict of voice measurement prediction result"),
    )
    vm_confidence: Optional[str] = Field(
        None,
        description=(
            "Final verdict of voice measurement confidence score in percentage"
        ),
    )
    vm_error: Optional[str] = Field(
        None,
        description=(
            "Error message if voice measurement prediction tool call failed. DO NOT FILL WITH VALUES if audio file not provided or tool call not performed"
        ),
    )
    hw_prediction: Optional[bool] = Field(
        None,
        description=("Final verdict of handwriting image classification result"),
    )
    hw_confidence: Optional[str] = Field(
        None,
        description=(
            "Final verdict of handwriting image classification confidence score in percentage"
        ),
    )
    hw_error: Optional[str] = Field(
        None,
        description=(
            "Error message if handwriting image classification tool call failed. DO NOT FILL WITH VALUES if image file not provided or tool call not performed"
        ),
    )
    message: str = Field(
        ...,
        description=("Human-readable message summarising the results"),
    )


class InputSchema(BaseModel):
    """Defines static runtime context for the agent."""

    image_binary: bytes = Field(..., description="Raw image data in bytes")
    audio_binary: bytes = Field(..., description="Raw audio data in bytes")


class State(AgentState):
    """Defines dynamic runtime context for the agent."""

    symptoms: str
    voice_measurement_prediction_model_result: Optional[bool] = field(default=None)
    voice_measurement_prediction_model_confidence: Optional[str] = field(default=None)
    voice_measurement_prediction_model_error: Optional[str] = field(default=None)
    mini_handwriting_image_classification_model_result: Optional[bool] = field(
        default=None,
    )
    mini_handwriting_image_classification_model_confidence: Optional[str] = field(
        default=None,
    )
    mini_handwriting_image_classification_model_error: Optional[str] = field(
        default=None,
    )
    structured_response: Optional[ScreeningResponse] = field(default=None)


@tool
async def mini_handwriting_image_classification_model() -> State:
    """Analyzes handwriting image to detect Parkinson's disease symptoms (non-blocking)."""
    runtime = get_runtime(InputSchema)

    # Check if image data is available
    if not runtime.context.image_binary or len(runtime.context.image_binary) == 0:
        return {
            "mini_handwriting_image_classification_model_result": None,
            "mini_handwriting_image_classification_model_confidence": None,
            "mini_handwriting_image_classification_model_error": None,
        }

    loop = asyncio.get_running_loop()
    hw_result = await loop.run_in_executor(
        None, hw_service.predict_from_file_content, runtime.context.image_binary,
    )

    return {
        "mini_handwriting_image_classification_model_result": hw_result.get(
            "hw_prediction",
        ),
        "mini_handwriting_image_classification_model_confidence": hw_result.get(
            "hw_confidence",
        ),
        "mini_handwriting_image_classification_model_error": hw_result.get("hw_error"),
    }


@tool
async def voice_measurement_prediction_model() -> State:
    """Analyzes voice recording to detect Parkinson's disease symptoms (non-blocking)."""
    runtime = get_runtime(InputSchema)

    # Check if audio data is available
    if not runtime.context.audio_binary or len(runtime.context.audio_binary) == 0:
        return {
            "voice_measurement_prediction_model_result": None,
            "voice_measurement_prediction_model_confidence": None,
            "voice_measurement_prediction_model_error": None,
        }

    loop = asyncio.get_running_loop()
    vm_result = await loop.run_in_executor(
        None, vm_service.predict_from_file_content, runtime.context.audio_binary,
    )

    return {
        "voice_measurement_prediction_model_result": vm_result.get("vm_prediction"),
        "voice_measurement_prediction_model_confidence": vm_result.get("vm_confidence"),
        "voice_measurement_prediction_model_error": vm_result.get("vm_error"),
    }


def prompt(state: State) -> list[AnyMessage]:
    symptoms = state["symptoms"]
    runtime = get_runtime(InputSchema)

    image_file_status = (
        "File gambar tersedia. Tool mini_handwriting_image_classification_model() dapat digunakan."
        if runtime.context.image_binary and len(runtime.context.image_binary) > 0
        else "File gambar tidak tersedia. Abaikan semua analisis terkait gambar."
    )
    audio_file_status = (
        "File audio tersedia. Tool voice_measurement_prediction_model() dapat digunakan."
        if runtime.context.audio_binary and len(runtime.context.audio_binary) > 0
        else "File audio tidak tersedia. Abaikan semua analisis terkait suara."
    )

    system_msg = f"""
    Anda adalah AI Agent medis yang membantu proses **screening penyakit Parkinson**.
    Tugas Anda adalah menganalisis **gejala pasien**, **rekaman suara**, dan **gambar tulisan tangan** 
    menggunakan tools yang tersedia, lalu menyimpulkan hasil dalam format `ScreeningResponse`.

    -------------------
    ## Tools yang tersedia
    1. **mini_handwriting_image_classification_model()**
       - Input: gambar tulisan tangan
       - Output: prediksi indikasi Parkinson (True/False), confidence score, error jika ada

    2. **voice_measurement_prediction_model()**
       - Input: rekaman suara pasien
       - Output: prediksi indikasi Parkinson (True/False), confidence score, error jika ada

    -------------------
    ## Status Ketersediaan Input
    - {image_file_status}
    - {audio_file_status}

    -------------------
    ## Aturan Penting
    1. **Jangan panggil tool** jika input tidak tersedia.
    2. **Jangan sebutkan** analisis yang tidak dilakukan karena input tidak ada.
    3. Jika tool gagal atau tidak mengembalikan hasil, set `success=False` dan isi field error sesuai schema.
    4. Jika tool berhasil mengembalikan prediksi (meski hasil negatif/positif), set `success=True`.
    5. Jangan mengisi field terkait gambar/audio bila inputnya tidak ada. Biarkan `None`.

    -------------------
    ## Validasi Hasil
    - Gunakan **informasi gejala pasien** untuk memvalidasi hasil tool.
    - Contoh:
      - Jika pasien tidak melaporkan kesulitan menulis tapi hasil gambar positif → kemungkinan false positive.
      - Jika pasien melaporkan tremor, suara serak, bicara lambat → hasil audio positif lebih valid.
    - Confidence score harus dipertimbangkan:
      - Skor tinggi → hasil lebih dapat dipercaya.
      - Skor rendah → mungkin false positive / false negative.

    -------------------
    ## Analisis Mandiri (Cross-check)
    - Untuk gambar tulisan tangan: Anda boleh melakukan analisis visual mandiri.
      - Jika gambar tampak normal tapi tool hasilnya positif → kemungkinan false positive.
      - Jika gambar tampak abnormal tapi tool hasilnya negatif → kemungkinan false negative.

    -------------------
    ## Output yang Diharapkan
    Anda harus menyusun jawaban akhir dalam bentuk **structured response** mengikuti schema `ScreeningResponse`:
    - `success` → status screening
    - `vm_prediction`, `vm_confidence`, `vm_error`
    - `hw_prediction`, `hw_confidence`, `hw_error`
    - `message` → ringkasan hasil screening dalam bahasa natural untuk pasien

    -------------------
    ## Gejala Pasien
    {symptoms}
    """

    return [{"role": "system", "content": system_msg}] + state["messages"]


__all__ = ["create_agent"]


async def run_agent(image_binary: bytes, audio_binary: bytes):

    agent = create_agent()

    detail_parameter = "high"
    base64_image = base64.b64encode(image_binary).decode("utf-8")
    image_data_url = f"data:image/png;base64,{base64_image}"

    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            HumanMessage(
                content="Pasien tidak memberikan informasi gejala penyakit yang relevan.",
            ),
            HumanMessagePromptTemplate.from_template(
                [
                    {
                        "image_url": {
                            "url": image_data_url,
                            "detail": detail_parameter,
                        },
                    },
                ],
            ),
        ],
    )

    prompt = chat_prompt_template.format()

    response = await agent.ainvoke(
        {
            "messages": [prompt],
            "symptoms": "Saya mengalami kesulitan berbicara dan tremor tangan.",
        },
        context={"image_binary": image_binary, "audio_binary": audio_binary},
    )

    print(response["structured_response"])

    inputs = {
        "messages": [prompt],
        "symptoms": "Saya mengalami kesulitan berbicara dan tremor tangan.",
    }

    async for chunk in agent.astream(
        input=inputs,
        context={"image_binary": image_binary, "audio_binary": audio_binary},
        stream_mode="updates",
    ):
        for node_name, node_update in chunk.items():
            print(f"\nNode: {node_name}")
            print(f"Update: {node_update}")
            print("-" * 40)


if __name__ == "__main__":

    with open("tests/assets/parkinson/sample.png", "rb") as f:
        image_binary = f.read()

    with open("tests/assets/parkinson/sample.wav", "rb") as f:
        audio_binary = f.read()

    asyncio.run(run_agent(image_binary, audio_binary))
