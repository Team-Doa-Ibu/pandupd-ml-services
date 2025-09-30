from pathlib import Path

import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.mark.anyio
async def test_screening_image_only_success(
    client: AsyncClient,
) -> None:
    """Test successful screening with image file only."""
    with Path("tests/assets/parkinson/sample.png").open("rb") as f:
        image_data = f.read()


    files = {"hw_file": ("test.png", image_data, "image/png")}
    data = {"symptoms": "tremors and difficulty writing"}

    response = await client.post("/api/screening", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    print("\n=== Full Response ===")
    print(f"Response: {data}")
    print("====================\n")

    assert data["success"] is True
    assert data["vm_prediction"] is None
    assert data["vm_confidence"] is None
    assert data["vm_error"] is None
    assert data["hw_prediction"] is not None
    assert data["hw_confidence"] is not None
    assert data["hw_error"] is None
    assert data["message"] is not None

@pytest.mark.anyio
async def test_screening_audio_only_success(
    client: AsyncClient,
) -> None:
    """Test successful screening with audio file only."""
    with Path("tests/assets/parkinson/sample.wav").open("rb") as f:
        audio_data = f.read()

    files = {"vm_file": ("test.wav", audio_data, "audio/wav")}
    data = {"symptoms": "kesulitan artikulasi dan bicara lambat"}

    response = await client.post("/api/screening", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    print("\n=== Full Response ===")
    print(f"Response: {data}")
    print("====================\n")

    assert data["success"] is True
    assert data["vm_prediction"] is not None
    assert data["vm_confidence"] is not None
    assert data["vm_error"] is None
    assert data["hw_prediction"] is None
    assert data["hw_confidence"] is None
    assert data["hw_error"] is None
    assert data["message"] is not None

@pytest.mark.anyio
async def test_screening_multi_modal_success(
    client: AsyncClient,
) -> None:
    """Test successful screening with both audio and image files."""
    with Path("tests/assets/parkinson/sample.wav").open("rb") as f:
        audio_data = f.read()
    with Path("tests/assets/parkinson/sample.png").open("rb") as f:
        image_data = f.read()

    files = {
        "vm_file": ("test.wav", audio_data, "audio/wav"),
        "hw_file": ("test.png", image_data, "image/png"),
    }
    data = {"symptoms": "multiple symptoms including voice and motor issues"}

    response = await client.post("/api/screening", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    print("\n=== Full Response ===")
    print(f"Response: {data}")
    print("====================\n")

    assert data["success"] is True
    assert data["vm_prediction"] is not None
    assert data["vm_confidence"] is not None
    assert data["vm_error"] is None
    assert data["hw_prediction"] is not None
    assert data["hw_confidence"] is not None
    assert data["hw_error"] is None
    assert data["message"] is not None

@pytest.mark.anyio
async def test_screening_no_files(client: AsyncClient) -> None:
    """Test screening endpoint when no files are provided."""
    response = await client.post("/api/screening", files={})
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "No screening files provided" in response.json()["detail"]


@pytest.mark.anyio
async def test_screening_invalid_audio_file(client: AsyncClient) -> None:
    """Test screening endpoint with invalid audio file type."""
    with Path("tests/assets/parkinson/sample.wav").open("rb") as f:
        audio_data = f.read()

    # Create multipart form data with invalid content type
    files = {"vm_file": ("test.txt", audio_data, "text/plain")}
    data = {"symptoms": "test symptoms"}

    response = await client.post("/api/screening", files=files, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid audio file format" in response.json()["detail"]


@pytest.mark.anyio
async def test_screening_invalid_image_file(client: AsyncClient) -> None:
    """Test screening endpoint with invalid image file type."""
    with Path("tests/assets/parkinson/sample.png").open("rb") as f:
        image_data = f.read()

    # Create multipart form data with invalid content type
    files = {"hw_file": ("test.txt", image_data, "text/plain")}
    data = {"symptoms": "test symptoms"}

    response = await client.post("/api/screening", files=files, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid image file format" in response.json()["detail"]


@pytest.mark.anyio
async def test_screening_multi_modal_no_symptoms(
    client: AsyncClient,
) -> None:
    """Test successful screening with both audio and image files without symptoms."""
    with Path("tests/assets/parkinson/sample.wav").open("rb") as f:
        audio_data = f.read()
    with Path("tests/assets/parkinson/sample.png").open("rb") as f:
        image_data = f.read()

    files = {
        "vm_file": ("test.wav", audio_data, "audio/wav"),
        "hw_file": ("test.png", image_data, "image/png"),
    }
    data = {} # No symptoms

    response = await client.post("/api/screening", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    print("\n=== Full Response ===")
    print(f"Response: {data}")
    print("====================\n")

    assert data["success"] is True
    assert data["vm_prediction"] is not None
    assert data["vm_confidence"] is not None
    assert data["vm_error"] is None
    assert data["hw_prediction"] is not None
    assert data["hw_confidence"] is not None
    assert data["hw_error"] is None
    assert data["message"] is not None
