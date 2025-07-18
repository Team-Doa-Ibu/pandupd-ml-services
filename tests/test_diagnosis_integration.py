from pathlib import Path

import pytest
from fastapi import status
from httpx import AsyncClient
from pytest_httpserver import HTTPServer


@pytest.mark.anyio
async def test_diagnosis_no_url(client: AsyncClient) -> None:
    """Test endpoint when no URL is provided."""
    response = await client.post("/api/diagnosis", json={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.anyio
async def test_vm_diagnosis_success(
    httpserver: HTTPServer,
    client: AsyncClient,
) -> None:
    """Test successful voice measurement diagnosis."""

    with Path("tests/assets/sample.wav").open("rb") as f:
        audio_data = f.read()
    httpserver.expect_request("/audio.wav").respond_with_data(
        audio_data,
        content_type="audio/wav",
    )
    response = await client.post(
        "/api/diagnosis",
        json={"vm_url": httpserver.url_for("/audio.wav")},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["vm_prediction"] is not None
    assert data["vm_error"] is None


@pytest.mark.anyio
async def test_hw_diagnosis_success(
    httpserver: HTTPServer,
    client: AsyncClient,
) -> None:
    """Test successful handwriting diagnosis."""

    with Path("tests/assets/sample.png").open("rb") as f:
        image_data = f.read()

    httpserver.expect_request("/image.jpg").respond_with_data(
        image_data,
        content_type="image/jpeg",
    )

    response = await client.post(
        "/api/diagnosis",
        json={"hw_url": httpserver.url_for("/image.jpg")},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["hw_prediction"] is not None
    assert data["hw_error"] is None


@pytest.mark.anyio
async def test_multi_modal_diagnosis_success(
    httpserver: HTTPServer,
    client: AsyncClient,
) -> None:
    """Test successful multi-modal diagnosis."""

    with Path("tests/assets/sample.wav").open("rb") as f:
        audio_data = f.read()
    httpserver.expect_request("/audio.wav").respond_with_data(
        audio_data,
        content_type="audio/wav",
    )

    with Path("tests/assets/sample.png").open("rb") as f:
        image_data = f.read()
    httpserver.expect_request("/image.jpg").respond_with_data(
        image_data,
        content_type="image/jpeg",
    )

    response = await client.post(
        "/api/diagnosis",
        json={
            "vm_url": httpserver.url_for("/audio.wav"),
            "hw_url": httpserver.url_for("/image.jpg"),
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["vm_prediction"] is not None
    assert data["hw_prediction"] is not None
    assert data["vm_error"] is None
    assert data["hw_error"] is None


@pytest.mark.anyio
async def test_multi_modal_diagnosis_one_fails(
    httpserver: HTTPServer,
    client: AsyncClient,
) -> None:
    """Test multi-modal diagnosis where one service fails."""

    httpserver.expect_request("/audio.wav").respond_with_data(status=404)
    with Path("tests/assets/sample.png").open("rb") as f:
        image_data = f.read()
    httpserver.expect_request("/image.jpg").respond_with_data(
        image_data,
        content_type="image/jpeg",
    )

    response = await client.post(
        "/api/diagnosis",
        json={
            "vm_url": httpserver.url_for("/audio.wav"),
            "hw_url": httpserver.url_for("/image.jpg"),
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is False
    assert data["vm_error"] is not None
    assert data["hw_prediction"] is not None
    assert data["hw_error"] is None


@pytest.mark.anyio
async def test_diagnosis_both_missing(
    httpserver: HTTPServer,
    client: AsyncClient,
) -> None:
    """Test diagnosis when both handwriting and voice predictions are missing."""

    httpserver.expect_request("/missing_audio.wav").respond_with_data(status=404)
    httpserver.expect_request("/missing_image.jpg").respond_with_data(status=404)

    response = await client.post(
        "/api/diagnosis",
        json={
            "vm_url": httpserver.url_for("/missing_audio.wav"),
            "hw_url": httpserver.url_for("/missing_image.jpg"),
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is False
    assert data["vm_prediction"] is None
    assert data["hw_prediction"] is None
    assert "No diagnosis could be made" in data["message"]
