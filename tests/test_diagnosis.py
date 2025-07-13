import pytest
from fastapi import status
from httpx import AsyncClient
from pytest_httpserver import HTTPServer

from diagnosis_service.web.application import get_app


@pytest.mark.anyio
async def test_diagnosis_no_url() -> None:
    """Test endpoint when no URL is provided."""
    app = get_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/diagnosis", json={})
        assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.anyio
async def test_vm_diagnosis_invalid_url() -> None:
    """Test voice measurement with an invalid URL."""
    app = get_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/diagnosis", json={"vm_url": "invalid-url"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.anyio
async def test_vm_diagnosis_failed_download(httpserver: HTTPServer) -> None:
    """Test voice measurement with a URL that fails to download."""
    app = get_app()
    httpserver.expect_request("/audio.wav").respond_with_data(status=404)
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/diagnosis", json={"vm_url": httpserver.url_for("/audio.wav")},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is False
        assert data["vm_error"] is not None


@pytest.mark.anyio
async def test_vm_diagnosis_success(httpserver: HTTPServer) -> None:
    """Test successful voice measurement diagnosis."""
    app = get_app()
    from pathlib import Path
    with Path("tests/assets/sample.wav").open("rb") as f:
        audio_data = f.read()
    httpserver.expect_request("/audio.wav").respond_with_data(
        audio_data, content_type="audio/wav",
    )
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/diagnosis", json={"vm_url": httpserver.url_for("/audio.wav")},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["vm_prediction"] is not None
        assert data["vm_error"] is None


@pytest.mark.anyio
async def test_hw_diagnosis_success(httpserver: HTTPServer) -> None:
    """Test successful handwriting diagnosis."""
    app = get_app()
    httpserver.expect_request("/image.jpg").respond_with_data(b"fake-image-data")
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/diagnosis", json={"hw_url": httpserver.url_for("/image.jpg")},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["hw_prediction"] is not None
        assert data["hw_error"] is None


@pytest.mark.anyio
async def test_multi_modal_diagnosis_success(httpserver: HTTPServer) -> None:
    """Test successful multi-modal diagnosis."""
    app = get_app()
    from pathlib import Path
    with Path("tests/assets/sample.wav").open("rb") as f:
        audio_data = f.read()
    httpserver.expect_request("/audio.wav").respond_with_data(
        audio_data, content_type="audio/wav",
    )
    httpserver.expect_request("/image.jpg").respond_with_data(b"fake-image-data")

    async with AsyncClient(app=app, base_url="http://test") as client:
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
async def test_multi_modal_diagnosis_one_fails(httpserver: HTTPServer) -> None:
    """Test multi-modal diagnosis where one service fails."""
    app = get_app()
    httpserver.expect_request("/audio.wav").respond_with_data(status=404)
    httpserver.expect_request("/image.jpg").respond_with_data(b"fake-image-data")

    async with AsyncClient(app=app, base_url="http://test") as client:
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
