from fastapi.routing import APIRouter

from diagnosis_service.web.api import echo, monitoring, diagnosis

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(diagnosis.router, tags=["diagnosis"])
