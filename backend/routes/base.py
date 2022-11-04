from routes import route_recognition
from fastapi import APIRouter


api_router = APIRouter()

api_router.include_router(route_recognition.router, prefix="", tags=["recognition"])