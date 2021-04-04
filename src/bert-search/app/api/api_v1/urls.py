from fastapi import APIRouter

from app.api.api_v1.endpoints import api

api_router = APIRouter()
api_router.include_router(api.router, prefix='', tags=['apis'])
