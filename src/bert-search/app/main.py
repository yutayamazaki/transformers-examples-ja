from fastapi import FastAPI

from app.api.api_v1.urls import api_router

app = FastAPI()
app.include_router(api_router)
