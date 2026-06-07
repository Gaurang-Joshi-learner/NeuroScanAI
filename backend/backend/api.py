"""
api.py — NeuroScanAI FastAPI entry point
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.database import create_tables, settings
from auth.router import router as auth_router
from routers.analysis import router as analysis_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    await create_tables()
    yield

app = FastAPI(
    title="NeuroScanAI API",
    version="1.0.0",
    description="EEG Seizure Detection & Speech Decoding Platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(analysis_router)

@app.get("/")
def root():
    return {"service": "NeuroScanAI", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}
