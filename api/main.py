from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.routes import router
from config import get_settings

settings = get_settings()

app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root health check for Railway
# Root health check for Railway
@app.get("/")
def health_root():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Include router
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
