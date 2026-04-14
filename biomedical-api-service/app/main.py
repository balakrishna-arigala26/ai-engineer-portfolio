import logging
from typing import List, Annotated
from fastapi import FastAPI, Depends, HTTPException, UploadFile as UF, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  
from pydantic import WithJsonSchema

# --- LOAD ENV VARS BEFORE IMPORTING YOUR AI ENGINE ---
from dotenv import load_dotenv
load_dotenv()
# -----------------------------------------------------

# Local application imports
from app.models import ChatRequest, ChatResponse
from app.engine import BiomedicalAIEngine
from app.utils import process_pdf 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Forces Swagger to recognize the upload as a file picker instead of an array<string>
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]
# ------------------------------

app = FastAPI(title="Enterprise Biomedical AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_ai_engine():
    return BiomedicalAIEngine()

# --- ROUTES ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "biomedical-ai"}

@app.post("/ask", response_model=ChatResponse)
async def ask_biomedical_ai(
    request: ChatRequest, 
    engine: BiomedicalAIEngine = Depends(get_ai_engine)
):
    try:
        logger.info(f"Processing question for session: {request.session_id}")
        answer = await engine.ask_question(request.question, request.session_id)
        return ChatResponse(answer=answer, session_id=request.session_id)
    except Exception as e:
        logger.error(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal AI Engine Error")

# --- NEW STREAMING ROUTE ADDED HERE ---
@app.post("/ask-stream")
async def ask_biomedical_ai_stream(
    request: ChatRequest, 
    engine: BiomedicalAIEngine = Depends(get_ai_engine)
):
    try:
        logger.info(f"Streaming question for session: {request.session_id}")
        # We return a StreamingResponse to keep the connection open for tokens
        return StreamingResponse(
            engine.stream_question(request.question, request.session_id),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Streaming Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal AI Streaming Error")
# --------------------------------------

@app.post("/upload-manuals")
async def upload_multiple_manuals(
    files: List[UploadFile] = File(...), 
    engine: BiomedicalAIEngine = Depends(get_ai_engine)
):
    try:
        total_processed = 0
        for file in files:
            logger.info(f"Uploading file: {file.filename}")
            file_bytes = await file.read()
            chunks = process_pdf(file_bytes, file.filename)
            engine.update_vector_store(chunks)
            total_processed += 1
            
        return {"message": f"Successfully processed {total_processed} manuals"}
    except Exception as e:
        logger.error(f"Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))