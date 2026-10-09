"""FastAPI backend: Streamlit -> FastAPI -> LangGraph -> Gemini."""
import asyncio, logging, os, uuid
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.graph import run_graph

load_dotenv()
log = logging.getLogger("content-generator")
VALID_TYPES = ["Blog Post", "Social Media Caption", "Text Summary"]

app = FastAPI(title="AI Content Generator API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    content_type: str = ""
    topic: str = ""
    tone: str = "Professional"
    instructions: str = ""
    previous_content: Optional[str] = None

class RefineRequest(BaseModel):
    content_type: str = ""
    topic: str = ""
    tone: str = "Professional"
    previous_content: str = ""
    refinement_instruction: str = ""

def _bad(msg: str, code: int = 400):
    raise HTTPException(status_code=code, detail=msg)

def _check_type(ctype: str):
    if ctype not in VALID_TYPES:
        _bad(f"Invalid content type '{ctype}'. Choose one of: {', '.join(VALID_TYPES)}.")

def _check_env():
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key == "your_api_key":
        _bad("GEMINI_API_KEY is missing. Set it in .env.", 500)

def _result(text: str) -> dict:
    return {"success": True, "content": text, "word_count": len(text.split()), "character_count": len(text)}

def _validate_generation(req: GenerateRequest):
    _check_type(req.content_type)
    if not req.topic.strip():
        _bad("Topic is required.")

def _payload(mode: str, req: GenerateRequest) -> dict:
    return {"mode": mode, "content_type": req.content_type, "topic": req.topic.strip(),
            "tone": req.tone, "instructions": req.instructions.strip(),
            "previous_content": req.previous_content, "variation_seed": uuid.uuid4().hex[:8]}

async def call_langgraph(payload: dict) -> str:
    _check_env()
    try:
        return await asyncio.to_thread(run_graph, payload)
    except Exception as exc:
        log.exception("LangGraph/Gemini failure")
        raise HTTPException(status_code=502, detail=str(exc))

@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=400, content={"success": False, "detail": f"Invalid request: {msg}"})

@app.exception_handler(HTTPException)
async def http_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "detail": exc.detail})

@app.get("/api/health")
async def health():
    key = os.getenv("GEMINI_API_KEY", "")
    return {"status": "ok", "gemini_key_configured": bool(key and key != "your_api_key"),
            "orchestration": "LangGraph", "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash")}

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    _validate_generation(req)
    return _result(await call_langgraph(_payload("generate", req)))

@app.post("/api/regenerate")
async def regenerate(req: GenerateRequest):
    _validate_generation(req)
    return _result(await call_langgraph(_payload("regenerate", req)))

@app.post("/api/refine")
async def refine(req: RefineRequest):
    _check_type(req.content_type)
    if not req.previous_content.strip():
        _bad("previous_content is required for refinement.")
    if not req.refinement_instruction.strip():
        _bad("refinement_instruction is required.")
    payload = {"mode": "refine", "content_type": req.content_type, "topic": req.topic.strip(),
               "tone": req.tone, "instructions": "", "previous_content": req.previous_content,
               "refinement_instruction": req.refinement_instruction.strip(),
               "variation_seed": uuid.uuid4().hex[:8]}
    return _result(await call_langgraph(payload))
