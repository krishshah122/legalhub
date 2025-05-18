import os
import sys
import logging
from pathlib import Path
from decouple import config, Config, RepositoryEnv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
import google.api_core.exceptions
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.debug(f"Added {project_root} to sys.path")

# Load .env file
env_path = os.path.join(project_root, '.env')
if not os.path.exists(env_path):
    logger.error(f".env file not found at: {env_path}")
    raise FileNotFoundError(f".env file not found at: {env_path}")

try:
    config_env = Config(RepositoryEnv(env_path))
    logger.debug(f".env file loaded successfully from: {env_path}")
except Exception as e:
    logger.error(f"Failed to load .env file: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to load .env file: {e}")

# Configure Gemini API
GOOGLE_API_KEY = config_env('GOOGLE_API_KEY', default='')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is missing in .env.")
    raise RuntimeError("GOOGLE_API_KEY is required.")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.debug("Google Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Google Gemini API: {e}")
    raise HTTPException(status_code=500, detail="Failed to configure Google Gemini API")

# Initialize FastAPI app
app = FastAPI(title="AI Legal Advisor Service")

# Setup templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Include verify_agent router
try:
    from legal.verify_agent import router as verify_router
    app.include_router(verify_router, prefix="/verification", tags=["verification"])
    logger.debug("verify_agent router included successfully.")
except Exception as e:
    logger.error(f"Failed to include verify_agent router: {e}")
    raise RuntimeError(f"Failed to include verify_agent router: {e}")

# Request model
class Prompt(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    logger.debug("Serving GET /")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/filling", response_class=HTMLResponse)
async def filing_agent(request: Request):
    return templates.TemplateResponse("filling.html", {"request": request})
@app.get("/agents", response_class=HTMLResponse)
async def get_agents_page(request: Request):
    logger.debug("Serving GET /agents")
    return templates.TemplateResponse("agent.html", {"request": request})
@app.get("/draft", response_class=HTMLResponse)
async def get_draft_page(request: Request):
    logger.debug("Serving GET /draft")
    return templates.TemplateResponse("draft.html", {"request": request})

@app.get("/test")
async def test():
    logger.debug("Serving GET /test")
    return {"message": "Test route working!"}

# Retry logic for Gemini API
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(32),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    before_sleep=lambda retry_state: logger.debug(
        f"Quota exceeded, retrying in 32 seconds... Attempt {retry_state.attempt_number}"
    )
)
async def generate_draft_content(prompt: str):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Failed to generate content with Gemini: {e}")
        raise

@app.post("/draft")
async def generate_draft(prompt: Prompt):
    logger.debug(f"Serving POST /draft with prompt: {prompt.prompt[:50]}...")
    try:
        draft_text = await generate_draft_content(prompt.prompt)
        logger.info("Draft generated successfully.")
        return JSONResponse(content={"draft": draft_text})
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"Gemini API quota exceeded after retries: {e}")
        return JSONResponse(content={"error": "Gemini API quota exceeded"}, status_code=429)
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return JSONResponse(content={"error": f"Error generating content: {str(e)}"}, status_code=500)

@app.get("/debug")
async def debug():
    logger.debug("Serving GET /debug")
    router_names = [r.tags[0] for r in app.routes if hasattr(r, 'tags') and r.tags]
    return {"message": "Server is running", "router_included": "verification" in router_names}