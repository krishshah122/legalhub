from fastapi import APIRouter, Request, Form, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging
import re
from docx import Document
from io import BytesIO

router = APIRouter()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Template directory
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Configure Gemini API
try:
    genai.configure(api_key="enter yout gemini api key")
    model = genai.GenerativeModel("gemini-1.5-flash")
    logger.info("✅ Gemini API initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini API: {e}")
    raise RuntimeError(f"Gemini API initialization failed: {e}")

# Load Book and Vector Index
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        logger.info(f"Extracted text from PDF: {path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def split_text(text, max_words=200):
    words = text.split()
    chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks

# Initialize vector index
try:
    book_text = extract_text_from_pdf(r"C:\Users\kriss\Downloads\CompaniesAct2013.pdf")
    book_chunks = split_text(book_text)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embedder.encode(book_chunks)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))
    logger.info("✅ Vector index initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize vector index: {e}")
    raise RuntimeError(f"Vector index initialization failed: {e}")

# Query Function
def get_top_k_chunks(query, k=3, threshold=0.5):
    try:
        query_embedding = embedder.encode([query])
        distances, indices = index.search(np.array(query_embedding), k)
        query_norm = np.linalg.norm(query_embedding)
        top_chunks = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk_vec = chunk_embeddings[idx]
            chunk_norm = np.linalg.norm(chunk_vec)
            cosine_similarity = 1 - dist / (query_norm * chunk_norm + 1e-10)
            if cosine_similarity > threshold:
                top_chunks.append(book_chunks[idx])
        logger.info(f"Retrieved {len(top_chunks)} relevant chunks for query.")
        return top_chunks
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []

def analyze_text(user_text):
    try:
        top_chunks = get_top_k_chunks(user_text)
        if top_chunks:
            context = "\n\n".join(top_chunks)
            prompt = f"""
You are a legal expert assistant trained on the Indian Companies Act, 2013:

--- BOOK CONTEXT START ---
{context}
--- BOOK CONTEXT END ---

Analyze the user's legal document for compliance with the Companies Act, 2013:

--- USER DOCUMENT ---
{user_text}

TASK:
- Identify any violations of the Companies Act, 2013, referencing specific sections.
- Suggest legal improvements, warnings, or missing clauses.
- If compliant, state: "✅ No legal issues found."
- Provide a brief explanation of the analysis, including relevant legal principles.

Respond in this exact format:
⚠️ Issue: [Describe any legal issues or "None"]
💡 Suggestion: [Suggest improvements or "None"]
✅ Validity: [State compliance status, e.g., "Compliant" or "Non-compliant"]
📝 Explanation: [Explain the analysis with references to the Act]
"""
        else:
            prompt = f"""
You are a legal expert on the Indian Companies Act, 2013.

Analyze the user's legal document:

--- USER DOCUMENT ---
{user_text}

TASK:
- Identify any violations of the Companies Act, 2013, referencing specific sections.
- Suggest legal improvements, warnings, or missing clauses.
- If compliant, state: "✅ No legal issues found."
- Provide a brief explanation of the analysis, including relevant legal principles.

Respond in this exact format:
⚠️ Issue: [Describe any legal issues or "None"]
💡 Suggestion: [Suggest improvements or "None"]
✅ Validity: [State compliance status, e.g., "Compliant" or "Non-compliant"]
📝 Explanation: [Explain the analysis with references to the Act]
"""
        response = model.generate_content(prompt)
        logger.info(f"Gemini response: {response.text[:200]}...")
        return response.text
    except Exception as e:
        logger.error(f"Error analyzing text with Gemini: {e}")
        return f"""
⚠️ Issue: Analysis failed
💡 Suggestion: Please try again or contact support
✅ Validity: Unknown
📝 Explanation: Failed to process the document due to an error: {str(e)}
"""

# Parse Gemini Response
def parse_gemini_response(response_text):
    result = {
        "issue": "None",
        "suggestion": "None",
        "validity": "Unknown",
        "explanation": "No analysis provided."
    }
    patterns = {
        "issue": r"⚠️ Issue: (.*?)(?=\n💡|\n✅|\n📝|$)",
        "suggestion": r"💡 Suggestion: (.*?)(?=\n✅|\n📝|$)",
        "validity": r"✅ Validity: (.*?)(?=\n📝|$)",
        "explanation": r"📝 Explanation: (.*?)(?=$)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
    return result

# Extract text from .doc or .docx
def extract_text_from_docx(file_content, filename):
    try:
        doc = Document(BytesIO(file_content))
        text = " ".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        logger.info(f"Extracted text from {filename}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process Word document: {str(e)}")

# Routes
@router.get("/", response_class=HTMLResponse)
async def show_verification_page(request: Request):
    return templates.TemplateResponse("verification.html", {"request": request, "results": None, "error": None})

@router.post("/", response_class=HTMLResponse)
async def verify_document(request: Request, text: str = Form(default=""), file: UploadFile = File(default=None)):
    try:
        user_text = None
        if file and file.filename:
            if not file.filename.lower().endswith(('.pdf', '.txt', '.doc', '.docx')):
                logger.warning(f"Invalid file type uploaded: {file.filename}")
                raise HTTPException(status_code=400, detail="Only PDF, text, or Word (.doc, .docx) files are allowed.")
            content = await file.read()
            if not content:
                logger.warning(f"Uploaded file is empty: {file.filename}")
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            try:
                if file.filename.lower().endswith('.pdf'):
                    doc = fitz.open(stream=content, filetype="pdf")
                    user_text = " ".join(page.get_text() for page in doc)
                    doc.close()
                    logger.info(f"Extracted PDF text from file: {file.filename}, size: {file.size} bytes")
                elif file.filename.lower().endswith(('.doc', '.docx')):
                    user_text = extract_text_from_docx(content, file.filename)
                    logger.info(f"Extracted Word document text from file: {file.filename}, size: {file.size} bytes")
                else:  # .txt
                    user_text = content.decode('utf-8', errors='ignore').strip()
                    logger.info(f"Extracted text from file: {file.filename}, size: {file.size} bytes")
            except fitz.fitz.FileDataError:
                logger.error(f"File is not a valid PDF: {file.filename}")
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")
            except UnicodeDecodeError:
                logger.error(f"File is not a valid text file: {file.filename}")
                raise HTTPException(status_code=400, detail="Uploaded file must be a valid text file (UTF-8 encoded).")
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {str(e)}")
        else:
            user_text = text.strip()
            logger.info(f"Received text input: {user_text[:50]}...")

        if not user_text:
            logger.warning("No valid input provided: text and file are empty.")
            raise HTTPException(status_code=400, detail="Please provide either text or a valid file.")

        # Analyze the text
        analysis_result = analyze_text(user_text)
        parsed_result = parse_gemini_response(analysis_result)
        logger.info(f"Parsed analysis result: {parsed_result}")

        return templates.TemplateResponse("verification.html", {
            "request": request,
            "results": parsed_result,
            "error": None
        })

    except HTTPException as e:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during verification: {e}")
        error_message = f"Verification failed: {str(e)}"
        return templates.TemplateResponse(
            "verification.html",
            {"request": request, "results": None, "error": error_message}
        )

@router.post("/analyze-text", response_class=JSONResponse)
async def analyze_text_api(text: str = Form(...)):
    try:
        result = analyze_text(text)
        parsed_result = parse_gemini_response(result)
        return JSONResponse(content={"analysis": parsed_result})
    except Exception as e:
        logger.error(f"Error in analyze-text API: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/analyze-file", response_class=JSONResponse)
async def analyze_pdf_api(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.pdf', '.txt', '.doc', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF, text, or Word (.doc, .docx) files are allowed.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if file.filename.lower().endswith('.pdf'):
            doc = fitz.open(stream=content, filetype="pdf")
            user_text = " ".join([page.get_text() for page in doc])
            doc.close()
        elif file.filename.lower().endswith(('.doc', '.docx')):
            user_text = extract_text_from_docx(content, file.filename)
        else:  # .txt
            user_text = content.decode('utf-8', errors='ignore').strip()
        result = analyze_text(user_text)
        parsed_result = parse_gemini_response(result)
        return JSONResponse(content={"analysis": parsed_result})
    except fitz.fitz.FileDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid text file (UTF-8 encoded).")
    except Exception as e:
        logger.error(f"Error in analyze-file API: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
