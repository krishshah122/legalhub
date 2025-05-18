import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("AIzaSyB3bz0Z6X2dXeaHwIpk5xlWBN8CB_-AcNM")
genai.configure(api_key=GEMINI_API_KEY)

# Set up Gemini model
model = genai.GenerativeModel("gemini-pro")

def generate_legal_draft(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"
