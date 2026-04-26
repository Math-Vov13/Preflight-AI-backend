from os import environ as env

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()


llm = GoogleGenAI(
    model=env.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
    api_key=env.get("GOOGLE_API_KEY"),
    temperature=0.3,
    max_tokens=7000,
)
