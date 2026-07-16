import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_PRO_KEY = os.getenv("GEMINI_PRO_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en")
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 8000)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")