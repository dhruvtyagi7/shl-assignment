"""
list_models.py

Quick utility to print all Gemini models that support content generation.
Useful for checking which models your API key has access to.

Usage: python list_models.py
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Models available for generateContent:\n")
for model in client.models.list():
    if hasattr(model, "supported_actions") and "generateContent" in (model.supported_actions or []):
        print(f"  {model.name}")
    elif hasattr(model, "supported_generation_methods") and "generateContent" in (model.supported_generation_methods or []):
        print(f"  {model.name}")
