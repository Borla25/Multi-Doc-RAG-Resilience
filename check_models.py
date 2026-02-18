import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- MODELLI DISPONIBILI PER LA TUA CHIAVE ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Nome: {m.name}")