import os
import ollama
from dotenv import load_dotenv
from google import genai
load_dotenv('./.env')



class EmbeddingModel:
    
    def get_gemini_text_embedding(self, text):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text)
        return result.embeddings[0].values

    def get_ollama_text_embedding(self, text):
        res = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return res['embedding']
