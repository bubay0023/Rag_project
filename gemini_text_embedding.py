import os
from dotenv import load_dotenv
from google import genai
load_dotenv('./.env')


def get_text_embedding(text):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=text)
    return result.embeddings[0].values


