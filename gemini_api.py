# gemini_api.py

import google.generativeai as genai
from config import GOOGLE_API_KEY  # Absolute import

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


def get_gemini_response(query):
    prompt = f"""
    You are an expert AI assistant specializing in Alzheimer's Disease. Respond to the user query with accurate, detailed, and helpful information. Focus your responses on providing relevant and comprehensive answers from your expertise of Alzheimer's.

    User's Question: {query}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"
