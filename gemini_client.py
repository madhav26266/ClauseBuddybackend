import os
from google.genai import Client, types

def query_gemini(prompt: str, model_name: str = "gemini-1.5-flash", max_tokens: int = 300):
    try:
        client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generation_config
        )

        return response.text.strip()

    except Exception as e:
        raise Exception(f"Gemini API Error: {e}")