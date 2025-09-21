import os
from google import genai
from google.genai.types import GenerateContentConfig

def query_gemini(prompt: str, model_name: str = "gemini-1.5-flash", max_tokens: int = 300):
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # Configure generation
        generation_config = GenerateContentConfig(
            max_output_tokens=max_tokens
        )

        # Generate response
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generation_config
        )

        return response.text.strip()

    except Exception as e:
        raise Exception(f"Gemini API Error: {e}")
