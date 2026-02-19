import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_summary(reviews_text, retries=3):
    prompt = f"""
    You are an expert product analyst for a MicroSD card.
    Analyze the following customer reviews and produce a concise summary.
    Return output strictly in this format:

    Pros:
    - point 1
    - point 2

    Cons:
    - point 1
    - point 2

    Reviews:
    {reviews_text}
    """

    for _ in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash", contents=prompt
            )
            return response.text
        except Exception:
            time.sleep(2)

    return "Model Busy. Try again later."
