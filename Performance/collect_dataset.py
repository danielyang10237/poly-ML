import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

assert api_key, "API key not found. Set it as an environment variable named 'OPENAI_API_KEY'."

# Define API URL and headers
url = "https://api.openai.com/v1/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Define categories and prompt parts
categories = [
    "Customer Service",
    "Audio",
    "Visual",
    "Bluetooth",
    "Microphone",
    "Battery Life",
    "Internet",
    "Performance/Speed",
    "Value/Price",
    "Ease of Use",
    "Comfortability/Fit",
    "Undetermined"
]
PROMPT_INTRO = "Label the product review topic into the following categories: " + ", ".join(categories)
PROMPT_SUBTITLE = ". Choose and return one option and nothing else. "
PROMPT_BODY = "Review: "

# Ensure the output directory exists
os.makedirs("reviews_data", exist_ok=True)

# Function to process a single review
def process_review(review):
    prompt = f"{PROMPT_INTRO}{PROMPT_SUBTITLE}{PROMPT_BODY}\"{review}\""
    data = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": prompt,
        "max_tokens": 5,
        "temperature": 0
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        print("Failed to get a response for the review:", review)
        return review, None

    response_json = response.json()
    response_text = response_json["choices"][0]["text"].strip()
    for char in ".!?,;:":
        response_text = response_text.replace(char, "")
    response_text = response_text.strip()
    return review, response_text

test_review = "The battery life is amazing, but the microphone quality is poor."

review, response_text = process_review(test_review)

print(f"Review: {review}")
print(f"Response: {response_text}")