import requests
import os
import csv
import time
from dotenv import load_dotenv

load_dotenv()

# Fetch the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key is not set. Please set the 'OPENAI_API_KEY' environment variable.")

checkpoint_file = "checkpoint.txt"

url = "https://api.openai.com/v1/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

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

def read_stopping_point(filename):
    try:
        with open(filename, 'r') as file:
            return int(file.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

current_point = read_stopping_point(checkpoint_file)
print(f"Resuming from: {current_point}")

def write_stopping_point(filename, stopping_point):
    with open(filename, 'w') as file:
        file.write(str(stopping_point))

# Ensure the output directory exists or handle the case where it does not exist
os.makedirs("reviews_data", exist_ok=True)

def fetch_label(prompt, retries=10):
    data = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": prompt,
        "max_tokens": 5,
        "temperature": 0
    }
    attempt = 0
    while attempt < retries:
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if response.status_code == 503:
                wait_time = (2 ** attempt) + (0.5 * attempt)  # Exponential backoff
                time.sleep(wait_time)∏
                attempt += 1
            else:
                raise
    raise Exception("Failed to fetch label after several retries.")

counter = 0
with open("reviews_data/reviews_labeled.csv", "a", newline="") as outfile:  # Changed to append mode
    writer = csv.writer(outfile)
    if os.path.getsize("reviews_data/reviews_labeled.csv") == 0:  # Check if file is empty
        writer.writerow(["Review", "Label"])  # Write header row if file is empty
    with open("reviews_data/all_reviews.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if counter < current_point:
                counter += 1
                continue

            review = row[0]
            prompt = f"{PROMPT_INTRO}{PROMPT_SUBTITLE}{PROMPT_BODY}\"{review}\""

            try:
                response_text = fetch_label(prompt)
                for char in ".!?,;:":
                    response_text = response_text.replace(char, "")

                response_text = response_text.strip()
                writer.writerow([review, response_text])
            except Exception as e:
                print(f"Failed to process review: {review} with error: {e}")
                write_stopping_point(checkpoint_file, counter)
                continue

            counter += 1

            # Update checkpoint after processing each review
            write_stopping_point(checkpoint_file, counter)

            if counter % 100 == 0:
                print(f"Processed {counter} reviews")

print("Done")
