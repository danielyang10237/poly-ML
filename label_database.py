import mysql.connector
import requests
import time
import re

# Establishing the MySQL connection
connection = mysql.connector.connect(
    host="localhost", user="root", password="Catswfq8!", database="competitor_db"
)
cursor = connection.cursor()

# Retrieving messages that don't have associated insights
query = "SELECT Message, id FROM Messages WHERE id NOT IN (SELECT Message_Id FROM INSIGHTS)"
cursor.execute(query)
messages = cursor.fetchall()

# URL for the API services∏
base_url = "http://thing0.polycom.com:5000"

print("Processing messages count", len(messages))

for message_text, message_id in messages:
    # Segmenting the message

    # Removing special characters
    message_text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()\-]', '', message_text).strip()
    
    segment_response = requests.get(f"{base_url}/segment", params={"input_text": message_text})
    if segment_response.status_code != 200:
        print("Segmentation failed with status code:", segment_response.status_code)
        continue
    segments = segment_response.json()

    if len(segments) == 0:
        print("No segments found for message:", message_text)
        continue

    # Processing each segment for sentiment and prediction
    for segment in segments:
        # Getting sentiment analysis
        sentiment_response = requests.get(f"{base_url}/sentiment", params={"input_text": segment})
        if sentiment_response.status_code != 200:
            print("Sentiment analysis failed with status code:", sentiment_response.status_code)
            continue
        sentiment = sentiment_response.text

        # Getting prediction for the performance area
        prediction_response = requests.get(f"{base_url}/predict", params={"input_text": segment})
        if prediction_response.status_code != 200:
            print("Prediction failed with status code:", prediction_response.status_code)
            continue
        performance_area = prediction_response.text

        # Inserting the results into the Insights table
        insert_query = "INSERT INTO Insights (Message_Id, Phrase, Sentiment, Performance_Area) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_query, (message_id, segment, sentiment, performance_area))

        print(f"Inserted segment '{segment}' with sentiment '{sentiment}' and performance area '{performance_area}'.")

        connection.commit()

    # Sleeping for 1 second to avoid rate limiting
    time.sleep(1)

# Closing cursor and printing completion message
cursor.close()
connection.close()
print("Processing complete.")
