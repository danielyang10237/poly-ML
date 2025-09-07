import mysql.connector
from mysql.connector import Error
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import google.protobuf
import sentencepiece

classes = [
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
class2id = {class_:id for id, class_ in enumerate(classes)}
id2class = {id:class_ for id, class_ in enumerate(classes)}
model_path = "microsoft/deberta-v3-large"

tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from a pretrained state
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification"
)

# Load model weights, ensure the mapping is to the correct device
model.load_state_dict(torch.load('model_epoch_2.pth', map_location=device))

# Move the model to the specified device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

print("using device", device)

def classify(custom_text):
    tokenized_text = tokenizer(custom_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    output = model(**tokenized_text)

    predicted_labels = torch.max(output.logits, 1)[1]
    predicted_class = id2class[predicted_labels.item()]

    return predicted_class

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Catswfq8!",
    database="hp_poly_db"
)
if connection.is_connected():
    db_info = connection.get_server_info()
    print(f"Connected to MySQL Server version {db_info}")
    cursor = connection.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    print(f"You're connected to database: {record}")
else:
    print("Error connecting to MySQL Server")
    raise Error

cursor = connection.cursor()

prev_progress = 0
with open("progress.txt", "r") as f:
    prev_progress = int(f.read())

print("starting from", prev_progress)

idx = 0
with connection.cursor() as cursor:
    cursor.execute('SELECT Id, Message FROM Messages')
    rows = cursor.fetchall()

    for row in rows:
        if idx < prev_progress:
            idx += 1
            continue

        idx += 1
        category = classify(row[1])
        cursor.execute('UPDATE Messages SET Performance_Area = %s WHERE Id = %s', (category, row[0]))

        print(f"Processed {idx} rows")

        if idx % 100 == 0:
            connection.commit()
            with open ("progress.txt", "w") as f:
                f.write(f"{idx}")

cursor.close()

connection.close()

print("Done")