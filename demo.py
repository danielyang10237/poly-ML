from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import torch.nn.functional as F
import google.protobuf
import sentencepiece

sentiment_tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english", num_labels=2)

device = torch.device("cpu")
sentiment_model.to(device)
print("Model is on CPU")

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

performance_tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

performance_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification"
)

performance_model.load_state_dict(torch.load('Performance/model_epoch_2.pth', map_location=device))

performance_model = performance_model.to(device)

performance_model.eval()

print("using device", device)

def get_performance(custom_text):
    tokenized_text = performance_tokenizer(custom_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    output = performance_model(**tokenized_text)

    predicted_labels = torch.max(output.logits, 1)[1]
    predicted_class = id2class[predicted_labels.item()]

    return predicted_class

def get_sentiment(CUSTOM_INPUT):
    tokenized_input = sentiment_tokenizer(CUSTOM_INPUT, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']

    with torch.no_grad():
        outputs = sentiment_model(input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = outputs.logits
        prediction = logits.argmax(dim=1).item()

    return "Positive" if prediction == 1 else "Negative"



def show_output():
    input_text = entry.get()
    sentiment_result = get_sentiment(input_text)
    performance_result = get_performance(input_text)

    sentiment_label.config(text=f"Sentiment: {sentiment_result}")
    performance_label.config(text=f"Performance: {performance_result}")
    entry.delete(0, tk.END)


import tkinter as tk
from tkinter import font

def show_output():
    input_text = entry.get()
    if input_text == "":
        return
    # Update the original text label
    original_text_label.config(text=f"Original Text: {input_text}", fg="#0000FF")  # Blue color for the original text
    print("getting sentiment")
    sentiment_result = get_sentiment(input_text)
    print("getting performance")
    performance_result = get_performance(input_text)
    sentiment_label.config(text=f"Sentiment: {sentiment_result}")
    performance_label.config(text=f"Performance: {performance_result}")
    entry.delete(0, tk.END)

def on_enter(e):
    button.config(bg='#004d00', fg='black', cursor='hand2')  # Darker green and black text on hover

def on_leave(e):
    button.config(bg='#006400', fg='white', cursor='')  # Original colors and cursor

# Create the main window
root = tk.Tk()
root.title("Futuristic GUI")

# Set the background color
root.configure(bg='#333')

# Adjusting font sizes
entry_font = font.Font(family="Helvetica", size=44, weight="bold")  # Doubled size for entry and labels
title_font = font.Font(family="Helvetica", size=56, weight="bold", slant="italic")  # Doubled size for title

# Title label
title_label = tk.Label(root, text="NLP Demo", font=title_font, fg="#00FF00", bg='#333')
title_label.pack(pady=(20, 10))

# Create a text entry widget with custom styling
entry = tk.Entry(root, font=entry_font, fg="#00FF00", bg="#111", insertbackground="#00FF00", width=50)
entry.pack(pady=20, padx=20)

# Bind the Enter key to trigger the show_output function when pressed
root.bind('<Return>', lambda event: show_output())

# Label to display the original text
original_text_label = tk.Label(root, text="", font=entry_font, fg="#0000FF", bg="#333")
original_text_label.pack(pady=10, padx=20)

# Create labels for both outputs
sentiment_label = tk.Label(root, text="", font=entry_font, fg="#FFF", bg="#333")
sentiment_label.pack(pady=10, padx=20)

performance_label = tk.Label(root, text="", font=entry_font, fg="#FFF", bg="#333")
performance_label.pack(pady=10, padx=20)

# Create a button with custom styling that triggers the show_output function
button = tk.Button(root, text="Submit", font=entry_font, fg="#FFF", bg="#006400", command=show_output)
button.pack(pady=10, padx=20)
button.bind('<Enter>', on_enter)
button.bind('<Leave>', on_leave)

# Start the GUI event loop
root.mainloop()