import random
import json
import datetime
import torch
import os
from ics import Calendar, Event

from chatbot_model import NeuralNet
from chatbot_nltk_utils import bag_of_words, tokenize_text

def generate_calendar_event(title, start_time, duration_minutes):
    calendar = Calendar()
    event = Event()
    event.name = title
    event.begin = start_time
    event.duration = {'minutes': duration_minutes}
    event.description = 'Meeting arranged by Sam.'
    calendar.events.add(event)

    with open('meeting.ics', 'w') as file:
        file.writelines(calendar)
    print("Event created and saved to meeting.ics. You can import this file into your calendar application.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as file:
    intents = json.load(file)

MODEL_PATH = "model.pth"
data = torch.load(MODEL_PATH)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Chatbot is ready! (type 'quit' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    user_input = tokenize_text(user_input)
    X = bag_of_words(user_input, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]
    if probability.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")

                if tag == "schedule_meeting":
                    start_time = datetime.datetime.now() + datetime.timedelta(days=1)
                    duration = 60  # 1 hour
                    generate_calendar_event("Meeting with Sam", start_time, duration)
                elif tag == "cancel_meeting":
                    print(f"{bot_name}: Sorry, cancelling a meeting isn't supported via this script.")
    else:
        print(f"{bot_name}: I do not understand...")
