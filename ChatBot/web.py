from flask import Flask, render_template, request
import random
import json
import datetime
import torch
import os
from ics import Calendar, Event

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Function to generate an iCalendar event on Linux
def generate_ics_event(title, start_time, duration_minutes):
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

with open('/home/shravan/Desktop/Internship/SkillRaace/Task_1/ChatBot/pytorch-chatbot/intents.json', 'r') as file:
    intents = json.load(file)

MODEL_PATH = "data.pth"
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

def get_response(user_input):
    user_input = tokenize(user_input)
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
                if tag == "schedule_meeting":
                    # For simplicity, using a fixed time and duration
                    start_time = datetime.datetime.now() + datetime.timedelta(days=1)
                    duration = 60  # 1 hour
                    generate_ics_event("Meeting with Sam", start_time, duration)
                elif tag == "cancel_meeting":
                    response = "Sorry, cancelling a meeting isn't supported via this script."
                return response
    else:
        return "I do not understand..."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))

if __name__ == "__main__":
    app.run()
