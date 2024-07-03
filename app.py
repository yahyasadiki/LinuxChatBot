from flask import Flask, request, render_template
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

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
chat_history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    global chat_history
    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input:
            # Append user input to chat history
            chat_history.append({"sender": "User", "text": user_input})

            sentence = tokenize(user_input)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        bot_response = random.choice(intent['responses'])
                        chat_history.append({"sender": bot_name, "text": bot_response})
            else:
                bot_response = "I do not understand..."
                chat_history.append({"sender": bot_name, "text": bot_response})

    return render_template('index.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")

