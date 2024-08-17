import random
import json
import torch
import numpy as np

from model import NeuralNet
from nltk_utils import bag_of_words, codificacion_sentence, stopwords, tokenize, padding
from tensorflow.keras.models import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding="utf8") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load models

# Torch model: classifies conversation topics
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Keras model: classifies sentiment
# Close to 0 for joy
# Close to 1 for sadness
model_sentiments = load_model('modelo_sentimientos_lemma.h5')

bot_name = "CompaniBot"

def get_response(msg, score):
    # Clean the sentence for the Torch model
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Clean the sentence for the Keras model
    sentence_sentiments = stopwords(sentence)
    X_sentimientos = codificacion_sentence(sentence_sentiments)
    
    # Torch prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    tag = tags[predicted.item()]

    # Keras prediction
    # Value between 0 and 1
    output_sentimientos = model_sentiments.predict(padding(X_sentimientos))
    input_score = output_sentimientos[0]
    # If the model can't recognize the input, set the value to 0
    non_valid_input_score = [0]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    pred_item = prob.item()
    
    # Change tags in order to lead the conversation
    if len(score) == 1:
        tag = 'interest'
    elif len(score) == 2:
        tag = 'time'
    elif len(score) == 3:
        tag = 'support'
       
    # Change tags in order to lead the conversation 
    #elif len(score) == 4:
      #  tag = 'Motivation'
    # Release the direction of the speech, so the user can access more conversation options or finish it anytime
    elif len(score) >= 4 and pred_item < 0.75 or len(score) > 4 and tag == 'about':
        # Tag-about prevents the bot from redirecting the conversation to the greeting phase if 'day' is used
        if tag == 'about':
            input_score = [0]
        score.pop(0)
        average = np.average(score)
        print(average)
        if average < 0.2:
            tag = 'joy'
        else:
            tag = 'support' 

    # Select an answer for our bot
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                # Check if the tag is 'Motivation'
                if tag == 'Motivation':
                    # Return a random response from the 'Motivation' intent
                    return random.choice(intent['responses']), input_score
                if tag == 'recommendation':
                    return random.choice(intent['responses']), input_score
                else:
                    # Return a random response from the intent
                    return random.choice(intent['responses']), input_score

    return "Sorry, what do you mean by that? \nI'm here to help", non_valid_input_score


if __name__ == "__main__":
    print(random.choice(intents['intents']['tag'=='greeting']['responses']))
    
    score_resp = []
    
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        answer = get_response(sentence, score_resp)
        resp = answer[0]
        score_resp.append(answer[1][0])
        
        print(resp)
        print(score_resp)

        if answer[0] != "Sorry, what do you mean by that? \nI'm here to help":
            # Check if the tag was 'support'
            if answer[0].startswith("Here are some recommendations for you:"):
                # Now you can provide recommendations
                print("Recommendation 1: Consider trying meditation for stress relief.")
                print("Recommendation 2: Take short breaks during work for better productivity.")
                print("Recommendation 3: Try reading a book or listening to music to relax.")

