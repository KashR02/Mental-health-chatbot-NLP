# Companibot
# Built with
Python
Tensorflow
Flask
javascript
HTML
CSS

## Project:
Companibot is a conversational interface designed for automated text interactions that identify behavioral patterns and provide support to individuals in need. Developed as a natural language understanding application, it utilizes Flask API, jinja2 templates, and JavaScript.

Tweets related to depression were categorized based on a labeled dataset collected through the Twitter API, using TF-IDF weighting to classify their polarity.



### System Diagram
app.py - GUI 

chat.py - Processes user input, identifies intents, and generates responses by interacting with the model and intents

intents.json - Stores user intents and corresponding responses in a structured JSON format

model.py - Defines the neural network used to enhance the chatbot's ability to understand and respond to queries

model_sentiments.h5- A pre-trained model for analyzing the sentiment of user input.

nltk_utils.py - Provides text processing functions (e.g., tokenization, stemming) using NLTK.

diccionary.json - Contains key terms and synonyms to help the chatbot interpret varied user input.

train.py- Trains the deep learning models using data from intents.json and saves the trained models for chatbot use

## Initial Setup:
Clone repo and create a virtual environment
Install dependencies
Install nltk packages

Run
```
$ (venv) python train.py
```
This will dump data.pth file. And then run
the following command to test it in the console.
```
$ (venv) python app.py
```
