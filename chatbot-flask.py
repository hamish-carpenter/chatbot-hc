import random
import json
import pickle
import numpy as np

import requests
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

import wikipediaapi

disable_warnings(InsecureRequestWarning)

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

ERROR_THRESHOLD = 0.50 # may need to lower?

lemmatizer = WordNetLemmatizer() # initialises nltk's WordNetLematizer to be used later
intents = json.loads(open('intents.json').read()) # loads intents file

# opens previously saved (in training.py) files for the words, classes arrays for the chatbot to compare to after lemmatising the user's query

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5") # loads the model created in training.py

def google_definition(query: str) -> str:
    site = requests.get(f"https://serpapi.com/search.json?engine=google&q={query}&api_key=ec8da0fa28fb02d9765e24b9eb569d59077718cde111d45a4f29c75da25966c7", verify=False).json() # uses the serpapi engine to fetch the list of results of a given query, then returns the JSON file gained from this

    if graph := site.get("knowledge_graph", False): # checks for the "knowledge graph" element of a google search page, and returns the title of this knowledge graph
        try:
            return graph["description"] # returns the description of the knowledge graph
        except:
            print("no description????")
    else:
        wiki_wiki = wikipediaapi.Wikipedia('en') # initialises wikipediaapi to be used later
        try:
            page_py = wiki_wiki.page(query) # searches wikipedia for the query
            return "From Wikipedia: ", page_py.title, "\nSummary: ", page_py.summary[0:250] # returns page title and summary of up to 250 character if applicable
        except:
            return "I'm sorry, I don't know what that is." # returns error message if no knowledge graph or wikipedia page is found



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) # splits the user input into separate words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] # lemmetises sentence_words: splits them into their root word (e.g. "worked", "working" -> "work")
    return sentence_words

def bag_of_words(sentence): # allows for the use of numbers instead of words as neural networks cannot accept strings as values
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) # creates initial array for bag of words
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 # sets array item to 1 if word is present
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0] # uses the model trained in training.py to give a response within a defined error threshold (0.25)

    predicted_class = np.argmax(result, axis=0)

    return classes[predicted_class], result[predicted_class]

def get_response(predicted_class): # chooses random reposnse from list of intents & returns it
    for t in intents['intents']:
        if t['tag'] == predicted_class:
            return random.choice(t['responses'])

print("bot is running")

while True: # uses previous functions to get input, split it into its lemmetised words, then predict & return a response
    message = input("")
    predicted_class, accuracy = predict_class(message)
    print(accuracy)
    if accuracy < ERROR_THRESHOLD:
        print("probability too low, returning google search answer")
        res = google_definition(message)
    else:
        res = get_response(predicted_class)
    print(res)
