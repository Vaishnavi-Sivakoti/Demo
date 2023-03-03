from keras.models import load_model
import nltk
import json
import numpy as np
import random
import pickle
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense,Activation, Dropout
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("C:\projects\my website\intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotV')


def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word) for word in sentence_words
    ]
    #print(sentence_words)
    return sentence_words


def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag =[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if w== word:
                bag[i] =1
    #print(bag)
    return np.array(bag)
    

def predicting_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    #print(res)
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    #print(results)
    results.sort(key=lambda x:x[1], reverse= True)
    return_list =[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
        return return_list
    
def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['response'])
            break
    return result

while(True):
    print("Enter your message:")
    message =input("")
    print("to end chatting say: End this")
    if(message == "End this"):
        break
    
    ints = predicting_class(message)
    res = get_response(ints,intents)
    print(res)
    