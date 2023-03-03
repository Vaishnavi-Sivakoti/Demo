import nltk
import json
import numpy as np
import random
import pickle
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense,Activation, Dropout
from keras.optimizers import SGD 
lemmatizer = WordNetLemmatizer();
intents = json.loads(open("C:\projects\my website\intents.json").read())

words =[]
classes =[]
documents =[]
ignore_letter =['?',',','.','!',';']
for intent in intents['intents']:
    for pattern in intent['pattern']:
        word_list = nltk.word_tokenize(pattern)
        #print(word_list)
        words.extend(word_list)
        #print(words)
        documents.append(((word_list),intent['tag']))
        #print(documents)
        if(intent['tag'] not in classes):
            classes.append(intent['tag'])
        #print(classes)
        
words =[lemmatizer.lemmatize(word) for word in words if word not in ignore_letter]

words = sorted(set(words))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
#print(words)

training=[]
output_empty =[0]*len(classes)

for document in documents:
    bag =[]
    word_patterns =document[0]
    word_patterns =[lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    #print(word_patterns)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] =1
    training.append([bag, output_row])
    #print(training)
    
#print(bag)
random.shuffle(training)
training = np.array(training,dtype=object)
#print(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
#print(train_x)
#print(train_y)

model = Sequential()

model.add(Dense(128,input_shape = (len(train_x[0]),),activation ='relu', ))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
#print(hist)
model.save("chatbotV", hist)