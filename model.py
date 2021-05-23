#import all libraries

import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

sw = stopwords.words('english')
wordnet = WordNetLemmatizer()

#Load the data

tdata = pd.read_csv("Twitter Sentiments.csv")

#Change the text to lower case
tdata.loc[:,'tweet'] = tdata.loc[:,'tweet'].apply(lambda x: x.lower())

#Remove @ attached to words @user
tdata.loc[:,'tweet'] = tdata.loc[:,'tweet'].apply(lambda x : re.sub(r"@#\S", " ",x))

#Remove punctuations,special characters, spaces
tdata.loc[:,'tweet'] = tdata.loc[:,'tweet'].apply(lambda x : re.sub("[^a-zA-Z#]", " ",x))

#Remove stopwords
tdata.loc[:,'tweet'] = tdata.loc[:,'tweet'].apply\
(lambda x : " ".join([wordnet.lemmatize(word, pos='v') for word in x.split() if word not in (sw)]))

#splitting the data into train and test
X_train,X_test,y_train,y_test = train_test_split(tdata['tweet'].values, tdata['label'].values, test_size = 0.2, random_state = 101)

train_data = pd.DataFrame({'tweet': X_train, 'label':y_train})
test_data = pd.DataFrame({'tweet': X_test, 'label':y_test})

#converting to vectors
vector = TfidfVectorizer()
train_vector = vector.fit_transform(train_data['tweet'])
test_vector = vector.transform(test_data['tweet'])



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score

#train
model = LogisticRegression()
model.fit(train_vector,y_train)

#testing
pred = model.predict(test_vector)
f1_score(y_test,pred)
accuracy_score(y_test,pred)

#Save the model
import joblib
model_file_name = 'TwitterSentiment.pkl'
vector_file_name  = 'TwitterVector.pkl'
joblib.dump(model, model_file_name)
joblib.dump(vector, vector_file_name)
