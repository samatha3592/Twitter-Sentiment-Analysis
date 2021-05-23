import joblib
from flask import Flask, request, render_template

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

app = Flask(__name__)

#Loading the model and tfidfVectorizer
vector = joblib.load('TwitterVector.pkl')
tmodel = joblib.load('TwitterSentiment.pkl')

def LGR(tweet):
    vectorizer = vector.transform([tweet])
    my_pred = tmodel.predict(vectorizer)

    if my_pred==1:
        return("Tweet is Racist/Sexist")
    else:
        return('Tweet is not Racist/Sexist')


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/FinalTweet', methods = ['POST'])
def FinalTweet():
    tweet = request.form['Tweet']
    result = LGR(tweet)

    return render_template('index.html', final_text = result)


if __name__=="__main__":
    app.run(debug=True)


