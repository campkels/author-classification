# Thx https://medium.com/fintechexplained/flask-host-your-python-machine-learning-model-on-web-b598151886d

from flask import Flask, render_template, request
import re
import nltk
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# For Display
originallabels = {0: '@barackobama', 1: '@calvinstowell', 2: '@kimkardashian'}

# Customize stopwords, punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['wouldnt', 'wont', 'werent', 'wasnt', 'shouldnt', 'neednt', 'isnt', 'havent', 'hasnt', 'hadnt', 'ive','doesnt', 'didnt', 'couldnt', 'arent', 'aint', 'amp'])
punctuation = string.punctuation + '’' + '–' + '“' + '”'

# Text Processing
def textonly(text):
    # Remove URLS
    text = re.sub('https?:\/\/.*', '', text)
    # Remove Punctuation
    text = re.sub('—', ' ', text)
    text  = "".join([char for char in text if char not in punctuation])
    # Lowercase
    text = text.lower()
    # Remove Stopwords, Tokenize
    return [word for word in nltk.word_tokenize(text) if word not in stopwords]

# Feature Extraction
# Retrain Count Vectorizer model
wordfeats = pickle.load(open('wordfeats.pkl', 'rb'))
bow_transformer = CountVectorizer(analyzer=textonly).fit(wordfeats)

# Flask
app = Flask('author_predict')

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')
@app.route('/result', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
      # load the model
      model = pickle.load(open('finalized_model.pkl','rb')) 
      # Get input
      rawtext = request.form['tweet']
      # Transform raw input for model
      modelinput=bow_transformer.transform([rawtext])
      # Predict
      predicted = model.predict(modelinput)
      predicted_label = originallabels[predicted[0]]
      print(predicted)
      return render_template('resultsform.html', text=rawtext,   predicted_author=predicted_label)

app.run(port=5000, debug=True)