# Thx https://medium.com/fintechexplained/flask-host-your-python-machine-learning-model-on-web-b598151886d
# Make sure VM is activated - bash workon my-virtualenv (https://help.pythonanywhere.com/pages/Flask/)

from flask import Flask, render_template, request, url_for, redirect
import os
import re
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# For Display
originallabels = {0: '@barackobama', 1: '@calvinstowell', 2: '@kimkardashian'}

# Customize stopwords, punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['wouldnt', 'wont', 'werent', 'wasnt', 'shouldnt', 'neednt', 'isnt', 'havent', 'hasnt', 'hadnt', 'ive','doesnt', 'didnt', 'couldnt', 'arent', 'aint', 'amp'])
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~’–“”'

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
#wordfeats = pickle.load(open('/home/campkels/mysite/app/wordfeats.pkl', 'rb'))
bow_transformer = CountVectorizer(analyzer=textonly).fit(wordfeats)

# load the model
model = pickle.load(open('finalized_model.pkl','rb'))
#model = pickle.load(open('/home/campkels/mysite/app/finalized_model.pkl','rb')) 

# Flask
app = Flask(__name__)

@app.route('/')
def show_predict_form():
    return render_template('predictorform.html')
@app.route('/result', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
      # Get input
      rawtext = request.form['tweet']
      # Transform raw input for model
      modelinput=bow_transformer.transform([rawtext])
      # Predict
      predicted = model.predict(modelinput)
      predicted_label = originallabels[predicted[0]]
      print(predicted)
      return render_template('resultsform.html', text=rawtext,   predicted_author=predicted_label)

@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
	if 'csv' in request.files:
		csv = request.files['csv']
		if csv.filename != '':
			csv.save(os.path.join('C:/Users/Kelsey/Desktop/jobs/TheTrevorProject/Project/app/uploads', csv.filename))
	return redirect(url_for('handleFileUpload'))

# Comment this out when deploy
app.run(port=5000, debug=True)
