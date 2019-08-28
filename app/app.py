#-------------------------------------------------------------------------------------------
#
# Tweet Author Classification Tool
#
# Thx https://medium.com/fintechexplained/flask-host-your-python-machine-learning-model-on-web-b598151886d
# Make sure VM is activated - bash workon my-virtualenv (https://help.pythonanywhere.com/pages/Flask/)
#
#-------------------------------------------------------------------------------------------

from flask import Flask, render_template, request, url_for, redirect
import re
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer

#-------------------------------------------------------------------------------------------
# For Display
#-------------------------------------------------------------------------------------------

originallabels = {0: '@barackobama', 1: '@calvinstowell', 2: '@kimkardashian'}

def howsure(probests, prediction):
    '''Defines logic to translate class probability estimates to sureness gif'''
    probpred = probests[0][prediction]
    otherprobs = float(sum(probests[0]) - probpred)
    # Not sure
    if probpred < .7:
        return "not sure though."
    # Very sure
    elif probpred > .95 and otherprobs < .1:
        return "very sure!"
    else:
        return "kinda sure."

def surestatic(howsuretext):
  '''Gets gif to show sureness'''
  if howsuretext == "not sure though.":
    return "sadkim.gif"
  elif howsuretext == "very sure!":
    return "happyobama.gif"
  else:
    return "maybe.gif"

#-------------------------------------------------------------------------------------------
# Text Processing
#-------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------
# Model Import
#-------------------------------------------------------------------------------------------

# Feature Extraction
# Retrain Count Vectorizer model
wordfeats = pickle.load(open('/home/campkels/mysite/app/wordfeats.pkl', 'rb'))
bow_transformer = CountVectorizer(analyzer=textonly).fit(wordfeats)

# load the model
model = pickle.load(open('/home/campkels/mysite/app/finalized_model.pkl','rb'))

#-------------------------------------------------------------------------------------------
# Flask
#-------------------------------------------------------------------------------------------

# Flask
app = Flask(__name__)

@app.route('/')
def show_predict_form():
    return render_template('predictorform.html')

@app.route('/result', methods=['POST'])
def results():
    if request.method == 'POST':
      # Get input
      rawtext = request.form['tweet']
      # Transform raw input for model
      modelinput=bow_transformer.transform([rawtext])
      # Predict
      predicted = model.predict(modelinput)
      predicted_label = originallabels[predicted[0]]
      print(predicted)
      # Get Sureness
      probests = model.predict_proba(modelinput)
      sureness = howsure(probests, predicted)
      suregif = surestatic(sureness)

      return render_template('resultsform.html', text=rawtext, predicted_author=predicted_label, sureness=sureness, suregif=suregif)

@app.route("/batchresult", methods=['POST'])
def handleFileUpload():
    #Implement!
    return render_template('batchresultsform.html')

#app.run(port=5000, debug=True)
# if __name__ == '__main__':
#     app.run()