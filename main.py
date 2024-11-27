import os
import tensorflow as tf
import numpy as np
import re
import nltk
from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
from google.cloud import firestore
from tensorflow.keras import saving
from tensorflow.keras.layers import Layer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from nltk.data import find
from datetime import datetime

try:
    find('tokenizers/punkt')
    find('tokenizers/punkt_tab')
    print('Punkt ditemukan')
except LookupError:
    print('Punkt tidak ditemukan! downloading...')
    nltk.download('punkt')
    nltk.download('punkt_tab')

app = Flask(__name__)

db = firestore.Client.from_service_account_json('./firestore.json')

@saving.register_keras_serializable()
class RescaleOutput(Layer):
    def __init__(self, **kwargs):
        super(RescaleOutput, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * 5 + 1

# Load ml model dengan custom_objects
model = tf.keras.models.load_model('final_model.h5', custom_objects={'RescaleOutput': RescaleOutput})

# Load pre-trained Word2Vec model
model_w2v = Word2Vec.load('word2vec_model.bin')
VECTOR_SIZE = model_w2v.vector_size

# Text preprocessing functions
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def get_average_word2vec(tokens_list, model, vector_size):
    vec = np.zeros(vector_size)
    valid_words = 0
    for word in tokens_list:
        if word in model.wv:
            vec += model.wv[word]
            valid_words += 1
    if valid_words > 0:
        vec /= valid_words
    return vec

def preprocess_essay(essay_text):
    cleaned_text = clean_text(essay_text)
    tokens = word_tokenize(cleaned_text)
    text_vector = get_average_word2vec(tokens, model_w2v, VECTOR_SIZE)
    text_vector_3d = np.expand_dims(np.expand_dims(text_vector, axis=0), axis=1)
    return text_vector_3d

# Sugesstion
def predict_suggestion(score):
    if score == 1:
        suggestion = 'Suggestion score 1!'
        return suggestion
    elif score == 2:
        suggestion = 'Suggestion score 2!'
        return suggestion
    elif score == 3:
        suggestion = 'Suggestion score 3!'
        return suggestion
    elif score == 4:
        suggestion = 'Suggestion score 4!'
        return suggestion
    elif score == 5:
        suggestion = 'Suggestion score 5!'
        return suggestion
    elif score == 6:
        suggestion = 'Suggestion score 6!'
        return suggestion
    else:
        suggestion = 'Score not detected!'
        return suggestion

@app.route('/', methods=['GET'])
def index():
    hello_json = {
        'error': False,
        'message': 'Success testing the API!',
    }
    db.collection('testing').add(hello_json)
    return jsonify(hello_json), 200

@app.route('/predict', methods=['POST'])
def predict_score():
    try:
        # data = request.json
        # essay_text = data.get('essay', '')  # Pastikan key yang digunakan adalah 'essay'
        
        user_email = request.form.get('email', '')
        essay_title = request.form.get('title', '')
        essay_text = request.form.get('essay', '')
        
        if not essay_text:
            return jsonify({
                'error': True,
                'message': 'Essay text is required!'
            }), 400

        # Preprocessing teks essay
        processed_input = preprocess_essay(essay_text)
        
        # score
        prediction = model.predict(processed_input)
        raw_predicted_score = float(prediction[0][0])
        rounded_predicted_score = int(np.clip(np.rint(raw_predicted_score), 1, 6))
        
        suggestion = predict_suggestion(rounded_predicted_score)
        createdAt = datetime.now().isoformat()+'Z'

        response_data = {
            'error': False,
            'message': 'Essay has been predicted!',
            'title': essay_title,
            'essay': essay_text,
            'predicted_result': {
                'raw_score': raw_predicted_score,
                'score': rounded_predicted_score,
                'suggestion': suggestion
            },
            'createdAt': createdAt,
            'user_email': user_email
        }
        
        db.collection("predictions").add(response_data)
        
        return jsonify(response_data), 200
    except Exception as e:
        print('Error occurred: %s', str(e))
        return jsonify({
            'error': True,
            'message': 'An error occurred while processing the request!',
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', 'localhost')
    http_server = WSGIServer((host, port), app)
    print(f"Server is Ready on {host}:{port}")
    http_server.serve_forever()
