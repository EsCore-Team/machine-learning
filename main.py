import os
import tensorflow as tf
import numpy as np
import re
import nltk
import pickle
import fitz
from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
from google.cloud import firestore, storage
from nltk.corpus import stopwords
from nltk.data import find
from datetime import datetime
from keras.preprocessing.sequence import pad_sequences

try:
    find('tokenizers/punkt')
    find('tokenizers/punkt_tab')
    find('tokenizers/stopwords')
except LookupError:
    print('Punkt tidak ditemukan! downloading...')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

app = Flask(__name__)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

db = firestore.Client.from_service_account_json('service.json')
gcs = storage.Client.from_service_account_json('service.json')

class QuadraticWeightedKappa(tf.keras.metrics.Metric):
    def __init__(self, name='quadratic_weighted_kappa', **kwargs):
        super(QuadraticWeightedKappa, self).__init__(name=name, **kwargs)
        self._qwk = self.add_weight(name='qwk', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        qwk_value = tf.numpy_function(self._quadratic_weighted_kappa, (y_true, y_pred), tf.float32)
        qwk_value.set_shape([])
        self._qwk.assign(qwk_value)

    def result(self):
        return self._qwk

    def reset_states(self):
        self._qwk.assign(0.0)

model = tf.keras.models.load_model('best_model_lstm.h5', custom_objects={'QuadraticWeightedKappa': QuadraticWeightedKappa})

# Muat tokenizer dari file
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def preprocess_essay(essay_text):
    cleaned_text = clean_text(essay_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_input = pad_sequences(sequences, maxlen=1024)
    return padded_input

def predict_suggestion(rounded_score):
    suggestions = {
        1: 'Suggestion score 1!',
        2: 'Suggestion score 2!',
        3: 'Suggestion score 3!',
        4: 'Suggestion score 4!',
        5: 'Suggestion score 5!',
        6: 'Suggestion score 6!'
    }
    return suggestions.get(rounded_score, 'No suggestions available.')

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ''
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text('text')
        text += page_text
    if text:
        return text
    else:
        return 'No text found in this PDF.'

def upload_to_gcs(bucket_name, file_obj, destination_blob_name):
    file_obj.stream.seek(0)
    bucket = gcs.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file_obj.stream, content_type=file_obj.mimetype)
    
    public_url = f'https://storage.googleapis.com/{bucket_name}/{destination_blob_name}'
    print(f'File is publicly accessible at: {public_url}')

    return public_url

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
        user_email = request.form.get('user_email', '')
        essay_title = request.form.get('title', '')
        essay_text = request.form.get('essay', '')
        pdf_file = request.files.get('pdf_file', '')
        
        gcs_link = None
        timestamp = datetime.now().isoformat()+'Z'
        
        if not user_email or not essay_title:
            return jsonify({
                'error': True,
                'message': 'All fields are required!'
            }), 400
        
        if not essay_text and not pdf_file:
            return jsonify({
                'error': True,
                'message': 'Essay text or PDF file is required!'
            }), 400
        
        # Query untuk mencari email di collection "users"
        user_ref = db.collection('users')
        query = user_ref.where('email', '==', user_email).limit(1)
        user_doc = query.get()

        if not user_doc:
            return jsonify({
                'error': True,
                'message': 'User email not found!'
            }), 404
        
        if pdf_file:
            if not pdf_file.filename.endswith('.pdf'):
                return jsonify({
                    'error': True,
                    'message': 'Uploaded file is not a PDF!'
                }), 400
                
            pdf_path = os.path.join('uploads', pdf_file.filename)
            pdf_file.save(pdf_path)
            
            # Upload ke GCS
            bucket_name = 'escore-storage'
            destination_blob_name = f'essays/{timestamp}-{pdf_file.filename}'
            gcs_link = upload_to_gcs(bucket_name, pdf_file, destination_blob_name)
            
            essay_text = extract_text_from_pdf(pdf_path)

        # Preprocessing teks essay
        processed_input = preprocess_essay(essay_text)

        # Prediction
        prediction = model.predict(processed_input)
        raw_score = float(prediction[0][0])
        
        if raw_score < 1:
            rounded_score = 1
        else:
            rounded_score = int(np.round(raw_score))
        
        format_score = f'{rounded_score}/6'
        suggestion = predict_suggestion(rounded_score)

        response_data = {
            'title': essay_title,
            'essay': essay_text,
            'predicted_result': {
                'raw_score': raw_score,
                'score': format_score,
                'suggestion': suggestion
            },
            'createdAt': timestamp,
        }
        
        if gcs_link:
            response_data['gcsLink'] = gcs_link
        
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Save predict result
        user_ref = db.collection('users').document(user_doc[0].reference.id)
        user_ref.collection('predictions').add(response_data)
        
        return jsonify({
            'error': False,
            'message': 'Essay has been predicted!',
            'result': response_data
        }), 200
    except Exception as e:
        print('Error occurred:', str(e))
        return jsonify({
            'error': True,
            'message': 'An error occurred while processing the request!',
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', 'localhost')
    http_server = WSGIServer((host, port), app)
    print(f'Server is Ready on {host}:{port}')
    http_server.serve_forever()
