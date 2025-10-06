from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import regex as re
import nltk

app = Flask(__name__, static_folder='public', static_url_path='')

# ---- Load Model and Preprocessing Function ----
pipeline = joblib.load('models/pipeline.joblib')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk_stops_words = nltk.corpus.stopwords.words('english')
lemma = nltk.WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    alpha_tokens = [w for w in tokens if w.isalpha()]
    no_stops = [word for word in alpha_tokens if word not in nltk_stops_words]
    lemmatized_tokens = [lemma.lemmatize(token) for token in no_stops]
    return ' '.join(lemmatized_tokens)

# ---- API Routes ----
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    full_text = " ".join([
        data.get('title', ''),
        data.get('location', ''),
        data.get('department', ''),
        data.get('company_profile', ''),
        data.get('description', ''),
        data.get('requirements', ''),
        data.get('benefits', ''),
        data.get('employment_type', ''),
        data.get('required_experience', ''),
        data.get('required_education', ''),
        data.get('industry', ''),
        data.get('function', '')
    ])

    if not full_text.strip():
        return jsonify({'error': 'No text provided for analysis.'}), 400

    processed_text = preprocess_text(full_text)

    prediction = pipeline.predict([processed_text])[0]
    probability = pipeline.predict_proba([processed_text])[0][1]

    result_text = "Fraudulent" if prediction == 1 else "Legitimate"

    return jsonify({
        'prediction': result_text,
        'confidence': f"{probability*100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)