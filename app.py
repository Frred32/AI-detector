from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and vectorizer once at the start
try:
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        loaded_vectorizer = pickle.load(vec_file)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure that 'model.pkl' and 'vectorizer.pkl' exist.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the model or vectorizer: {e}")
    exit(1)

# Function to predict whether text is AI or human
def predict(text):
    # Transform the input text
    text_vec = loaded_vectorizer.transform([text])

    # Get prediction probabilities
    probabilities = loaded_model.predict_proba(text_vec)[0]

    # Extract probabilities for AI and human classes
    human_probability = probabilities[1] * 100  # Probability of Human
    ai_probability = probabilities[0] * 100      # Probability of AI

    return ai_probability, human_probability

# Route to handle prediction API
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    
    # Check if the 'text' key is in the request data
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    ai_prob, human_prob = predict(text)

    response = {
        "Human Probability": f"{human_prob:.2f}%",
        "AI Probability": f"{ai_prob:.2f}%"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
