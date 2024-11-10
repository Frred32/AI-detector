from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Function to predict whether text is AI or human
def predict(text):
    try:
        # Load the model and vectorizer
        with open('model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vec_file:
            loaded_vectorizer = pickle.load(vec_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure that 'model.pkl' and 'vectorizer.pkl' exist.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the model or vectorizer: {e}")
        return None, None

    # Transform the input text
    text_vec = loaded_vectorizer.transform([text])

    # Get prediction probabilities
    probabilities = loaded_model.predict_proba(text_vec)[0]

    # Extract probabilities for AI and human classes
    human_probability = probabilities[1] * 100  # Probability of Human
    ai_probability = probabilities[0] * 100      # Probability of AI

    return ai_probability, human_probability

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    text = data['text']
    ai_prob, human_prob = predict(text)

    if ai_prob is not None and human_prob is not None:
        response = {
            "Human Probability": f"{human_prob:.2f}%",
            "AI Probability": f"{ai_prob:.2f}%"
        }
        return jsonify(response)
    else:
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=4000)
