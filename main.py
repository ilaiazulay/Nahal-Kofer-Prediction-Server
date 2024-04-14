from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract features in the order the model expects
        features = [data['water_level_distance'], data['water_opacity'], data['precipitation']]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})  # Return the prediction as int (0 or 1)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=8000)
