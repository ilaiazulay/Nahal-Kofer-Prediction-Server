from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import os
import json

app = Flask(__name__)

# Define the path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'flood_model_5.keras')

# Load the trained model and scaler
model = tf.keras.models.load_model(model_path)
with open('scaler_5.pkl', 'rb') as f:
    scaler = pickle.load(f)

SEQ_LENGTH = 5  # Updated sequence length
buffer_file = 'data_buffer.json'

# Function to read buffer from file
def read_buffer():
    if os.path.exists(buffer_file):
        with open(buffer_file, 'r') as f:
            return json.load(f)
    else:
        return []

# Function to write buffer to file
def write_buffer(buffer):
    with open(buffer_file, 'w') as f:
        json.dump(buffer, f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if isinstance(data, list):
            features = data[-1]  # Use the last element in the list if data is sent as a list
        else:
            features = [data['water_level_distance'], data['water_opacity'], data['precipitation']]

        # Scale the input features
        scaled_features = scaler.transform([features])

        # Read the current buffer
        data_buffer = read_buffer()
        data_buffer.append(scaled_features[0].tolist())

        # Debugging statement to monitor buffer
        print(f"Current buffer size: {len(data_buffer)}")
        print(f"Current buffer content: {data_buffer}")

        # Ensure the buffer only keeps the last SEQ_LENGTH data points
        if len(data_buffer) > SEQ_LENGTH:
            data_buffer.pop(0)

        # Write the updated buffer back to the file
        write_buffer(data_buffer)

        # Only make a prediction if we have SEQ_LENGTH data points
        if len(data_buffer) == SEQ_LENGTH:
            sequence = np.array(data_buffer).reshape(1, SEQ_LENGTH, len(features))

            # Make the prediction
            prediction = model.predict(sequence)
            return jsonify({'prediction': int(prediction[0][0] > 0.5)})  # Return the prediction as int (0 or 1)
        else:
            return jsonify({'error': f'Not enough data to make a prediction. Collecting more data points. Current buffer size: {len(data_buffer)}'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
