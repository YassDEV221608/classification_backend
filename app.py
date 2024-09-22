from flask import Flask, request, jsonify
from keras.api.models import load_model
from keras.api.preprocessing import image
import numpy as np
from flask_cors import CORS
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your saved model
model = load_model("model.h5")

# Class names for prediction
class_names = {0: "buildings", 1: "forest", 2: "glacier", 3: "mountain", 4: "sea", 5: "street"}  # Adjust as necessary

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    # Convert FileStorage to BytesIO
    img_bytes = io.BytesIO(file.read())

    # Preprocess the image
    img = image.load_img(img_bytes, target_size=(150, 150))  # Adjust size to match your model input
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Scaling the image (since the model was trained with rescaling)
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    response = {
        'prediction': class_names[predicted_class],
        'confidence': float(np.max(predictions))
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)