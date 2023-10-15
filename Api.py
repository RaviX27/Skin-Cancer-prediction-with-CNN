from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

model = load_model('skin_cancer_model.h5')

app = Flask(__name__)

# Function to make predictions for a single image and return the class label with the highest probability
def predict_skin_cancer(image_data):
    try:
        # Read image data as bytes
        img = image.load_img(io.BytesIO(image_data), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale pixel values to 0-1
        predictions = model.predict(img_array)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[predicted_class_index]

        class_labels = ['benign', 'malignant']
        predicted_class = class_labels[predicted_class_index]

        return predicted_class, float(predicted_probability)
    except Exception as e:
        # Handle any errors that occur during prediction
        return "Error", str(e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image'].read()
        # Get prediction
        predicted_class, predicted_probability = predict_skin_cancer(file)
        # Return the prediction as JSON response
        response = {
            'class': predicted_class,
            'probability': predicted_probability
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)