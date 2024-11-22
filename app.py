from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to store uploaded images

# Load your model
model_id = '1vuHQcEXt1sY2dqbAjmNXI7465SRO4YZW'

model_path = 'crop_model.h5'

# Function to download the model from Google Drive
def download_model():
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        url = f'https://drive.google.com/uc?export=download&id={model_id}'
        gdown.download(url, model_path, quiet=False)

# Download the model when the app starts
download_model()

model = load_model(model_path)

class_names = ['corn_Blight', 'corn_Common_Rust', 'corn_Gray_Leaf_Spot',
               'maize_streak virus', 'rice_Bacterial leaf blight', 'rice_Brown spot',
               'rice_Leaf smut', 'sugarcane_Mosaic', 'sugarcane_RedRot',
               'sugarcane_Rust', 'sugarcane_Yellow', 'wheat_sepotoria', 'wheat_stripe_rust']


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)

        return render_template(
            'index.html',
            uploaded_file_url=file_path,
            predicted_class=predicted_class,
            confidence=confidence
        )


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
