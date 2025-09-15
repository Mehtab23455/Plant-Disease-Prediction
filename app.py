import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# Define Flask app
app = Flask(__name__)

# --- Rebuild the DenseNet model in code ---
from densenet_model import build_densenet  # Make sure your build_densenet() function is in model.py

# Initialize model architecture
model = build_densenet(num_classes=15)

# Load the weights only
# model.load_weights("PlantDNet.h5")  # original trained weights
model.summary()


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # normalize
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    f = request.files['file']

    # Save uploaded file
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Predict
    preds = model_predict(file_path, model)
    disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                     'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                     'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                     'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                     'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

    ind = np.argmax(preds[0])
    result = disease_class[ind]
    return result


if __name__ == '__main__':
    app.run(debug=True)
