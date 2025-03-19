import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, url_for

# Initialize Flask app
app = Flask(__name__)

# Set up the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'model', 'plant_disease_prediction_model.h5')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices_path = os.path.join(working_dir, 'class_indices.json')
class_indices = json.load(open(class_indices_path))

# Complete disease data
disease_data = {
    'Potato Early Blight': {
        'causes': 'Caused by the fungus Alternaria solani. It thrives in warm, moist conditions.',
        'precautions': 'Use resistant varieties of potatoes, practice crop rotation, and apply fungicides. Ensure proper spacing of plants to allow for air circulation.',
        'cure': 'Apply fungicides such as chlorothalonil or copper-based products. Remove and destroy affected plant parts.',
        'more_info': 'More information about Potato Early Blight can be found [here](https://www.example.com/potato-early-blight).'
    },
    'Tomato Blight': {
        'causes': 'Caused by the fungus Phytophthora infestans. It spreads rapidly in cool, wet weather.',
        'precautions': 'Use resistant tomato varieties, avoid overhead watering, and ensure proper drainage. Practice crop rotation.',
        'cure': 'Apply fungicides like copper sulfate or mancozeb. Remove and destroy infected plants.',
        'more_info': 'More information about Tomato Blight can be found [here](https://www.example.com/tomato-blight).'
    },
    'Powdery Mildew': {
        'causes': 'Caused by various fungi in the order Erysiphales. It appears as white powdery spots on leaves and stems.',
        'precautions': 'Ensure proper air circulation around plants, avoid overhead watering, and use resistant varieties.',
        'cure': 'Apply fungicides such as sulfur or potassium bicarbonate. Remove and destroy affected plant parts.',
        'more_info': 'More information about Powdery Mildew can be found [here](https://www.example.com/powdery-mildew).'
    },
    'Downy Mildew': {
        'causes': 'Caused by the fungus-like oomycete Peronospora spp. It affects a wide range of plants and causes yellowing and wilting.',
        'precautions': 'Avoid overhead irrigation, space plants properly, and use resistant varieties.',
        'cure': 'Apply fungicides like metalaxyl or mefenoxam. Remove and destroy infected plants and debris.',
        'more_info': 'More information about Downy Mildew can be found [here](https://www.example.com/downy-mildew).'
    },
    'Leaf Spot': {
        'causes': 'Caused by various fungi or bacteria, resulting in dark, sunken spots on leaves.',
        'precautions': 'Avoid overhead watering, ensure proper drainage, and remove infected leaves.',
        'cure': 'Apply appropriate fungicides or bactericides based on the pathogen. Prune affected areas and improve air circulation.',
        'more_info': 'More information about Leaf Spot can be found [here](https://www.example.com/leaf-spot).'
    },
    'Rust': {
        'causes': 'Caused by rust fungi that form reddish-brown pustules on leaves and stems.',
        'precautions': 'Use resistant plant varieties and avoid planting in highly humid conditions.',
        'cure': 'Apply fungicides such as triazoles or strobilurins. Remove and destroy infected plant parts.',
        'more_info': 'More information about Rust can be found [here](https://www.example.com/rust).'
    },
    'Bacterial Wilt': {
        'causes': 'Caused by the bacterium Ralstonia solanacearum. It causes wilting and yellowing of leaves.',
        'precautions': 'Use resistant varieties and practice crop rotation. Avoid working in fields when plants are wet.',
        'cure': 'There are no effective chemical treatments. Remove and destroy infected plants, and practice good sanitation.',
        'more_info': 'More information about Bacterial Wilt can be found [here](https://www.example.com/bacterial-wilt).'
    },
    'Root Rot': {
        'causes': 'Caused by various fungi and soil-borne pathogens that infect the root system, leading to decay.',
        'precautions': 'Ensure well-drained soil and avoid overwatering. Use soil amendments to improve drainage.',
        'cure': 'Apply fungicides such as systemic or contact fungicides. Improve soil conditions and remove infected plants.',
        'more_info': 'More information about Root Rot can be found [here](https://www.example.com/root-rot).'
    },
    'Scab': {
        'causes': 'Caused by fungal pathogens like Cladosporium spp. It results in rough, raised lesions on fruits and leaves.',
        'precautions': 'Use resistant varieties and ensure proper plant spacing. Avoid working in wet conditions.',
        'cure': 'Apply fungicides such as copper-based products or mancozeb. Remove and destroy affected parts.',
        'more_info': 'More information about Scab can be found [here](https://www.example.com/scab).'
    },
    'Black Spot': {
        'causes': 'Caused by the fungus Diplocarpon rosae. It appears as black spots with fringed edges on rose leaves.',
        'precautions': 'Remove and destroy fallen leaves, avoid overhead watering, and use resistant varieties.',
        'cure': 'Apply fungicides like chlorothalonil or copper-based products. Prune affected areas and improve air circulation.',
        'more_info': 'More information about Black Spot can be found [here](https://www.example.com/black-spot).'
    }
}

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    # Replace underscores with spaces
    formatted_class_name = predicted_class_name.replace('_', ' ')
    return formatted_class_name

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    disease_info = {}

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            prediction = predict_image_class(model, file, class_indices)
            print(f"Predicted class: {prediction}")  # Debug: Print the prediction
            disease_info = disease_data.get(prediction, {})

    return render_template('index.html', prediction=prediction, disease_info=disease_info)

@app.route('/disease_info/<disease>/<info_type>', methods=['GET'])
def get_disease_info(disease, info_type):
    print(f"Fetching {info_type} for {disease}")  # Debug: Log what is being fetched
    info = disease_data.get(disease, {}).get(info_type, 'No information available.')
    return jsonify(info=info)

if __name__ == '__main__':
    app.run(debug=True)\\\1
