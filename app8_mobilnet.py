from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle
model = load_model('model_mobilenet.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    
    # Prétraitement de l'image
    img = image.load_img(file, target_size=(256, 256))  # Changez selon la taille d'entrée du modèle
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch

    # Prédire
    predictions = model.predict(img_array)
    predicted_mask = np.argmax(predictions, axis=-1)  # Obtenir les classes prédites
    predicted_mask = predicted_mask.squeeze()  # Retirer la dimension supplémentaire

    return jsonify({'mask': predicted_mask.tolist()})  # Convertir en liste pour JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5006)))
