import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# Añadir directorio del modelo al path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model.predict import predict_plant_disease

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Diccionario de clases
CLASS_INDICES = {
    'Blight': 0,
    'Curl': 1,
    'Green Mite': 2,
    'Healthy': 3,
    'Leaf Miner': 4,
    'Leaf Spot': 5,
    'Mosaic': 6,
    'Powdery': 7,
    'Rust': 8,
    'Streak Virus': 9
}

# Verificar extensión de archivo permitida
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rutas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analizar')
def analizar():
    return render_template('analizar.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si la solicitud tiene el archivo
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No se ha proporcionado ningún archivo'
        })
    
    file = request.files['file']
    
    # Si el usuario no selecciona un archivo
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No se ha seleccionado ningún archivo'
        })
    
    if file and allowed_file(file.filename):
        # Guardar archivo
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Realizar predicción
            result = predict_plant_disease(file_path, CLASS_INDICES, visualize=False)
            
            # Eliminar archivo después de procesarlo (opcional)
            # os.remove(file_path)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error al procesar la imagen: {str(e)}'
            })
    
    return jsonify({
        'success': False,
        'error': 'Tipo de archivo no permitido. Por favor, sube una imagen (PNG, JPG, JPEG).'
    })

# Manejador de errores para archivos demasiado grandes
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'El archivo es demasiado grande. El tamaño máximo permitido es 16MB.'
    }), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)