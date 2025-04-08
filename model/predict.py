import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# Ruta al modelo guardado
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'plant_disease_model_final_finetuned.h5')

# Tamaño de imagen esperado por el modelo
IMAGE_SIZE = (224, 224)

# Diccionario de clases en español
CLASS_NAMES_ES = {
    'Blight': 'Tizón',
    'Curl': 'Enrollamiento',
    'Green Mite': 'Ácaro Verde',
    'Healthy': 'Saludable',
    'Leaf Miner': 'Minador de Hojas',
    'Leaf Spot': 'Mancha Foliar',
    'Mosaic': 'Mosaico',
    'Powdery': 'Oídio',
    'Rust': 'Roya',
    'Streak Virus': 'Virus del Rayado'
}

# Cargar el modelo entrenado
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        print(f"Modelo cargado desde: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Preprocesar imagen para predicción
def preprocess_image(img_path):
    try:
        # Cargar imagen y redimensionar
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        
        # Convertir a array y expandir dimensiones
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocesar para EfficientNet
        img_array = preprocess_input(img_array)
        
        return img_array, img
    except Exception as e:
        print(f"Error al preprocesar la imagen: {e}")
        return None, None

# Realizar predicción
def predict_disease(model, img_path, class_indices):
    # Preprocesar imagen
    img_array, original_img = preprocess_image(img_path)
    
    if img_array is None:
        return None, None, None
    
    # Realizar predicción
    predictions = model.predict(img_array)
    
    # Obtener la clase con mayor probabilidad
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100
    
    # Mapear índice a nombre de clase
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class = class_names[predicted_class_index]
    
    # Traducir al español si está disponible
    predicted_class_es = CLASS_NAMES_ES.get(predicted_class, predicted_class)
    
    return predicted_class, predicted_class_es, confidence, original_img, predictions[0]

# Visualizar resultados
def visualize_prediction(img, class_name_es, confidence, all_predictions=None, class_indices=None):
    plt.figure(figsize=(12, 6))
    
    # Mostrar imagen
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicción: {class_name_es}")
    plt.axis('off')
    
    # Mostrar confianza
    if all_predictions is not None and class_indices is not None:
        # Invertir el diccionario de índices de clase
        class_names = {v: k for k, v in class_indices.items()}
        
        # Obtener las 5 principales predicciones
        top_indices = np.argsort(all_predictions)[-5:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_classes_es = [CLASS_NAMES_ES.get(cls, cls) for cls in top_classes]
        top_confidences = [all_predictions[i] * 100 for i in top_indices]
        
        # Mostrar gráfico de barras
        plt.subplot(1, 2, 2)
        bars = plt.barh(top_classes_es, top_confidences, color='lightgreen')
        plt.xlabel('Confianza (%)')
        plt.title('Top 5 Predicciones')
        plt.xlim(0, 100)
        
        # Añadir etiquetas de porcentaje
        for bar, conf in zip(bars, top_confidences):
            plt.text(min(conf + 3, 95), bar.get_y() + bar.get_height()/2, 
                    f"{conf:.1f}%", va='center')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.close()

# Función principal para predicción
def predict_plant_disease(img_path, class_indices, visualize=True):
    # Cargar modelo
    model = load_trained_model()
    
    if model is None:
        return {
            'success': False,
            'error': 'No se pudo cargar el modelo'
        }
    
    # Realizar predicción
    try:
        predicted_class, predicted_class_es, confidence, img, all_predictions = predict_disease(model, img_path, class_indices)
        
        if predicted_class is None:
            return {
                'success': False,
                'error': 'Error al procesar la imagen'
            }
        
        # Visualizar si se solicita
        if visualize:
            visualize_prediction(img, predicted_class_es, confidence, all_predictions, class_indices)
        
        # Preparar resultados
        result = {
            'success': True,
            'class': predicted_class,
            'class_es': predicted_class_es,
            'confidence': float(confidence),
            'top_predictions': []
        }
        
        # Añadir top 5 predicciones
        class_names = {v: k for k, v in class_indices.items()}
        top_indices = np.argsort(all_predictions)[-5:][::-1]
        
        for idx in top_indices:
            class_name = class_names[idx]
            class_name_es = CLASS_NAMES_ES.get(class_name, class_name)
            result['top_predictions'].append({
                'class': class_name,
                'class_es': class_name_es,
                'confidence': float(all_predictions[idx] * 100)
            })
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Este código se ejecuta solo si se llama directamente al script
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Ejemplo de diccionario de índices de clase (debe ser reemplazado por el real)
        # En una aplicación real, esto se cargaría desde el modelo entrenado
        sample_class_indices = {
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
        
        result = predict_plant_disease(image_path, sample_class_indices)
        print(result)
    else:
        print("Por favor proporciona la ruta a una imagen de planta.")
        print("Uso: python predict.py ruta/a/imagen.jpg")