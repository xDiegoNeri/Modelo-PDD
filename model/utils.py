import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

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

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Carga y preprocesa una imagen para la predicción.
    
    Args:
        img_path: Ruta a la imagen
        target_size: Tamaño objetivo para redimensionar la imagen
        
    Returns:
        Imagen preprocesada como array de numpy
    """
    try:
        # Cargar imagen
        img = load_img(img_path, target_size=target_size)
        
        # Convertir a array
        img_array = img_to_array(img)
        
        # Expandir dimensiones para el batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocesar para EfficientNet
        img_array = preprocess_input(img_array)
        
        return img_array, img
    except Exception as e:
        print(f"Error al cargar la imagen {img_path}: {e}")
        return None, None

# Función para visualizar imágenes del conjunto de datos
def visualize_dataset_samples(data_dir, num_samples=5, classes=None):
    """Visualiza muestras aleatorias del conjunto de datos.
    
    Args:
        data_dir: Directorio que contiene las carpetas de clases
        num_samples: Número de muestras a visualizar por clase
        classes: Lista de clases a visualizar (si es None, se usan todas)
    """
    if classes is None:
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    plt.figure(figsize=(15, len(classes) * 2))
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Obtener archivos de imagen
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Seleccionar muestras aleatorias
        if len(image_files) > num_samples:
            image_files = np.random.choice(image_files, num_samples, replace=False)
        
        # Mostrar imágenes
        for j, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            img = load_img(img_path, target_size=(150, 150))
            
            plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            if j == 0:
                class_name_es = CLASS_NAMES_ES.get(class_name, class_name)
                plt.ylabel(class_name_es, fontsize=12)
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset_samples.png'))
    plt.close()

# Función para guardar el historial de entrenamiento
def save_training_history(history, save_path):
    """Guarda gráficos del historial de entrenamiento.
    
    Args:
        history: Objeto history devuelto por model.fit()
        save_path: Ruta donde guardar el gráfico
    """
    plt.figure(figsize=(12, 4))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del Modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del Modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Función para crear un TFRecord a partir de imágenes
def create_tfrecord(image_dir, output_path, image_size=(224, 224)):
    """Crea un archivo TFRecord a partir de un directorio de imágenes.
    
    Args:
        image_dir: Directorio que contiene las carpetas de clases
        output_path: Ruta donde guardar el archivo TFRecord
        image_size: Tamaño al que redimensionar las imágenes
    """
    # Función para convertir una imagen a un ejemplo de TF
    def _bytes_feature(value):
        """Devuelve un feature de bytes."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList no acepta tensores de tf
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Devuelve un feature de int64."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # Obtener clases
    classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Crear escritor de TFRecord
    with tf.io.TFRecordWriter(output_path) as writer:
        for class_name in classes:
            class_dir = os.path.join(image_dir, class_name)
            class_idx = class_to_idx[class_name]
            
            # Procesar cada imagen
            for img_file in os.listdir(class_dir):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # Cargar y preprocesar imagen
                    img = tf.io.read_file(img_path)
                    
                    # Crear ejemplo
                    feature = {
                        'image': _bytes_feature(img),
                        'label': _int64_feature(class_idx)
                    }
                    
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                except Exception as e:
                    print(f"Error al procesar {img_path}: {e}")
    
    print(f"TFRecord creado en {output_path}")
    print(f"Clases: {class_to_idx}")

# Función para evaluar el modelo en el conjunto de prueba
def evaluate_model(model, test_dir, batch_size=32, image_size=(224, 224)):
    """Evalúa el modelo en un conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        test_dir: Directorio con imágenes de prueba
        batch_size: Tamaño del batch
        image_size: Tamaño de las imágenes
        
    Returns:
        Diccionario con métricas de evaluación
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Crear generador de datos
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluar modelo
    results = model.evaluate(test_generator)
    
    # Obtener predicciones
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Obtener etiquetas reales
    y_true = test_generator.classes
    
    # Calcular matriz de confusión
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    
    # Generar informe de clasificación
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Añadir valores a la matriz
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'confusion_matrix.png'))
    plt.close()
    
    return {
        'loss': results[0],
        'accuracy': results[1],
        'confusion_matrix': cm,
        'classification_report': report
    }