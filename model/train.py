import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configuración
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 5
LEARNING_RATE = 0.001

# Rutas de datos
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Plant Disease Data')
TRAIN_DIR = os.path.join(BASE_DIR, 'Plant Disease Data - Copy - Copy (2)', 'Plant Disease Data - Copy - Copy', 'Train')
VAL_DIR = os.path.join(BASE_DIR, 'Plant Disease Data - Copy - Copy (2)', 'Plant Disease Data - Copy - Copy', 'Validation')

# Verificar rutas
print(f"Directorio de entrenamiento: {TRAIN_DIR}")
print(f"Directorio de validación: {VAL_DIR}")

# Crear generadores de datos con aumento de datos
def create_data_generators():
    # Aumento de datos para el conjunto de entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Solo reescalado para validación
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Cargar imágenes
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, val_generator

# Crear modelo con transfer learning
def create_model(num_classes):
    # Cargar modelo base pre-entrenado (sin incluir la capa de clasificación)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Congelar las capas del modelo base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Crear nuevo modelo
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Función para entrenar el modelo
def train_model():
    # Crear generadores de datos
    train_generator, val_generator = create_data_generators()
    
    # Obtener número de clases
    num_classes = len(train_generator.class_indices)
    print(f"Número de clases: {num_classes}")
    print(f"Clases: {train_generator.class_indices}")
    
    # Crear modelo
    model = create_model(num_classes)
    
    # Crear directorio para guardar modelos si no existe
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models'), exist_ok=True)
    
    # Callbacks para mejorar el entrenamiento
    callbacks = [
        ModelCheckpoint(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'plant_disease_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Entrenar modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Guardar modelo final
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'plant_disease_model_final.h5'))
    
    # Guardar historial de entrenamiento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del Modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del Modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_history.png'))
    
    return model, history

# Función para descongelar capas y realizar fine-tuning
def fine_tune_model(model, train_generator, val_generator):
    # Descongelar algunas capas del modelo base
    for layer in model.layers[0].layers[-20:]:  # Descongelar las últimas 20 capas
        layer.trainable = True
    
    # Recompilar modelo con una tasa de aprendizaje más baja
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks para fine-tuning
    callbacks = [
        ModelCheckpoint(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'plant_disease_model_finetuned.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Realizar fine-tuning
    history_ft = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=10,  # Menos épocas para fine-tuning
        callbacks=callbacks
    )
    
    # Guardar modelo final con fine-tuning
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'plant_disease_model_final_finetuned.h5'))
    
    return model, history_ft

# Función principal
def main():
    print("Iniciando entrenamiento del modelo de detección de enfermedades en plantas...")
    
    # Entrenar modelo inicial
    model, history = train_model()
    
    # Crear generadores de datos para fine-tuning
    train_generator, val_generator = create_data_generators()
    
    # Realizar fine-tuning
    print("\nIniciando fine-tuning del modelo...")
    model, history_ft = fine_tune_model(model, train_generator, val_generator)
    
    print("\nEntrenamiento completado. Modelo guardado en 'model/saved_models/'")

if __name__ == "__main__":
    main()