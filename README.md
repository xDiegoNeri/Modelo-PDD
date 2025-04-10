# Sistema de Detección de Enfermedades en Plantas 🌱

Este proyecto implementa un sistema de detección de enfermedades en plantas utilizando técnicas de aprendizaje profundo.

## 📋 Descripción

El sistema analiza imágenes de plantas para identificar posibles enfermedades utilizando un modelo de deep learning entrenado con un amplio dataset de imágenes de plantas con diferentes condiciones patológicas.

## 🗂️ Estructura del Proyecto

```
├── README.md                 # Documentación del proyecto
├── requirements.txt          # Dependencias de Python
├── model/                    # Código para el modelo de aprendizaje profundo
│   ├── train.py              # Script para entrenar el modelo
│   ├── predict.py            # Script para realizar predicciones
│   └── utils.py              # Funciones de utilidad para el modelo
├── web/                      # Aplicación web
│   ├── app.py                # Backend de Flask
│   ├── static/               # Archivos estáticos
│   │   ├── css/              # Estilos CSS
│   │   ├── js/               # Scripts JavaScript
│   │   └── img/              # Imágenes
│   └── templates/            # Plantillas HTML
└── .gitignore                # Archivos y carpetas ignorados por Git
```

## 🔍 Dataset

El dataset utilizado para entrenar el modelo está disponible en Kaggle:

[Plant Disease Data](https://www.kaggle.com/datasets/ddubs420/plant-disease-data)

> **Nota**: La carpeta de datos no está incluida en este repositorio. Debe descargarse por separado desde el enlace proporcionado.

## 🛠️ Requisitos

- Python 3.8+
- TensorFlow 2.x
- Keras
- Flask
- NumPy
- Pandas
- Pillow
- scikit-learn

## 🚀 Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/sistema-deteccion-enfermedades-plantas.git
   cd sistema-deteccion-enfermedades-plantas
   ```

2. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Descargar el dataset desde Kaggle y colocarlo en la carpeta `Plant Disease Data`

## 💻 Uso

1. Ejecutar la aplicación web:
   ```bash
   python run.py --all #Entrenar el modelo y luego iniciar la aplicación web
   python run.py --web #Iniciar la aplicación web
   python run.py --train #Entrenar el modelo
   ```

2. Abrir el navegador en `http://localhost:5000`

![image](https://github.com/user-attachments/assets/4442f1d1-1911-4316-939c-f3fb1b998011)
![image](https://github.com/user-attachments/assets/059dde92-fe95-4f2f-aa0e-70c73ca4cacb)

## 🌿 Clases de Enfermedades

El sistema puede detectar las siguientes condiciones en plantas:

- Blight (Tizón)
- Curl (Enrollamiento)
- Green Mite (Ácaro Verde)
- Healthy (Saludable)
- Leaf Miner (Minador de Hojas)
- Leaf Spot (Mancha Foliar)
- Mosaic (Mosaico)
- Powdery (Oídio)
- Rust (Roya)
- Streak Virus (Virus del Rayado)

![image](https://github.com/user-attachments/assets/c13a4c2d-79bb-42de-9246-538e7f1257b0)



## 📝 Licencia

Este proyecto ya no está bajo la Licencia MIT :(
