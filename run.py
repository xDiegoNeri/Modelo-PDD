#!/usr/bin/env python
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Sistema de Detección de Enfermedades en Plantas')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--web', action='store_true', help='Iniciar la aplicación web')
    parser.add_argument('--all', action='store_true', help='Entrenar el modelo y luego iniciar la aplicación web')
    
    args = parser.parse_args()
    
    # Obtener directorio base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Si no se proporciona ningún argumento, mostrar ayuda
    if not (args.train or args.web or args.all):
        parser.print_help()
        return
    
    # Entrenar modelo
    if args.train or args.all:
        print("\n" + "=" * 50)
        print("Iniciando entrenamiento del modelo...")
        print("=" * 50)
        
        train_script = os.path.join(base_dir, 'model', 'train.py')
        try:
            subprocess.run([sys.executable, train_script], check=True)
            print("\n✅ Entrenamiento completado con éxito.")
        except subprocess.CalledProcessError:
            print("\n❌ Error durante el entrenamiento del modelo.")
            if not args.all:  # Si solo se pidió entrenar, salir con error
                return 1
    
    # Iniciar aplicación web
    if args.web or args.all:
        print("\n" + "=" * 50)
        print("Iniciando aplicación web...")
        print("=" * 50)
        
        web_app = os.path.join(base_dir, 'web', 'app.py')
        try:
            print("\nLa aplicación web estará disponible en: http://localhost:5000")
            print("Presiona Ctrl+C para detener el servidor.\n")
            subprocess.run([sys.executable, web_app], check=True)
        except KeyboardInterrupt:
            print("\n\n✅ Servidor web detenido.")
        except subprocess.CalledProcessError:
            print("\n❌ Error al iniciar la aplicación web.")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())