# Clasificación de sentimientos con una red LSTM

## 1. Descripción del proyecto

Este proyecto utiliza una red neuronal recurrente de tipo LSTM para clasificar reseñas de películas en dos categorías: sentimiento positivo y sentimiento negativo.

El proyecto incluye exploración de datos, preprocesamiento, entrenamiento, validación, evaluación y una aplicación web desarrollada con Streamlit.

## 2. Dataset

Se utiliza el conjunto IMDb de reseñas de películas integrado en TensorFlow/Keras. Las reseñas están etiquetadas como positivas o negativas.

## 3. Evaluación

El modelo se evalúa con un conjunto de prueba independiente. Se calculan accuracy, precision, recall y F1-score, y se genera una matriz de confusión.

Los resultados obtenidos se encuentran en la carpeta `results/`.

## 4. Estructura del repositorio

- `notebooks/`: notebook de experimentación.
- `models/`: modelo entrenado y vocabulario.
- `results/`: gráficas, matriz de confusión y predicciones.
- `app.py`: aplicación web.
- `requirements.txt`: dependencias.
- `README.md`: documentación.

## 5. Instalación y ejecución

Crear un entorno virtual e instalar las dependencias:
1. python -m venv .venv
2. pip install -r requirements.txt

En Windows, activar el entorno virtual con:
3. .venv\Scripts\activate

Ejecutar la aplicación:
4. python -m streamlit run app.py

La aplicación permite introducir una reseña en inglés y obtener una predicción de sentimiento.

## 6. Limitaciones

El modelo se entrena con reseñas en inglés. Por tanto, no se garantiza un rendimiento adecuado para textos en español, sarcasmo u opiniones ambiguas.