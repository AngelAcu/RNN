
import json
import re
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mejor_modelo.keras"
VOCAB_PATH = BASE_DIR / "models" / "word_index.json"

VOCAB_SIZE = 10000
MAX_LEN = 200

st.set_page_config(
    page_title="Analizador de sentimientos",
    page_icon="💬",
    layout="centered"
)


@st.cache_resource
def cargar_modelo():
    return keras.models.load_model(MODEL_PATH)


@st.cache_data
def cargar_vocabulario():
    with open(VOCAB_PATH, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def preparar_texto(texto, word_index):
    palabras = re.findall(r"[a-zA-Z0-9']+", texto.lower())
    secuencia = [1]  # Token de inicio de reseña

    for palabra in palabras:
        indice = word_index.get(palabra, 2) + 3 if palabra in word_index else 2

        if indice >= VOCAB_SIZE + 3:
            indice = 2

        secuencia.append(indice)

    return keras.utils.pad_sequences(
        [secuencia],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )


st.title("💬 Analizador de sentimientos")
st.write(
    "Escribe una reseña de película en inglés. "
    "La red neuronal LSTM estimará si el sentimiento es positivo o negativo."
)

try:
    modelo = cargar_modelo()
    word_index = cargar_vocabulario()

    texto = st.text_area(
        "Escribe tu reseña",
        placeholder="The movie was wonderful and I really enjoyed it."
    )

    if st.button("Analizar sentimiento", type="primary"):
        if not texto.strip():
            st.warning("Por favor, escribe una reseña.")
        else:
            entrada = preparar_texto(texto, word_index)
            probabilidad = float(modelo.predict(entrada, verbose=0)[0][0])

            if probabilidad >= 0.5:
                st.success("Sentimiento positivo 😊")
            else:
                st.error("Sentimiento negativo 😞")

            st.metric(
                "Probabilidad estimada de sentimiento positivo",
                f"{probabilidad * 100:.2f}%"
            )

            st.progress(probabilidad)

            st.caption(
                "La probabilidad es una estimación del modelo, "
                "no una garantía de que la clasificación sea correcta."
            )

except Exception as error:
    st.error(f"No se pudieron cargar los archivos del modelo: {error}")
    st.info(
        "Comprueba que existen models/mejor_modelo.keras "
        "y models/word_index.json."
    )
