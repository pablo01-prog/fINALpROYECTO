
import streamlit as st
import joblib
import os
import re
import easyocr
import numpy as np
import speech_recognition as sr # VOLVEMOS AL DE TU PROFESORA
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. CONFIGURACIÓN DE SEGURIDAD Y RECURSOS ---
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    st.error("Error: No se encontró la API_KEY en el archivo .env")
    st.stop()

genai.configure(api_key=api_key)
# Usamos gemini-pro para EVITAR el error 404 en Streamlit
model_gemini = genai.GenerativeModel('gemini-pro')

@st.cache_resource
def cargar_recursos():
    try:
        modelo_ml = joblib.load('modelo_libros.pkl')
    except Exception as e:
        modelo_ml = None
        
    lector_ocr = easyocr.Reader(['es'], gpu=False) 
    return modelo_ml, lector_ocr

modelo_local, reader = cargar_recursos()

if modelo_local is None:
    st.warning("⚠️ No se pudo cargar 'modelo_libros.pkl'.")

# --- 2. FUNCIONES DE APOYO ---
def es_entrada_valida(texto):
    if not texto or len(texto.strip()) < 3:
        return False, "La entrada es demasiado corta."
    if not re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', texto):
        return False, "Entrada no válida: Por favor usa palabras."
    return True, ""

def procesar_solicitud(texto_entrada):
    valido, mensaje_error = es_entrada_valida(texto_entrada)
    if not valido:
        return None, mensaje_error
    
    categoria = "Desconocido"
    if modelo_local is not None:
        try:
            categoria = modelo_local.predict([texto_entrada])[0]
        except:
            categoria = "Error en predicción"

    prompt = (
        f"El usuario busca libros basados en esta descripción: '{texto_entrada}'. "
        f"El sistema ha detectado el género: {categoria}. "
        f"Recomienda 3 libros específicos (con autor) de este género. "
        f"Incluye una breve frase de por qué leer cada uno."
    )
    
    try:
        response = model_gemini.generate_content(prompt)
        if response and response.text:
            return categoria, response.text
        else:
            return categoria, "Gemini no devolvió una respuesta válida."
    except Exception as e:
        return categoria, f"Error al conectar con Gemini: {str(e)}"

# --- 3. INTERFAZ DE USUARIO ---
st.set_page_config(page_title="Biblioteca Inteligente", page_icon="📚", layout="centered")
st.title("📚 Mi Biblioteca Virtual Inteligente")
st.markdown("---")

tab_txt, tab_img, tab_aud = st.tabs(["✍️ Texto", "📷 Imagen (OCR)", "🎙️ Audio"])

# --- PESTAÑA 1: TEXTO ---
with tab_txt:
    user_input = st.text_area("¿Qué te apetece leer hoy?", placeholder="Ej: Me gustan las historias de crímenes...")
    if st.button("Analizar y Recomendar", key="btn_texto"):
        with st.spinner("Analizando tu petición..."):
            cat, resultado = procesar_solicitud(user_input)
            if cat:
                st.success(f"🎭 Género detectado: **{cat}**")
                st.markdown(resultado)
            else:
                st.warning(resultado)

# --- PESTAÑA 2: IMAGEN (OCR) ---
with tab_img:
    archivo_img = st.file_uploader("Sube una foto", type=['jpg', 'jpeg', 'png'])
    if archivo_img:
        img_pil = Image.open(archivo_img)
        st.image(img_pil, caption="Imagen cargada", use_container_width=True)
        img_array = np.array(img_pil) 
        
        if st.button("Escanear Imagen y Recomendar", key="btn_img"):
            with st.spinner("Leyendo texto de la imagen..."):
                try:
                    resultado_ocr = reader.readtext(img_array, detail=0)
                    texto_extraido = " ".join(resultado_ocr)
                    if texto_extraido.strip():
                        st.info(f"**Texto detectado:** {texto_extraido}")
                        cat, resultado = procesar_solicitud(texto_extraido)
                        if cat:
                            st.success(f"🎭 Género detectado: **{cat}**")
                            st.markdown(resultado)
                    else:
                        st.error("No se detectó texto legible en la imagen.")
                except Exception as e:
                    st.error(f"Error en OCR: {e}")

# --- PESTAÑA 3: AUDIO ---
with tab_aud:
    archivo_audio = st.file_uploader("Sube un archivo .wav", type=['wav'])
    
    if archivo_audio:
        st.audio(archivo_audio)
        if st.button("Transcribir y Analizar", key="btn_aud"):
            with st.spinner("Escuchando el audio..."):
                r = sr.Recognizer()
                try:
                    # Usamos el modelo de tu profesora tal cual lo pediste
                    with sr.AudioFile(archivo_audio) as source:
                        audio_data = r.record(source)
                    
                    texto_voz = r.recognize_google(audio_data, language="es-ES")
                    st.info(f"**Transcripción:** {texto_voz}")
                    
                    cat, resultado = procesar_solicitud(texto_voz)
                    if cat:
                        st.success(f"🎭 Género detectado: **{cat}**")
                        st.markdown(resultado)
                        
                except sr.UnknownValueError:
                    st.error("No pude entender el audio. ¿Seguro que se escucha bien?")
                except Exception as e:
                    st.error(f"Error al procesar el audio: {e}")
