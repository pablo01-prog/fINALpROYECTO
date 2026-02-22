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
import requests

# --- 1. CONFIGURACIÓN DE SEGURIDAD Y RECURSOS ---
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    st.error("Error: No se encontró la API_KEY en el archivo .env")
    st.stop()

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

if not deepgram_api_key:
    st.warning("Aviso: No se encontró la DEEPGRAM_API_KEY en el archivo .env. El audio no funcionará.")


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
    # 1. Ampliamos los formatos permitidos
    archivo_audio = st.file_uploader("Sube un archivo de audio (.wav, .mp3, .m4a)", type=['wav', 'mp3', 'm4a'])
    
    if archivo_audio:
        st.audio(archivo_audio)
        if st.button("Transcribir y Analizar", key="btn_aud"):
            if not deepgram_api_key:
                st.error("Falta la API Key de Deepgram para poder transcribir.")
            else:
                with st.spinner("Enviando audio a Deepgram para transcribir..."):
                    try:
                        # 2. Averiguamos la extensión para decirle a Deepgram qué tipo de archivo es
                        ext = archivo_audio.name.split('.')[-1].lower()
                        if ext == "wav":
                            content_type = "audio/wav"
                        elif ext == "mp3":
                            content_type = "audio/mpeg"
                        elif ext in ["m4a", "mp4"]:
                            content_type = "audio/mp4"
                        else:
                            content_type = "application/octet-stream"

                        # 3. Preparamos la petición a Deepgram
                        headers = {
                            "Authorization": f"Token {deepgram_api_key}",
                            "Content-Type": content_type,
                        }
                        params = {
                            "model": "nova-3",
                            "language": "es",
                            "smart_format": "true",
                        }

                        # Leemos los bytes directamente del archivo subido
                        audio_bytes = archivo_audio.read()

                        # Enviamos la petición
                        response = requests.post(
                            "https://api.deepgram.com/v1/listen",
                            headers=headers,
                            params=params,
                            data=audio_bytes,
                            timeout=60,
                        )

                        # 4. Procesamos la respuesta
                        if response.status_code == 200:
                            data = response.json()
                            texto_voz = data["results"]["channels"][0]["alternatives"][0]["transcript"]
                            
                            if texto_voz.strip():
                                st.info(f"**Transcripción:** {texto_voz}")
                                
                                # Le pasamos el texto a Gemini/Modelo Local
                                cat, resultado = procesar_solicitud(texto_voz)
                                if cat:
                                    st.success(f"🎭 Género detectado: **{cat}**")
                                    st.markdown(resultado)
                            else:
                                st.warning("El audio se procesó, pero no se detectó ninguna voz.")
                        else:
                            st.error(f"Error en la API de Deepgram (HTTP {response.status_code}): {response.text}")

                    except Exception as e:
                        st.error(f"Error inesperado al procesar el audio: {e}")

