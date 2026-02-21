import streamlit as st
import joblib
import os
import re
import easyocr
import numpy as np
import whisper # Reemplazamos speech_recognition por Whisper
import tempfile # Necesario para procesar el audio temporalmente
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. CONFIGURACIÓN DE SEGURIDAD Y RECURSOS ---
# Cargamos las variables de entorno (tu API key de Gemini)
load_dotenv()
api_key = os.getenv("API_KEY")

# Validación de seguridad: detener la app si no hay clave
if not api_key:
    st.error("Error: No se encontró la API_KEY en el archivo .env")
    st.stop()

# Configuración de Gemini
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-pro')

# Usamos st.cache_resource para cargar los modelos pesados solo una vez.
# Esto evita que Streamlit los recargue cada vez que el usuario hace un clic.
@st.cache_resource
def cargar_recursos():
    # 1. Cargar el modelo de Machine Learning (Scikit-Learn)
    try:
        modelo_ml = joblib.load('modelo_libros.pkl')
    except Exception as e:
        modelo_ml = None
        
    # 2. Cargar el modelo OCR (EasyOCR)
    lector_ocr = easyocr.Reader(['es'], gpu=False) 
    
    # 3. Cargar el modelo de Transcripción (Whisper - Modelo 'base' para que sea rápido)
    modelo_audio = whisper.load_model("base")
    
    return modelo_ml, lector_ocr, modelo_audio

# Instanciamos los recursos
modelo_local, reader, whisper_model = cargar_recursos()

# Advertencia si falta el modelo local
if modelo_local is None:
    st.warning("⚠️ No se pudo cargar 'modelo_libros.pkl'. Asegúrate de subirlo a tu repositorio de GitHub.")

# --- 2. FUNCIONES DE APOYO ---
def es_entrada_valida(texto):
    """Valida que el texto no esté vacío y contenga letras para evitar errores en la API."""
    if not texto or len(texto.strip()) < 3:
        return False, "La entrada es demasiado corta. Escribe un poco más."
    if not re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', texto):
        return False, "Entrada no válida: Por favor usa palabras, no solo números o símbolos."
    return True, ""

def procesar_solicitud(texto_entrada):
    """Clasifica el texto con el modelo local y genera recomendaciones con Gemini."""
    valido, mensaje_error = es_entrada_valida(texto_entrada)
    if not valido:
        return None, mensaje_error
    
    # 1. Clasificación local (ML)
    categoria = "Desconocido"
    if modelo_local is not None:
        try:
            categoria = modelo_local.predict([texto_entrada])[0]
        except Exception as e:
            categoria = "Error en predicción"

    # 2. Generación con Gemini
    prompt = (
        f"El usuario busca libros basados en esta descripción: '{texto_entrada}'. "
        f"El sistema de Machine Learning ha detectado el género: {categoria}. "
        f"Actúa como un bibliotecario experto y recomienda 3 libros específicos (con autor) "
        f"que encajen perfectamente. Incluye una breve y atractiva frase de por qué leer cada uno."
    )
    
    try:
        response = model_gemini.generate_content(prompt)
        if response and response.text:
            return categoria, response.text
        else:
            return categoria, "Gemini no devolvió una respuesta válida."
    except Exception as e:
        return categoria, f"Error al conectar con Gemini: {str(e)}"

# --- 3. INTERFAZ DE USUARIO (STREAMLIT) ---
st.set_page_config(page_title="Biblioteca Inteligente", page_icon="📚", layout="centered")
st.title("📚 Mi Biblioteca Virtual Inteligente")
st.markdown("Clasificación mediante **Machine Learning local**, OCR, Whisper y recomendaciones de **Gemini 1.5 Flash**.")
st.markdown("---")

# Creación de pestañas para las distintas funcionalidades
tab_txt, tab_img, tab_aud = st.tabs(["✍️ Texto", "📷 Imagen (OCR)", "🎙️ Audio (Whisper)"])

# --- PESTAÑA 1: TEXTO ---
with tab_txt:
    st.subheader("Búsqueda por descripción")
    user_input = st.text_area("¿Qué te apetece leer hoy?", placeholder="Ej: Me gustan las historias de crímenes en la época victoriana...")
    
    if st.button("Analizar y Recomendar", key="btn_texto"):
        with st.spinner("Analizando tu petición..."):
            cat, resultado = procesar_solicitud(user_input)
            if cat:
                st.success(f"🎭 Género detectado por el modelo: **{cat}**")
                st.markdown(resultado)
            else:
                st.warning(resultado)

# --- PESTAÑA 2: IMAGEN (OCR) ---
with tab_img:
    st.subheader("Extraer texto de una contraportada o sinopsis")
    archivo_img = st.file_uploader("Sube una foto", type=['jpg', 'jpeg', 'png'])
    
    if archivo_img:
        # Mostrar la imagen
        img_pil = Image.open(archivo_img)
        st.image(img_pil, caption="Imagen cargada", use_container_width=True)
        # Convertir a numpy array para EasyOCR
        img_array = np.array(img_pil) 
        
        if st.button("Escanear Imagen y Recomendar", key="btn_img"):
            with st.spinner("Leyendo texto de la imagen..."):
                try:
                    # detail=0 devuelve solo una lista de textos
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
                    st.error(f"Error en el procesamiento OCR: {e}")

# --- PESTAÑA 3: AUDIO (WHISPER) ---
with tab_aud:
    st.subheader("Recomendación por voz")
    archivo_audio = st.file_uploader("Sube un archivo de audio (.wav, .mp3)", type=['wav', 'mp3', 'm4a'])
    
    if archivo_audio:
        st.audio(archivo_audio)
        if st.button("Transcribir y Analizar", key="btn_aud"):
            with st.spinner("Transcribiendo el audio con Whisper..."):
                # Whisper requiere un archivo físico en disco, así que creamos uno temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    tmp_audio.write(archivo_audio.read())
                    tmp_audio_path = tmp_audio.name
                
                try:
                    # Transcribir usando el modelo cargado en caché
                    resultado_whisper = whisper_model.transcribe(tmp_audio_path, language="es")
                    texto_voz = resultado_whisper["text"]
                    
                    st.info(f"**Transcripción:** {texto_voz}")
                    
                    # Pasar el texto transcrito al pipeline de ML + Gemini
                    cat, resultado = procesar_solicitud(texto_voz)
                    if cat:
                        st.success(f"🎭 Género detectado: **{cat}**")
                        st.markdown(resultado)
                        
                except Exception as e:
                    st.error(f"Error al procesar el audio: {e}")
                finally:
                    # Limpieza: eliminar el archivo temporal del servidor
                    if os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)

st.markdown("---")
st.caption("Desarrollado con ❤️ usando Streamlit, Scikit-Learn, Whisper, EasyOCR y Gemini API")