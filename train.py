import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# 1. Dataset AMPLIADO para mayor precisión
data = {
    'texto' : [
        # Fantasía
        'magia dragones espada guerrero aventura hechizo varita elfo enano mundo magico',
        'una historia de magos y dragones con espadas legendarias y mucha aventura',
        'un reino de fantasia donde los elfos y enanos luchan con magia',
        # Policial / Detectives
        'crimen detective asesinato misterio policia huellas culpable investigacion forense',
        'un detective busca al asesino en un misterio policial lleno de intriga',
        'novela negra sobre un inspector que investiga un crimen sin resolver',
        'el detective privado descubre al culpable gracias a las pistas',
        # Romance
        'amor romance pareja enamorados boda pasion corazon novios cita romantica',
        'historia de amor sobre una pareja de enamorados que planean su boda',
        'un drama romantico sobre dos amantes cruzados por el destino',
        # Ciencia Ficción
        'futuro naves espaciales robots planetas galaxia tecnologia alienigenas cosmos',
        'viaje al futuro en naves espaciales con robots inteligentes y otros planetas',
        'ciencia ficcion sobre inteligencia artificial y viajes intergalacticos',
        # Terror
        'fantasmas terror miedo susto sangre oscuro pesadilla monstruo espiritu grito',
        'un relato de terror con fantasmas y monstruos en un ambiente oscuro y de miedo',
        'una casa encantada llena de espiritus que te haran gritar de terror',
        # Histórica
        'historia antigua guerra reyes imperio epoca medieval caballero batalla siglo',
        'narración sobre la historia antigua con reyes y batallas de un imperio caido',
        'ficcion historica ambientada en la segunda guerra mundial'
    ],
    'genero': [
        'Fantasia', 'Fantasia', 'Fantasia',
        'Policial', 'Policial', 'Policial', 'Policial',
        'Romance', 'Romance', 'Romance',
        'Ciencia Ficcion', 'Ciencia Ficcion', 'Ciencia Ficcion',
        'Terror', 'Terror', 'Terror',
        'Historica', 'Historica', 'Historica'
    ]
}

df = pd.DataFrame(data)

modelo = make_pipeline(
    TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2)), 
    MultinomialNB()
)

print("Entrenando el modelo de predicción mejorado...")
modelo.fit(df['texto'], df['genero'])

joblib.dump(modelo, 'modelo_libros.pkl')
print("✅ Nuevo 'modelo_libros.pkl' generado con éxito. ¡Súbelo a GitHub!")