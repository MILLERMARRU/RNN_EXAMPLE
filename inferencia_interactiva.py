"""
SCRIPT DE INFERENCIA INTERACTIVA
==================================
Usa el modelo entrenado para predecir la siguiente palabra en tiempo real.
"""

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar modelo y tokenizer
print("Cargando modelo y tokenizer...")
modelo = load_model('modelo_saludos_rnn.keras')
with open('tokenizer_saludos.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Obtener longitud máxima (debe coincidir con el entrenamiento)
longitud_maxima = modelo.input_shape[1] + 1

print("=" * 70)
print("PREDICTOR DE SIGUIENTE PALABRA - SALUDOS")
print("=" * 70)
print("Escribe una frase inicial y el modelo predecirá la siguiente palabra.")
print("Escribe 'salir' para terminar.")
print("=" * 70)

def predecir_siguiente_palabra(texto_entrada, top_n=5):
    """Predice las siguientes palabras más probables"""
    # Tokenizar el texto de entrada
    token_list = tokenizer.texts_to_sequences([texto_entrada])[0]
    
    # Aplicar padding
    token_list = pad_sequences([token_list], maxlen=longitud_maxima-1, padding='pre')
    
    # Predecir probabilidades
    probabilidades = modelo.predict(token_list, verbose=0)[0]
    
    # Obtener los índices de las palabras más probables
    indices_top = np.argsort(probabilidades)[-top_n:][::-1]
    
    # Crear diccionario inverso (índice -> palabra)
    indice_a_palabra = {indice: palabra for palabra, indice in tokenizer.word_index.items()}
    
    # Obtener palabras y probabilidades
    predicciones = []
    for idx in indices_top:
        if idx in indice_a_palabra:
            palabra = indice_a_palabra[idx]
            probabilidad = probabilidades[idx]
            predicciones.append((palabra, probabilidad))
    
    return predicciones

def generar_frase_completa(texto_semilla, max_palabras=10):
    """Genera una frase completa a partir de texto semilla"""
    texto_resultado = texto_semilla
    
    for _ in range(max_palabras):
        predicciones = predecir_siguiente_palabra(texto_resultado, top_n=1)
        
        if predicciones:
            siguiente_palabra = predicciones[0][0]
            texto_resultado += " " + siguiente_palabra
            
            # Evitar bucles infinitos
            palabras = texto_resultado.split()
            if len(palabras) > 2 and palabras[-1] == palabras[-2]:
                break
        else:
            break
    
    return texto_resultado

# Loop interactivo
while True:
    print("\n" + "-" * 70)
    texto = input("📝 Ingresa una frase: ").strip().lower()
    
    if texto == 'salir':
        print("¡Hasta luego!")
        break
    
    if not texto:
        continue
    
    print("\n🔍 TOP 5 PREDICCIONES:")
    predicciones = predecir_siguiente_palabra(texto, top_n=5)
    
    if predicciones:
        for i, (palabra, prob) in enumerate(predicciones, 1):
            print(f"   {i}. '{palabra}' → {prob*100:.2f}%")
    else:
        print("   No se encontraron predicciones.")
    
    print("\n💬 FRASE GENERADA:")
    frase_generada = generar_frase_completa(texto, max_palabras=5)
    print(f"   {frase_generada}")
