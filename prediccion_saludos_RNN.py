"""
PREDICCIÓN DE SIGUIENTE PALABRA CON RNN
========================================
Este script entrena una RNN para predecir la siguiente palabra en frases de saludo.

Pasos principales:
a) Tokenización y padding de secuencias
b) Arquitectura: Embedding + SimpleRNN + Dense
c) Entrenamiento y evaluación
d) Inferencia con predicción de palabras
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Fijar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# ============================================================================
print("="*70)
print("1. CARGANDO Y PREPROCESANDO DATOS")
print("="*70)

# Cargar frases de saludo desde archivo
with open('saludos_dataset.txt', 'r', encoding='utf-8') as f:
    frases = f.read().splitlines()

print(f"Total de frases en el dataset: {len(frases)}")
print(f"Ejemplos de frases: {frases[:5]}")

# ============================================================================
# 2. TOKENIZACIÓN
# ============================================================================
print("\n" + "="*70)
print("2. TOKENIZACIÓN DE TEXTO")
print("="*70)

"""
TOKENIZACIÓN: Convierte palabras en números únicos.
- Cada palabra del vocabulario recibe un índice único
- El tokenizer aprende el vocabulario del corpus
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts(frases)
total_palabras = len(tokenizer.word_index) + 1  # +1 para el índice 0 (padding)

print(f"Vocabulario total: {total_palabras} palabras")
print(f"Primeras 20 palabras del diccionario:")
for palabra, indice in list(tokenizer.word_index.items())[:20]:
    print(f"  '{palabra}': {indice}")

# ============================================================================
# 3. CREACIÓN DE SECUENCIAS DE ENTRENAMIENTO
# ============================================================================
print("\n" + "="*70)
print("3. CREACIÓN DE SECUENCIAS DE ENTRENAMIENTO")
print("="*70)

"""
Para cada frase, creamos múltiples secuencias de entrenamiento.
Ejemplo: "hola buenos días"
  Secuencias:
  - [hola] -> buenos
  - [hola, buenos] -> días
  
Esto permite que el modelo aprenda a predecir la siguiente palabra
dada una secuencia de palabras anteriores.
"""

secuencias_entrada = []

for frase in frases:
    # Convertir frase a secuencia de tokens
    token_list = tokenizer.texts_to_sequences([frase])[0]
    
    # Crear múltiples secuencias de n-gramas
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        secuencias_entrada.append(n_gram_sequence)

print(f"Total de secuencias de entrenamiento generadas: {len(secuencias_entrada)}")
print(f"\nEjemplos de secuencias (primeras 10):")
for i, seq in enumerate(secuencias_entrada[:10]):
    palabras = [list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(idx)] for idx in seq]
    print(f"  {seq} -> {' '.join(palabras)}")

# ============================================================================
# 4. PADDING DE SECUENCIAS
# ============================================================================
print("\n" + "="*70)
print("4. PADDING DE SECUENCIAS")
print("="*70)

"""
PADDING: Todas las secuencias deben tener la misma longitud para entrenar.
- Se rellena con ceros (0) al inicio de las secuencias más cortas
- La longitud máxima es la de la secuencia más larga
"""

# Encontrar la longitud máxima
longitud_maxima = max([len(seq) for seq in secuencias_entrada])
secuencias_padded = pad_sequences(secuencias_entrada, maxlen=longitud_maxima, padding='pre')

print(f"Longitud máxima de secuencia: {longitud_maxima}")
print(f"Forma del array de secuencias: {secuencias_padded.shape}")
print(f"\nEjemplo de padding (primeras 5 secuencias):")
for i in range(5):
    print(f"  {secuencias_padded[i]}")

# ============================================================================
# 5. SEPARACIÓN EN X (entrada) e Y (salida)
# ============================================================================
print("\n" + "="*70)
print("5. PREPARACIÓN DE DATOS X (entrada) e Y (salida)")
print("="*70)

"""
X: Todas las palabras de la secuencia excepto la última
Y: La última palabra de la secuencia (objetivo a predecir)
"""

X = secuencias_padded[:, :-1]  # Todas las columnas excepto la última
y = secuencias_padded[:, -1]    # Solo la última columna

# Convertir Y a one-hot encoding (clasificación categórica)
y = to_categorical(y, num_classes=total_palabras)

print(f"Forma de X (entrada): {X.shape}")
print(f"Forma de Y (salida): {y.shape}")
print(f"\nEjemplo de entrada X[0]: {X[0]}")
print(f"Ejemplo de salida Y[0] (one-hot): {np.argmax(y[0])} (palabra: '{list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(np.argmax(y[0]))]}')")

# ============================================================================
# 6. ARQUITECTURA DEL MODELO
# ============================================================================
print("\n" + "="*70)
print("6. CONSTRUCCIÓN DEL MODELO RNN")
print("="*70)

"""
ARQUITECTURA:
1. Embedding Layer: Convierte índices de palabras en vectores densos
   - Dimensión del embedding: 100 (cada palabra se representa con 100 números)
   
2. SimpleRNN Layer: Procesa la secuencia
   - 150 unidades recurrentes
   - Captura patrones temporales en las secuencias
   
3. Dense Layer: Capa de salida
   - total_palabras neuronas (una por cada palabra del vocabulario)
   - Activación softmax para obtener probabilidades
"""

modelo = Sequential([
    Embedding(input_dim=total_palabras, 
              output_dim=100, 
              input_length=longitud_maxima-1),
    
    SimpleRNN(150, return_sequences=False),
    
    Dense(total_palabras, activation='softmax')
])

modelo.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(modelo.summary())

# ============================================================================
# 7. ENTRENAMIENTO DEL MODELO
# ============================================================================
print("\n" + "="*70)
print("7. ENTRENAMIENTO DEL MODELO")
print("="*70)

historia = modelo.fit(
    X, y,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ============================================================================
# 8. EVALUACIÓN DEL MODELO
# ============================================================================
print("\n" + "="*70)
print("8. EVALUACIÓN DEL DESEMPEÑO")
print("="*70)

# Evaluar en los datos de entrenamiento
perdida_final, precision_final = modelo.evaluate(X, y, verbose=0)
print(f"Pérdida final: {perdida_final:.4f}")
print(f"Precisión final: {precision_final:.4f}")

# Graficar evolución del entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historia.history['loss'], label='Pérdida (Entrenamiento)')
plt.plot(historia.history['val_loss'], label='Pérdida (Validación)')
plt.title('Evolución de la Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(historia.history['accuracy'], label='Precisión (Entrenamiento)')
plt.plot(historia.history['val_accuracy'], label='Precisión (Validación)')
plt.title('Evolución de la Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('entrenamiento_metricas.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfica de métricas guardada en 'entrenamiento_metricas.png'")

# ============================================================================
# 9. FUNCIÓN DE PREDICCIÓN DE SIGUIENTE PALABRA
# ============================================================================
print("\n" + "="*70)
print("9. FUNCIÓN DE PREDICCIÓN")
print("="*70)

def predecir_siguiente_palabra(texto_entrada, top_n=5):
    """
    Predice las siguientes palabras más probables dado un texto de entrada.
    
    Args:
        texto_entrada (str): Texto inicial (ej: "hola")
        top_n (int): Número de predicciones a mostrar
    
    Returns:
        list: Lista de tuplas (palabra, probabilidad)
    """
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

def generar_texto(texto_semilla, num_palabras=5):
    """
    Genera una secuencia de palabras a partir de un texto semilla.
    
    Args:
        texto_semilla (str): Texto inicial
        num_palabras (int): Número de palabras a generar
    
    Returns:
        str: Texto generado
    """
    texto_resultado = texto_semilla
    
    for _ in range(num_palabras):
        # Predecir siguiente palabra
        predicciones = predecir_siguiente_palabra(texto_resultado, top_n=1)
        
        if predicciones:
            siguiente_palabra = predicciones[0][0]
            texto_resultado += " " + siguiente_palabra
        else:
            break
    
    return texto_resultado

# ============================================================================
# 10. EJEMPLOS DE PREDICCIÓN
# ============================================================================
print("\n" + "="*70)
print("10. EJEMPLOS DE PREDICCIÓN DE SIGUIENTE PALABRA")
print("="*70)

# Ejemplos de entrada para probar
ejemplos_entrada = [
    "hola",
    "buenos días",
    "cómo estás",
    "qué tal",
    "buenas noches",
    "hola amigo",
    "buenos",
    "qué"
]

print("\n" + "─"*70)
print("PREDICCIONES CON PROBABILIDADES:")
print("─"*70)

for ejemplo in ejemplos_entrada:
    print(f"\n📝 Entrada: '{ejemplo}'")
    predicciones = predecir_siguiente_palabra(ejemplo, top_n=5)
    
    print("   Siguientes palabras más probables:")
    for i, (palabra, prob) in enumerate(predicciones, 1):
        print(f"   {i}. '{palabra}' → {prob*100:.2f}%")
        
print("\n" + "─"*70)
print("GENERACIÓN DE TEXTO COMPLETO:")
print("─"*70)

for ejemplo in ["hola", "buenos días", "cómo"]:
    texto_generado = generar_texto(ejemplo, num_palabras=4)
    print(f"\n🔹 Semilla: '{ejemplo}'")
    print(f"   Generado: '{texto_generado}'")

# ============================================================================
# 11. GUARDAR EL MODELO
# ============================================================================
print("\n" + "="*70)
print("11. GUARDANDO MODELO Y TOKENIZER")
print("="*70)

modelo.save('modelo_saludos_rnn.keras')
print("✓ Modelo guardado en 'modelo_saludos_rnn.keras'")

# Guardar tokenizer
import pickle
with open('tokenizer_saludos.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✓ Tokenizer guardado en 'tokenizer_saludos.pickle'")

print("\n" + "="*70)
print("¡ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS!")
print("="*70)
