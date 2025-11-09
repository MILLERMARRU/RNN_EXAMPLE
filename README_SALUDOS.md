# Predicción de Siguiente Palabra con RNN - Proyecto de Saludos

## 📋 Objetivo
Entrenar una Red Neuronal Recurrente (RNN) para predecir la siguiente palabra en frases de saludo en español.

---

## 📁 Estructura del Proyecto

```
ejemploRNN/
├── saludos_dataset.txt              # Dataset con 93 frases de saludo
├── prediccion_saludos_RNN.py        # Script principal de entrenamiento
├── inferencia_interactiva.py        # Script para pruebas interactivas
├── modelo_saludos_rnn.keras         # Modelo entrenado guardado
├── tokenizer_saludos.pickle         # Tokenizer guardado
├── entrenamiento_metricas.png       # Gráficas de métricas
└── README_SALUDOS.md               # Este archivo
```

---

## 📊 Dataset

**Archivo:** `saludos_dataset.txt`

- **Total de frases:** 93
- **Vocabulario:** 60 palabras únicas
- **Ejemplos:**
  - "hola"
  - "hola amigo"
  - "buenos días"
  - "cómo estás"
  - "qué tal"

El dataset contiene saludos comunes en español con diferentes variaciones y combinaciones.

---

## 🔧 Proceso de Entrenamiento

### a) Tokenización y Padding

#### **Tokenización:**
Convierte palabras en números únicos (índices).

```python
Ejemplo: "hola amigo" → [1, 3]
```

Cada palabra del vocabulario recibe un índice único:
- 'hola': 1
- 'cómo': 2
- 'amigo': 3
- 'qué': 4
- etc.

#### **Creación de Secuencias:**
Para cada frase, se crean múltiples secuencias de entrenamiento:

```
Frase: "hola buenos días"
Secuencias generadas:
  [hola] → buenos
  [hola, buenos] → días
```

Esto genera 174 secuencias de entrenamiento a partir de las 93 frases originales.

#### **Padding:**
Todas las secuencias se rellenan con ceros para tener la misma longitud (longitud máxima = 4):

```
[1, 3]        → [0, 0, 1, 3]
[1, 6, 7]     → [0, 1, 6, 7]
[1, 6, 7, 3]  → [1, 6, 7, 3]
```

---

### b) Arquitectura del Modelo

```
┌─────────────────────────────────────┐
│ CAPA EMBEDDING                      │
│ - Input dim: 60 palabras            │
│ - Output dim: 100 (vectores densos) │
│ - Convierte índices → vectores      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ CAPA SimpleRNN                      │
│ - 150 unidades recurrentes          │
│ - Captura patrones temporales       │
│ - return_sequences=False            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ CAPA DENSE (Salida)                 │
│ - 60 neuronas (una por palabra)     │
│ - Activación: softmax               │
│ - Output: probabilidades            │
└─────────────────────────────────────┘
```

**Parámetros del modelo:**
- Total de parámetros: ~38,000
- Optimizador: Adam
- Función de pérdida: Categorical Crossentropy
- Métrica: Accuracy

---

### c) Entrenamiento

**Configuración:**
- **Épocas:** 200
- **Batch size:** 32
- **Validación:** 20% de los datos

**Resultados finales:**
- **Pérdida (training):** 0.8738
- **Pérdida (validación):** 8.6929
- **Precisión (training):** 56.83%
- **Precisión (validación):** 8.57%

**Nota:** La diferencia entre training y validación indica overfitting, lo cual es esperado dado el tamaño pequeño del dataset. El modelo memoriza bien los patrones de entrenamiento.

---

### d) Ejemplos de Entrada y Salida

#### Ejemplo 1: "hola"
```
Entrada: "hola"

Top 5 predicciones:
1. 'amigo'  → 20.87%
2. 'señor'  → 10.63%
3. 'buenas' → 10.59%
4. 'amiga'  → 10.45%
5. 'mundo'  → 5.26%

Frase generada: "hola amigo cómo estás hoy"
```

#### Ejemplo 2: "buenos días"
```
Entrada: "buenos días"

Top 5 predicciones:
1. 'a'    → 28.42%
2. 'señor' → 14.58%
3. 'qué'   → 14.49%
4. 'cómo'  → 14.23%
5. 'amigo' → 13.95%

Frase generada: "buenos días a todos"
```

#### Ejemplo 3: "cómo estás"
```
Entrada: "cómo estás"

Top 5 predicciones:
1. 'hoy'   → 33.65%
2. 'todo'  → 32.90%
3. 'amigo' → 32.36%
4. 'encuentras' → 0.12%
5. 'va'    → 0.11%

Frase generada: "cómo estás hoy"
```

#### Ejemplo 4: "qué tal"
```
Entrada: "qué tal"

Top 5 predicciones:
1. 'todo'  → 39.92%
2. 'están' → 20.22%
3. 'estás' → 19.97%
4. 'amigo' → 19.21%
5. 'cuentas' → 0.14%

Frase generada: "qué tal todo bien"
```

---

### e) Evaluación del Desempeño

#### **Métricas principales:**

| Métrica | Training | Validación |
|---------|----------|------------|
| **Pérdida (Loss)** | 0.874 | 8.693 |
| **Precisión (Accuracy)** | 56.83% | 8.57% |

#### **Análisis de las gráficas:**

![Métricas de Entrenamiento](entrenamiento_metricas.png)

**Gráfica de Pérdida:**
- La pérdida de entrenamiento disminuye constantemente (de ~4.0 a ~0.87)
- La pérdida de validación aumenta (overfitting)
- Esto indica que el modelo memoriza bien los datos de entrenamiento

**Gráfica de Precisión:**
- La precisión de entrenamiento alcanza ~60%
- La precisión de validación se mantiene baja (~8%)
- El dataset es pequeño, por lo que el modelo se especializa en los ejemplos vistos

#### **Interpretación:**

✅ **Aspectos positivos:**
- El modelo aprende patrones correctamente
- Las predicciones para frases conocidas son precisas
- La generación de texto es coherente con el dominio (saludos)

⚠️ **Limitaciones:**
- Overfitting debido al tamaño pequeño del dataset
- Baja generalización a frases no vistas
- El modelo funciona mejor con las frases exactas del training

#### **Mejoras sugeridas:**

1. **Aumentar el dataset:** Agregar más variaciones de saludos
2. **Regularización:** Añadir Dropout layers
3. **Arquitectura:** Probar LSTM en lugar de SimpleRNN
4. **Embeddings pre-entrenados:** Usar Word2Vec o GloVe
5. **Data augmentation:** Crear más variaciones de las frases existentes

---

## 🚀 Cómo Usar

### 1. Entrenar el modelo

```bash
python prediccion_saludos_RNN.py
```

Esto generará:
- `modelo_saludos_rnn.keras` (modelo entrenado)
- `tokenizer_saludos.pickle` (tokenizer)
- `entrenamiento_metricas.png` (gráficas)

### 2. Usar el modelo interactivamente

```bash
python inferencia_interactiva.py
```

Luego ingresa frases de prueba:
```
📝 Ingresa una frase: hola

🔍 TOP 5 PREDICCIONES:
   1. 'amigo' → 20.87%
   2. 'señor' → 10.63%
   3. 'buenas' → 10.59%
   4. 'amiga' → 10.45%
   5. 'mundo' → 5.26%

💬 FRASE GENERADA:
   hola amigo cómo estás hoy
```

---

## 📦 Dependencias

```
tensorflow>=2.20.0
keras>=3.12.0
numpy>=2.3.0
matplotlib>=3.10.0
```

Instalar con:
```bash
pip install tensorflow keras numpy matplotlib
```

---

## 🎯 Casos de Uso

1. **Autocompletado de texto** en aplicaciones de chat
2. **Sugerencias de respuesta** en sistemas de mensajería
3. **Aprendizaje de patrones de lenguaje** en dominios específicos
4. **Generación de texto** para respuestas automáticas

---

## 📈 Resultados Visuales

### Predicciones por Entrada

| Entrada | Mejor Predicción | Confianza |
|---------|------------------|-----------|
| "hola" | "amigo" | 20.87% |
| "buenos" | "días" | 99.89% |
| "qué" | "tal" | 61.08% |
| "cómo estás" | "hoy" | 33.65% |
| "buenas noches" | "amigo" | 50.43% |

---

## 🧠 Conceptos Clave Aprendidos

1. **Tokenización:** Conversión de texto a números
2. **Padding:** Normalización de longitudes de secuencia
3. **Embeddings:** Representación densa de palabras
4. **RNN:** Procesamiento de secuencias temporales
5. **Softmax:** Distribución de probabilidades sobre vocabulario
6. **Overfitting:** Especialización excesiva en datos de entrenamiento

---

## 👨‍💻 Autor

Miller - Universidad Católica Sedes Sapientiae (UCSS)
Ciclo 09 - Inteligencia Artificial

---

## 📝 Notas Adicionales

- El modelo predice **una palabra a la vez**
- Las probabilidades suman 100% sobre todo el vocabulario
- El modelo puede generar secuencias completas iterativamente
- La calidad mejora significativamente con más datos de entrenamiento

---

## 🔍 Experimentos Adicionales

Prueba modificar estos parámetros en `prediccion_saludos_RNN.py`:

```python
# Línea 169: Cambiar tipo de capa RNN
SimpleRNN(150) → LSTM(150)  # Mejor memoria a largo plazo

# Línea 165: Cambiar dimensión de embedding
output_dim=100 → output_dim=200  # Vectores más ricos

# Línea 191: Aumentar épocas
epochs=200 → epochs=500  # Más entrenamiento
```

¡Explora y experimenta con diferentes configuraciones!
