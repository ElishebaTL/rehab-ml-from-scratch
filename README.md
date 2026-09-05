# REHAB ML From Scratch

Implementación de una técnica clásica de aprendizaje máquina sobre el dataset REHAB, desarrollada manualmente en Python sin utilizar frameworks de machine learning.

## Objetivo

Transformar las series temporales del dataset REHAB en una estructura adecuada para aprendizaje máquina e implementar manualmente un algoritmo capaz de clasificar las diferentes actividades.

## Dataset

Cada muestra original corresponde a una serie temporal con:

- 880 observaciones
- 6 señales

Para convertir cada muestra en un vector de características se extrajeron cuatro medidas descriptivas de cada señal:

- Media (`mean`)
- Desviación estándar (`std`)
- Mínimo (`min`)
- Máximo (`max`)

Esto produce:

6 señales × 4 características = 24 características por muestra.

La columna `activity` contiene la etiqueta correspondiente a la actividad.

La actividad `014` fue excluida siguiendo la recomendación realizada durante el desarrollo de la actividad.

El dataset final contiene 4257 muestras y 25 columnas: 24 características y una etiqueta.

## Separación de los datos

El dataset fue dividido de manera estratificada en:

- Training: 3400 muestras
- Conjunto reservado: 857 muestras

El conjunto reservado se mantiene separado y no se utiliza durante esta etapa de entrenamiento, con el objetivo de prevenir data leakage.

## Preprocesamiento

Las 24 características fueron normalizadas mediante estandarización.

Los parámetros de normalización fueron calculados utilizando exclusivamente el conjunto de entrenamiento.

Para cada característica se aplica:

`z = (x - media) / desviación estándar`

Los parámetros obtenidos se almacenan por separado para poder aplicar posteriormente la misma transformación a nuevos datos sin recalcularlos.

## Algoritmo

Se implementó manualmente el algoritmo K-Nearest Neighbors (KNN).

La implementación:

1. Almacena las muestras y etiquetas de entrenamiento.
2. Calcula la distancia euclidiana entre una observación y las muestras de entrenamiento.
3. Selecciona los `k` vecinos más cercanos.
4. Cuenta las clases de los vecinos.
5. Asigna la clase con mayor frecuencia.

Para esta implementación se utilizó:

`k = 5`

No se utiliza una implementación de KNN proporcionada por una biblioteca o framework de aprendizaje máquina.

## Resultados de entrenamiento

El modelo fue probado sobre las 3400 muestras del conjunto de entrenamiento.

Resultados:

- Predicciones correctas: 2756 de 3400
- Accuracy de entrenamiento: 81.06 %

Este resultado corresponde únicamente al conjunto de entrenamiento y no representa todavía una evaluación de la capacidad de generalización del modelo.

## Estructura del proyecto

```text
rehab-ml-from-scratch/
│
├── data/
│   ├── rehab_features.csv
│   ├── train.csv
│   ├── train_normalized.csv
│   ├── reserved.csv
│   └── normalization_parameters.csv
│
├── src/
│   ├── data_split.py
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── model.py
│   └── main.py
│
└── README.md
```

## Ejecución

El modelo final puede ejecutarse directamente desde Python:

```bash
python src/main.py
```

El programa carga el conjunto de entrenamiento normalizado, crea y entrena el modelo KNN y realiza predicciones de prueba.

## Tecnologías

- Python
- NumPy
- pandas

NumPy y pandas se utilizan para el manejo y transformación de los datos. El algoritmo KNN fue implementado manualmente.