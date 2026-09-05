from pathlib import Path
import pandas as pd
import numpy as np

# RUTAS

BASE_PATH = Path(__file__).parent.parent
TRAIN_PATH = BASE_PATH / "data" / "train.csv"

# CARGAR TRAINING

df = pd.read_csv(TRAIN_PATH, dtype={"activity": str})

# Separar características y etiqueta
X = df.drop(columns=["activity"])
y = df["activity"]

print("Training original:")
print("Shape X:", X.shape)
print("Shape y:", y.shape)

# CALCULAR PARÁMETROS SOLO CON TRAINING

means = X.mean()
stds = X.std(ddof=0)

# Evitar división entre cero si alguna característica
# no presenta variación
stds = stds.replace(0, 1)

# NORMALIZACIÓN Z-SCORE

X_normalized = (X - means) / stds

# Volver a agregar la etiqueta
train_normalized = X_normalized.copy()
train_normalized["activity"] = y.values

# GUARDAR TRAINING NORMALIZADO

OUTPUT_PATH = BASE_PATH / "data" / "train_normalized.csv"

train_normalized.to_csv(OUTPUT_PATH, index=False)

# GUARDAR PARÁMETROS DE NORMALIZACIÓN

parameters = pd.DataFrame({
    "feature": X.columns,
    "mean": means.values,
    "std": stds.values
})

PARAMETERS_PATH = BASE_PATH / "data" / "normalization_parameters.csv"

parameters.to_csv(PARAMETERS_PATH, index=False)

# COMPROBACIONES

print("\nTraining normalizado:")
print("Shape:", train_normalized.shape)

print("\nPrimeras filas:")
print(train_normalized.head())

print("\nMedia aproximada después de normalizar:")
print(X_normalized.mean().head())

print("\nDesviación estándar aproximada:")
print(X_normalized.std(ddof=0).head())

print("\nArchivos creados:")
print(OUTPUT_PATH)
print(PARAMETERS_PATH)