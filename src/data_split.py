from pathlib import Path
import pandas as pd
import numpy as np

# RUTAS

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data" / "rehab_features.csv"

# CARGAR DATASET

df = pd.read_csv(DATA_PATH, dtype={"activity": str})

print("Dataset original:")
print("Shape:", df.shape)

# PARTICIÓN ESTRATIFICADA MANUAL

np.random.seed(42)

train_parts = []
reserved_parts = []

for activity in sorted(df["activity"].unique()):

    group = df[df["activity"] == activity].copy()

    # Mezclar únicamente las filas de esta actividad
    indices = np.random.permutation(len(group))
    group = group.iloc[indices]

    # 80 % para training
    split_index = int(len(group) * 0.8)

    train_parts.append(group.iloc[:split_index])
    reserved_parts.append(group.iloc[split_index:])

# UNIR LOS GRUPOS

train_df = pd.concat(train_parts, ignore_index=True)
reserved_df = pd.concat(reserved_parts, ignore_index=True)

# Mezclar el orden final
train_df = train_df.iloc[np.random.permutation(len(train_df))].reset_index(drop=True)
reserved_df = reserved_df.iloc[np.random.permutation(len(reserved_df))].reset_index(drop=True)

# GUARDAR

train_path = BASE_PATH / "data" / "train.csv"
reserved_path = BASE_PATH / "data" / "reserved.csv"

train_df.to_csv(train_path, index=False)
reserved_df.to_csv(reserved_path, index=False)

# COMPROBACIONES

print("\nTRAINING")
print("Shape:", train_df.shape)
print(train_df["activity"].value_counts().sort_index())

print("\nRESERVADO")
print("Shape:", reserved_df.shape)
print(reserved_df["activity"].value_counts().sort_index())

print("\nTotal:")
print(len(train_df) + len(reserved_df))

print("\nArchivos creados:")
print(train_path)
print(reserved_path)