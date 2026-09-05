from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).parent.parent
data_dir = base / "data"
output_dir = base / "figures"

output_dir.mkdir(exist_ok=True)

dataset = pd.read_csv(data_dir / "rehab_features.csv")
train = pd.read_csv(data_dir / "train.csv")
reserved = pd.read_csv(data_dir / "reserved.csv")

# Distribución de clases del dataset completo

class_counts = dataset["activity"].astype(str).str.zfill(3).value_counts().sort_index()

plt.figure(figsize=(11, 6))
plt.bar(class_counts.index, class_counts.values)

plt.title("Distribución de actividades en el dataset generado")
plt.xlabel("Actividad")
plt.ylabel("Número de observaciones")

plt.tight_layout()

path1 = output_dir / "class_distribution.png"
plt.savefig(path1, dpi=300)
plt.close()

print("Gráfica creada:", path1)


# Comparación entre training y conjunto reservado

train_counts = (
    train["activity"]
    .astype(str)
    .str.zfill(3)
    .value_counts()
    .sort_index()
)

reserved_counts = (
    reserved["activity"]
    .astype(str)
    .str.zfill(3)
    .value_counts()
    .sort_index()
)

activities = train_counts.index

x = np.arange(len(activities))
width = 0.38

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    train_counts.values,
    width,
    label="Training"
)

plt.bar(
    x + width / 2,
    reserved_counts.values,
    width,
    label="Reserved"
)

plt.title("Distribución de actividades después de la partición")
plt.xlabel("Actividad")
plt.ylabel("Número de observaciones")

plt.xticks(x, activities)
plt.legend()

plt.tight_layout()

path2 = output_dir / "train_reserved_distribution.png"
plt.savefig(path2, dpi=300)
plt.close()

print("Gráfica creada:", path2)