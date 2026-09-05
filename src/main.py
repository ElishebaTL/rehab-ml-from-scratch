from pathlib import Path
import pandas as pd

from model import KNNClassifier


# Rutas
BASE_PATH = Path(__file__).parent.parent
TRAIN_PATH = BASE_PATH / "data" / "train_normalized.csv"


# Cargar training
df = pd.read_csv(
    TRAIN_PATH,
    dtype={"activity": str}
)

X_train = df.drop(columns=["activity"]).to_numpy()
y_train = df["activity"].to_numpy()

print("=== DATOS DE ENTRENAMIENTO ===")
print("Muestras:", X_train.shape[0])
print("Características:", X_train.shape[1])
print("Clases:", sorted(set(y_train)))


# Crear y entrenar KNN
k = 5

model = KNNClassifier(k=k)
model.fit(X_train, y_train)

print("\n=== MODELO ===")
print(f"KNN creado con k = {k}")
print("Modelo entrenado correctamente.")


# Probar algunas muestras del training
num_examples = 5

X_examples = X_train[:num_examples]
y_real = y_train[:num_examples]

predictions = model.predict(X_examples)

print("\n=== PREDICCIONES DE PRUEBA ===")

for i in range(num_examples):
    print(
        f"Muestra {i + 1}: "
        f"Real = {y_real[i]} | "
        f"Predicción = {predictions[i]}"
    )