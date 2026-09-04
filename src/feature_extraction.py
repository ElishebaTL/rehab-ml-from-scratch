from pathlib import Path
import numpy as np

# Ruta al dataset original
DATASET_PATH = Path(
    r"C:\Users\elish\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Desktop\bs\REHAB\REHAB\Rehab_exercise\d02_processed_data"
)

# Archivo de prueba
archivo = DATASET_PATH / "000_1.npy"

datos = np.load(archivo)

print("Shape del archivo:", datos.shape)

# Tomamos SOLO la primera muestra/ejecución
muestra = datos[0]

print("Shape de una muestra:", muestra.shape)

# Extraer features por canal
features = []

for canal in range(muestra.shape[1]):
    valores = muestra[:, canal]

    features.extend([
        np.mean(valores),
        np.std(valores),
        np.min(valores),
        np.max(valores)
    ])

print("Cantidad de features:", len(features))
print("Features:")
print(features)