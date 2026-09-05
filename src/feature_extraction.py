from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# RUTA AL DATASET ORIGINAL
# --------------------------------------------------

DATASET_PATH = Path(
    r"C:\Users\elish\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Desktop\bs\REHAB\REHAB\Rehab_exercise\d02_processed_data"
)

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

canales = ["f1", "f2", "f3", "f4", "f5", "pitch3"]

actividades_excluidas = {"014"}

filas = []

# --------------------------------------------------
# RECORRER ACTIVIDADES
# --------------------------------------------------

for i in range(16):
    actividad = f"{i:03d}"

    # Excluir actividad 014
    if actividad in actividades_excluidas:
        print(f"Actividad {actividad} excluida.")
        continue

    archivo = DATASET_PATH / f"{actividad}_2.npy"

    if not archivo.exists():
        print(f"No se encontró: {archivo.name}")
        continue

    datos = np.load(archivo)

    print(
        f"Procesando actividad {actividad} "
        f"con {datos.shape[0]} muestras..."
    )

    # --------------------------------------------------
    # CADA MUESTRA TIENE SHAPE (880, 6)
    # --------------------------------------------------

    for muestra in datos:

        fila = {}

        for indice_canal, nombre_canal in enumerate(canales):

            valores = muestra[:, indice_canal]

            fila[f"{nombre_canal}_mean"] = np.mean(valores)
            fila[f"{nombre_canal}_std"] = np.std(valores)
            fila[f"{nombre_canal}_min"] = np.min(valores)
            fila[f"{nombre_canal}_max"] = np.max(valores)

        # Etiqueta de clase
        fila["activity"] = actividad

        filas.append(fila)

# --------------------------------------------------
# CREAR DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(filas)

print("\n=== DATASET GENERADO ===")
print("Shape:", df.shape)
print("\nPrimeras filas:")
print(df.head())

print("\nActividades incluidas:")
print(sorted(df["activity"].unique()))

print("\nCantidad de muestras por actividad:")
print(df["activity"].value_counts().sort_index())

# --------------------------------------------------
# GUARDAR CSV
# --------------------------------------------------

OUTPUT_PATH = Path(__file__).parent.parent / "data"

OUTPUT_PATH.mkdir(exist_ok=True)

archivo_salida = OUTPUT_PATH / "rehab_features.csv"

df.to_csv(archivo_salida, index=False)

print("\nArchivo guardado en:")
print(archivo_salida)