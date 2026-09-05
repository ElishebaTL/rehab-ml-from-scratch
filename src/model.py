import numpy as np


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        


    # ENTRENAMIENTO

    def fit(self, X, y):
        """
        Guarda las características y etiquetas del conjunto de entrenamiento.
        KNN es un algoritmo basado en instancias, por lo que no calcula
        parámetros internos durante el entrenamiento.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(
                "X y y deben contener la misma cantidad de muestras."
            )

        if self.k <= 0:
            raise ValueError("k debe ser mayor que 0.")

        if self.k > len(X):
            raise ValueError(
                "k no puede ser mayor que la cantidad de muestras."
            )

        self.X_train = X
        self.y_train = y


    # DISTANCIA EUCLIDIANA

    def _euclidean_distance(self, x1, x2):
        """
        Calcula manualmente la distancia euclidiana
        entre dos observaciones.
        """

        return np.sqrt(np.sum((x1 - x2) ** 2))


    # PREDICCIÓN DE UNA MUESTRA

    def predict_one(self, sample):
        if self.X_train is None:
            raise ValueError(
                "El modelo debe entrenarse antes de realizar predicciones."
            )

        sample = np.asarray(sample, dtype=float)

        distances = []

        # Calcular distancia entre la muestra nueva
        # y todas las muestras de training
        for index, train_sample in enumerate(self.X_train):

            distance = self._euclidean_distance(
                sample,
                train_sample
            )

            distances.append((distance, index))

        # Ordenar de menor a mayor distancia
        distances.sort(key=lambda item: item[0])

        # Tomar los k vecinos más cercanos
        nearest = distances[:self.k]

        # Obtener las etiquetas de esos vecinos
        neighbor_labels = [
            self.y_train[index]
            for _, index in nearest
        ]

        # Contar votos manualmente
        votes = {}

        for label in neighbor_labels:
            votes[label] = votes.get(label, 0) + 1

        # Elegir la clase con mayor cantidad de votos
        predicted_class = max(
            votes,
            key=votes.get
        )

        return predicted_class


    # PREDICCIÓN DE VARIAS MUESTRAS

    def predict(self, X):
        predictions = []

        for sample in X:
            prediction = self.predict_one(sample)
            predictions.append(prediction)

        return np.array(predictions)

if __name__ == "__main__":
    X_demo = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [5.0, 5.0],
        [5.2, 4.9]
    ])

    y_demo = np.array([
        "A",
        "A",
        "B",
        "B"
    ])

    modelo = KNNClassifier(k=3)
    modelo.fit(X_demo, y_demo)

    nueva_muestra = np.array([
        [0.1, 0.2]
    ])

    prediccion = modelo.predict(nueva_muestra)

    print("Predicción:", prediccion)