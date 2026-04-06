import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA

def generar_caso_de_uso_detectar_anomalias_mahalanobis(seed=None):
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función detectar_anomalias_mahalanobis.
    """

    # ---------------------------------------------------------
    # 0. Semilla opcional
    # ---------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ---------------------------------------------------------
    # 1. Dimensiones
    # ---------------------------------------------------------
    n_filas = random.randint(80, 150)
    n_features = random.randint(4, 10)

    # ---------------------------------------------------------
    # 2. Datos correlacionados (bien condicionados)
    # ---------------------------------------------------------
    base = np.random.randn(n_filas, 1)
    ruido = np.random.randn(n_filas, n_features) * 0.3
    X = base @ np.random.uniform(0.5, 2.0, size=(1, n_features)) + ruido

    columnas = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columnas)

    # ---------------------------------------------------------
    # 3. Parámetros
    # ---------------------------------------------------------
    n_componentes = random.randint(2, min(n_features, 5))

    # ---------------------------------------------------------
    # 4. PCA (DETERMINISTA)
    # ---------------------------------------------------------
    pca = PCA(n_components=n_componentes)
    X_reducido = pca.fit_transform(df)

    # ---------------------------------------------------------
    # 5. Media y covarianza (SIN regularización)
    # ---------------------------------------------------------
    media = np.mean(X_reducido, axis=0)
    cov = np.cov(X_reducido, rowvar=False)

    try:
      cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
      cov_inv = np.linalg.pinv(cov)

    # ---------------------------------------------------------
    # 6. Distancia de Mahalanobis
    # ---------------------------------------------------------
    distancias = []

    for x in X_reducido:
        diff = x - media
        d = np.sqrt(diff.T @ cov_inv @ diff)
        distancias.append(d)

    distancias = np.array(distancias)

    # ---------------------------------------------------------
    # 7. Umbral (percentil → NO circular)
    # ---------------------------------------------------------
    percentil = random.uniform(85, 98)
    umbral = float(np.percentile(distancias, percentil))

    # ---------------------------------------------------------
    # 8. INPUT
    # ---------------------------------------------------------
    input_data = {
        "datos": df.copy(),
        "n_componentes": n_componentes,
        "umbral": umbral
    }

    # ---------------------------------------------------------
    # 9. OUTPUT
    # ---------------------------------------------------------
    etiquetas = (distancias > umbral).astype(int)
    output_data = (etiquetas, distancias)

    return input_data, output_data

if __name__ == "__main__":
    
    input_data, expected_output = generar_caso_de_uso_detectar_anomalias_mahalanobis()
    
    print("Input:\n")
    print(input_data)

    print("\nOutput:\n")
    print(expected_output)
