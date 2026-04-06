import pandas as pd
import numpy as np
import random
from sklearn.decomposition import IncrementalPCA

def generar_caso_de_uso_reducir_dimensionalidad_incremental(seed=None):
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función reducir_dimensionalidad_incremental.
    """

    # ---------------------------------------------------------
    # 0. Semilla opcional
    # ---------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ---------------------------------------------------------
    # 1. Dimensiones base
    # ---------------------------------------------------------
    n_filas_base = random.randint(80, 200)
    n_features = random.randint(5, 12)

    n_componentes = random.randint(2, min(n_features - 1, 6))

    # ---------------------------------------------------------
    # 2. batch_size válido
    # ---------------------------------------------------------
    batch_size = random.randint(
        n_componentes + 1,
        max(n_componentes + 2, n_filas_base // 5)
    )

    # ---------------------------------------------------------
    # 3. Ajustar n_filas para evitar batches inválidos
    # ---------------------------------------------------------
    n_filas = (n_filas_base // batch_size) * batch_size

    # Garantía adicional (caso extremo)
    if n_filas == 0:
        n_filas = batch_size

    # ---------------------------------------------------------
    # 4. Generación de datos correlacionados
    # ---------------------------------------------------------
    base = np.random.randn(n_filas, 1)
    ruido = np.random.randn(n_filas, n_features) * 0.3

    X = base @ np.random.uniform(0.5, 2.0, size=(1, n_features)) + ruido

    columnas = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columnas)

    # ---------------------------------------------------------
    # 5. INPUT
    # ---------------------------------------------------------
    input_data = {
        "datos": df.copy(),
        "n_componentes": n_componentes,
        "batch_size": batch_size
    }

    # ---------------------------------------------------------
    # 6. OUTPUT esperado (ground truth)
    # ---------------------------------------------------------
    modelo = IncrementalPCA(
        n_components=n_componentes,
        batch_size=batch_size
    )

    total_samples_fit = 0

    for i in range(0, n_filas, batch_size):
        batch = df.iloc[i:i + batch_size]
        modelo.partial_fit(batch)
        total_samples_fit += len(batch)

    assert total_samples_fit == n_filas

    # Transformación completa
    X_transformado = modelo.transform(df)

    output_data = X_transformado

    return input_data, output_data

if __name__ == "__main__":
    
    input_data, expected_output = generar_caso_de_uso_reducir_dimensionalidad_incremental()
    
    print("Input:\n")
    print(input_data)

    print("\nOutput:\n")
    print(expected_output)
