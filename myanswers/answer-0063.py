import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def analizar_estabilidad_coeficientes(X, y, n_bootstrap=100):
    """
    Realiza un análisis de estabilidad de coeficientes usando Bootstrap y Lasso.
    """
    coefs_acumulados = []
    n_filas = X.shape[0]

    np.random.seed(42)  # fijar semilla para reproducibilidad
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_filas, size=n_filas, replace=True)
        X_res = X.iloc[indices]
        y_res = y.iloc[indices]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)

        modelo = Lasso(alpha=0.1)
        modelo.fit(X_scaled, y_res)

        coefs_acumulados.append(modelo.coef_)

    df_coefs = pd.DataFrame(coefs_acumulados, columns=X.columns)

    medias = df_coefs.mean()
    desviaciones = df_coefs.std()
    cv_valores = desviaciones / medias

    resultado = pd.DataFrame({
        'variable': X.columns,
        'cv': cv_valores.values
    })

    return resultado.sort_values(by='cv').reset_index(drop=True)
