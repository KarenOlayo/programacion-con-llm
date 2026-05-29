import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def analizar_estabilidad_coeficientes(X, y, n_bootstrap=100):

    """
    Analiza la estabilidad de los coeficientes de un modelo Lasso
    mediante bootstrap. Devuelve un DataFrame ordenado por el
    Coeficiente de Variación (CV = std / mean) de menor a mayor.
    """
  
    coefs_acumulados = []
    n_filas = X.shape[0]

    for _ in range(n_bootstrap):
      
        # 1. Remuestreo con reemplazo (Numpy)
        indices = np.random.choice(n_filas, size=n_filas, replace=True)
        X_res = X.iloc[indices]
        y_res = y.iloc[indices]

        # 2. Escalado + Lasso (Sklearn)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)

        modelo = Lasso(alpha=0.1)
        modelo.fit(X_scaled, y_res)

        # 3. Almacenar coeficientes
        coefs_acumulados.append(modelo.coef_)

    # 4. DataFrame de coeficientes por iteración
    df_coefs = pd.DataFrame(coefs_acumulados, columns=X.columns)

    # 5. Cálculo del CV = std / mean
    medias = df_coefs.mean()
    desviaciones = df_coefs.std()
    cv_valores = desviaciones / medias

    resultado = pd.DataFrame({
        'variable': X.columns,
        'cv': cv_valores.values
    })

    return resultado.sort_values(by='cv').reset_index(drop=True)
