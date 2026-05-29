import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def entrenar_y_predecir(df: pd.DataFrame) -> float:
    """
    Limpia, transforma, entrena un modelo de Regresión Logística
    y devuelve el accuracy sobre el conjunto de prueba (20%).
    """
    # 1. Limpieza: eliminar filas con valores nulos
    df_clean = df.dropna()

    # 2. Transformación: one-hot encoding para 'Plan' (drop_first=True)
    df_proc = pd.get_dummies(df_clean, columns=['Plan'], drop_first=True)

    # 3. Separación de features y target
    X = df_proc.drop('Churn', axis=1)
    y = df_proc['Churn']

    # 4. División 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Entrenamiento
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    # 6. Evaluación
    predicciones = modelo.predict(X_test)
    return float(accuracy_score(y_test, predicciones))