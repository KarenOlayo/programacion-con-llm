import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def predecir_consumo_energetico(df: pd.DataFrame, target_col: str) -> dict:
    """
    Detecta outliers con IsolationForest, entrena un GradientBoostingRegressor
    y devuelve predicciones + métricas sobre el conjunto de test.
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Split 75/25 ANTES del IsolationForest (test nunca se toca)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Escalar con MinMaxScaler (fit solo en train)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # IsolationForest solo sobre train
    iso = IsolationForest(contamination=0.05, random_state=42)
    mask = iso.fit_predict(X_train_scaled) == 1
    n_outliers = int((~mask).sum())

    X_train_clean = X_train_scaled[mask]
    y_train_clean = y_train[mask]

    # Entrenar regresor
    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(X_train_clean, y_train_clean)

    # Predecir sobre test
    y_pred = reg.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    return {
        "predicciones":          y_pred,
        "rmse":                  rmse,
        "r2":                    r2,
        "n_outliers_eliminados": n_outliers,
    }