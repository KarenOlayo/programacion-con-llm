import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


def evaluar_importancia_temporal(df: pd.DataFrame, target_col: str, fecha_col: str) -> pd.DataFrame:
    """
    Calcula la importancia de características por permutación usando
    agrupación temporal por año. Usa los dos primeros años como train
    y el último como evaluación.
    """
    columnas_vacias = ["feature", "importancia_media", "importancia_std"]

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df["_anio"] = df[fecha_col].dt.year
    anios = sorted(df["_anio"].unique())

    if len(anios) < 3:
        return pd.DataFrame(columns=columnas_vacias)

    anios_train = anios[:2]
    anio_test   = anios[-1]

    feature_cols = [
        c for c in df.columns
        if c not in [fecha_col, target_col, "_anio"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    mask_train = df["_anio"].isin(anios_train)
    mask_test  = df["_anio"] == anio_test

    X_train = df.loc[mask_train, feature_cols]
    y_train = df.loc[mask_train, target_col]
    X_test  = df.loc[mask_test,  feature_cols]
    y_test  = df.loc[mask_test,  target_col]

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train_scaled, y_train)

    result = permutation_importance(
        modelo, X_test_scaled, y_test,
        n_repeats=10, random_state=42
    )

    resultado = pd.DataFrame({
        "feature":           feature_cols,
        "importancia_media": result.importances_mean,
        "importancia_std":   result.importances_std,
    })

    return resultado.sort_values(by="importancia_media", ascending=False).reset_index(drop=True)
