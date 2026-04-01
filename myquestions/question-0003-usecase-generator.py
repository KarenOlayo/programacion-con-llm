import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_alinear_eventos_con_referencias(seed=None):
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función alinear_eventos_con_referencias.
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
    n_eventos = random.randint(50, 120)
    n_refs = random.randint(20, 80)

    # ---------------------------------------------------------
    # 2. Generación de timestamps (datetime, sin duplicados)
    # ---------------------------------------------------------
    base_time = pd.Timestamp("2024-01-01")

    # Referencias empiezan más tarde
    tiempos_refs = base_time + pd.to_timedelta(
        np.random.choice(np.arange(800, 10000), size=n_refs, replace=False),
        unit="s"
    )

    # Eventos:
    # - algunos antes → generan NaN
    n_early = min(5, n_eventos // 4)

    tiempos_early = base_time + pd.to_timedelta(
        np.random.choice(np.arange(0, 500), size=n_early, replace=False),
        unit="s"
    )

    tiempos_late = base_time + pd.to_timedelta(
        np.random.choice(np.arange(1000, 12000), size=n_eventos - n_early, replace=False),
        unit="s"
    )

    tiempos_eventos = np.sort(np.concatenate([tiempos_early, tiempos_late]))

    # ---------------------------------------------------------
    # 3. Construcción de DataFrames
    # ---------------------------------------------------------
    eventos = pd.DataFrame({
        "timestamp": tiempos_eventos,
        "evento_valor": np.random.randn(n_eventos)
    })

    referencias = pd.DataFrame({
        "timestamp": tiempos_refs,
        "valor_ref": np.random.uniform(0, 100, size=n_refs)
    })

    columna_tiempo = "timestamp"
    columna_valor = "valor_ref"

    # ---------------------------------------------------------
    # 4. INPUT
    # ---------------------------------------------------------
    input_data = {
        "eventos": eventos.copy(),
        "referencias": referencias.copy(),
        "columna_tiempo": columna_tiempo,
        "columna_valor": columna_valor
    }

    # ---------------------------------------------------------
    # 5. OUTPUT esperado (ground truth)
    # ---------------------------------------------------------
    eventos_sorted = eventos.sort_values(
        by=columna_tiempo
    ).reset_index(drop=True)

    referencias_sorted = referencias.sort_values(
        by=columna_tiempo
    ).reset_index(drop=True)

    resultado = pd.merge_asof(
        eventos_sorted,
        referencias_sorted,
        on=columna_tiempo,
        direction="backward"
    )

    resultado = resultado.rename(columns={
        columna_valor: "valor_referencia"
    })

    output_data = resultado

    return input_data, output_data
