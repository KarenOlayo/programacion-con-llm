import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analizar_series_por_grupo(seed=None):
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función analizar_series_por_grupo.
    """
    
    # ----------------------------------------------------------------
    # 0. Semilla opcional para reproducibilidad
    # ----------------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ----------------------------------------------------------------
    # 1. Configuración aleatoria
    # ----------------------------------------------------------------
    n_grupos = random.randint(2, 5)
    n_filas_por_grupo = random.randint(10, 20)
    ventana = random.randint(3, 5)
    umbral_zscore = round(random.uniform(1.0, 2.5), 2)

    columna_grupo = 'grupo'
    columna_tiempo = 'timestamp'
    columna_valor = 'valor'

    # ----------------------------------------------------------------
    # 2. Generar datos aleatorios
    # ----------------------------------------------------------------
    grupos = [f'grupo_{i}' for i in range(n_grupos)]
    filas = []

    for grupo in grupos:
        timestamps = pd.date_range(
            start=pd.Timestamp('2023-01-01') + pd.Timedelta(days=random.randint(0, 30)),
            periods=n_filas_por_grupo,
            freq='h'
        )
        valores = np.random.randn(n_filas_por_grupo) * random.uniform(1, 10) + random.uniform(-5, 5)
        for t, v in zip(timestamps, valores):
            filas.append({columna_grupo: grupo, columna_tiempo: t, columna_valor: v})

    # Mezclar filas para que no estén ordenadas por grupo
    random.shuffle(filas)
    datos = pd.DataFrame(filas).reset_index(drop=True)

    # ----------------------------------------------------------------
    # 3. Construir INPUT
    # ----------------------------------------------------------------
    input_data = {
        'datos': datos.copy(),
        'columna_grupo': columna_grupo,
        'columna_tiempo': columna_tiempo,
        'columna_valor': columna_valor,
        'ventana': ventana,
        'umbral_zscore': umbral_zscore
    }

    # ----------------------------------------------------------------
    # 4. Calcular OUTPUT esperado (Ground Truth)
    # ----------------------------------------------------------------
    df = datos.copy()

    # Preservar índice original antes de ordenar
    df['_original_index'] = df.index
    df = df.sort_values(by=[columna_grupo, columna_tiempo])

    # Calcular media móvil y std móvil por grupo
    df['media_movil'] = (
        df.groupby(columna_grupo)[columna_valor]
        .transform(lambda x: x.rolling(window=ventana, min_periods=ventana).mean())
    )
    df['std_movil'] = (
        df.groupby(columna_grupo)[columna_valor]
        .transform(lambda x: x.rolling(window=ventana, min_periods=ventana).std())
    )

    # Calcular z-score controlando división por cero
    df['zscore'] = np.where(
        df['std_movil'] > 0,
        (df[columna_valor] - df['media_movil']) / df['std_movil'],
        np.nan
    )

    # Calcular es_cambio
    df['es_cambio'] = df['zscore'].abs() > umbral_zscore
    df['es_cambio'] = df['es_cambio'].where(df['zscore'].notna(), other=False)

    # Restaurar orden original exacto y eliminar columna auxiliar
    df = df.sort_values('_original_index').drop(columns=['_original_index'])

    output_data = df

    return input_data, output_data
