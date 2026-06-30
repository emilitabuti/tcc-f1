from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 04 - Normalização das variáveis numéricas
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "preprocessing"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_encoded_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_encoded_2018_2025.csv"
INPUT_BASE_LIMPA_2018_2024 = PROCESSED_DIR / "base_historica_encoded_2018_2024.csv"
INPUT_BASE_LIMPA_2018_2025 = PROCESSED_DIR / "base_historica_encoded_2018_2025.csv"


# Arquivos de saida
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_normalizado_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_normalizado_2018_2025.csv"
OUTPUT_BASE_LIMPA_2018_2024 = PROCESSED_DIR / "base_historica_normalizado_2018_2024.csv"
OUTPUT_BASE_LIMPA_2018_2025 = PROCESSED_DIR / "base_historica_normalizado_2018_2025.csv"

STANDARD_SCALER_HISTORICO = MODELS_DIR / "standard_scaler_historico.joblib"
MINMAX_SCALER_HISTORICO = MODELS_DIR / "minmax_scaler_historico.joblib"
STANDARD_SCALER_BASE_LIMPA = MODELS_DIR / "standard_scaler_base_historica.joblib"
MINMAX_SCALER_BASE_LIMPA = MODELS_DIR / "minmax_scaler_base_historica.joblib"


# variáveis contínuas que vão pro Z-score
ZSCORE_COLUMNS = [
    "laps",
    "fastf1_laps_count",
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_avg_sector1",
    "fastf1_avg_sector2",
    "fastf1_avg_sector3",
    "fastf1_max_tyre_life",
    "fastf1_stints_count",
    "fastf1_pit_in_count",
    "fastf1_pit_out_count",
]

# grid e laps ficam no MinMax - têm limite natural de valores
MINMAX_COLUMNS = [
    "grid_position",
    "laps",
]


# Funções auxiliares
def validar_colunas(df, colunas_obrigatorias, nome_base):
    #Valida se as colunas obrigatórias existem no DataFrame.

    colunas_ausentes = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"As seguintes colunas estão ausentes em {nome_base}: "
            f"{colunas_ausentes}"
        )


def repo_relative(path):
    # Registra caminhos portáveis no relatório, independentemente da máquina.
    return path.relative_to(BASE_DIR).as_posix()


def selecionar_colunas_existentes(df, colunas):
    #Retorna apenas as colunas que existem no DataFrame.
    return [coluna for coluna in colunas if coluna in df.columns]


def preencher_nulos_com_mediana(df_treino, df_aplicacao, colunas):
    # preenche nulos usando a mediana da base de treino - evita vazar info da 2025
    df_treino = df_treino.copy()
    df_aplicacao = df_aplicacao.copy()

    medianas = {}

    for coluna in colunas:
        mediana = df_treino[coluna].median()

        if pd.isna(mediana):
            mediana = 0

        medianas[coluna] = mediana

        df_treino[coluna] = df_treino[coluna].fillna(mediana)
        df_aplicacao[coluna] = df_aplicacao[coluna].fillna(mediana)

    return df_treino, df_aplicacao, medianas


def aplicar_normalizacao(df_2018_2024, df_2018_2025, nome_base_2024,
                         nome_base_2025, standard_scaler_path,
                         minmax_scaler_path):
    # aplica Z-score nas continuas e MinMax em grid/laps
    # scaler sempre fitado na 2024 e aplicado nas duas bases

    df_2018_2024 = df_2018_2024.copy()
    df_2018_2025 = df_2018_2025.copy()

    # Garantir que colunas obrigatorias existem
    validar_colunas(
        df_2018_2024,
        ["grid_position", "laps"],
        nome_base_2024
    )

    validar_colunas(
        df_2018_2025,
        ["grid_position", "laps"],
        nome_base_2025
    )

    # pega só as colunas que realmente existem no df
    zscore_cols = selecionar_colunas_existentes(df_2018_2024, ZSCORE_COLUMNS)
    minmax_cols = selecionar_colunas_existentes(df_2018_2024, MINMAX_COLUMNS)

    # força conversão pra numérico antes de normalizar
    for coluna in zscore_cols + minmax_cols:
        df_2018_2024[coluna] = pd.to_numeric(
            df_2018_2024[coluna],
            errors="coerce"
        )

        df_2018_2025[coluna] = pd.to_numeric(
            df_2018_2025[coluna],
            errors="coerce"
        )

    # preenche nulos com mediana da base de treino antes de escalar
    df_2018_2024, df_2018_2025, medianas = preencher_nulos_com_mediana(
        df_2018_2024,
        df_2018_2025,
        zscore_cols + minmax_cols
    )

    # Z-score - fit na 2024, aplica nas duas
    standard_scaler = StandardScaler()

    if zscore_cols:
        standard_scaler.fit(df_2018_2024[zscore_cols])
        joblib.dump(standard_scaler, standard_scaler_path)

        zscore_2024 = standard_scaler.transform(df_2018_2024[zscore_cols])
        zscore_2025 = standard_scaler.transform(df_2018_2025[zscore_cols])

        # cria colunas novas com sufixo _zscore, mantendo o original
        for i, coluna in enumerate(zscore_cols):
            df_2018_2024[f"{coluna}_zscore"] = zscore_2024[:, i]
            df_2018_2025[f"{coluna}_zscore"] = zscore_2025[:, i]

    # MinMax - mesma lógica do Z-score
    minmax_scaler = MinMaxScaler()

    if minmax_cols:
        minmax_scaler.fit(df_2018_2024[minmax_cols])
        joblib.dump(minmax_scaler, minmax_scaler_path)

        minmax_2024 = minmax_scaler.transform(df_2018_2024[minmax_cols])
        minmax_2025 = minmax_scaler.transform(df_2018_2025[minmax_cols])

        for i, coluna in enumerate(minmax_cols):
            df_2018_2024[f"{coluna}_minmax"] = minmax_2024[:, i]
            df_2018_2025[f"{coluna}_minmax"] = minmax_2025[:, i]

    return df_2018_2024, df_2018_2025, zscore_cols, minmax_cols, medianas


def processar_base(input_2024, input_2025, output_2024, output_2025,
                   standard_scaler_path, minmax_scaler_path, rotulo):
    df_2018_2024 = pd.read_csv(input_2024)
    df_2018_2025 = pd.read_csv(input_2025)

    print(f"\nArquivos carregados com sucesso ({rotulo}).")
    print(f"{rotulo} encoded 2018-2024: {df_2018_2024.shape}")
    print(f"{rotulo} encoded 2018-2025: {df_2018_2025.shape}")

    normalizado_2018_2024, normalizado_2018_2025, zscore_cols, minmax_cols, medianas = aplicar_normalizacao(
        df_2018_2024,
        df_2018_2025,
        input_2024.name,
        input_2025.name,
        standard_scaler_path,
        minmax_scaler_path,
    )

    print(f"\nNormalização aplicada com sucesso ({rotulo}).")
    print("\nColunas com Z-score:")
    print(zscore_cols)
    print("\nColunas com MinMaxScaler:")
    print(minmax_cols)
    print("\nDimensões finais:")
    print(f"2018-2024: {normalizado_2018_2024.shape}")
    print(f"2018-2025: {normalizado_2018_2025.shape}")

    normalizado_2018_2024.to_csv(
        output_2024,
        index=False,
        encoding="utf-8-sig"
    )

    normalizado_2018_2025.to_csv(
        output_2025,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArquivos salvos com sucesso ({rotulo}):")
    print(output_2024)
    print(output_2025)
    print(standard_scaler_path)
    print(minmax_scaler_path)

    return {
        "rotulo": rotulo,
        "input_2024": input_2024,
        "input_2025": input_2025,
        "output_2024": output_2024,
        "output_2025": output_2025,
        "standard_scaler": standard_scaler_path,
        "minmax_scaler": minmax_scaler_path,
        "inicial_2024": df_2018_2024.shape,
        "inicial_2025": df_2018_2025.shape,
        "final_2024": normalizado_2018_2024.shape,
        "final_2025": normalizado_2018_2025.shape,
        "zscore_cols": zscore_cols,
        "minmax_cols": minmax_cols,
        "medianas": medianas,
    }


# processa os arquivos da etapa 03
resultados = []

resultados.append(processar_base(
    INPUT_FILE_2018_2024,
    INPUT_FILE_2018_2025,
    OUTPUT_FILE_2018_2024,
    OUTPUT_FILE_2018_2025,
    STANDARD_SCALER_HISTORICO,
    MINMAX_SCALER_HISTORICO,
    "Histórico enriquecido com FastF1"
))

resultados.append(processar_base(
    INPUT_BASE_LIMPA_2018_2024,
    INPUT_BASE_LIMPA_2018_2025,
    OUTPUT_BASE_LIMPA_2018_2024,
    OUTPUT_BASE_LIMPA_2018_2025,
    STANDARD_SCALER_BASE_LIMPA,
    MINMAX_SCALER_BASE_LIMPA,
    "Base histórica limpa"
))

print("\nEtapa 04 finalizada com sucesso.")
