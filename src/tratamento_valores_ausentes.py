from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 05 - Tratamento de valores ausentes
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_normalizado_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_normalizado_2018_2025.csv"


# Arquivos de saida
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2025.csv"


# tempos de volta/setores - vão receber mediana do circuito no ano
TIME_COLUMNS = [
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_avg_sector1",
    "fastf1_avg_sector2",
    "fastf1_avg_sector3",
]


# colunas que serão recalculadas com Z-score depois da imputação
ZSCORE_COLUMNS = [
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


# colunas que serão recalculadas com MinMaxScaler
MINMAX_COLUMNS = [
    "grid_position",
    "laps",
]


# mapeamento ordinal do composto - mesmo do encoding
COMPOUND_ORDINAL_MAP = {
    "SOFT": 3,
    "MEDIUM": 2,
    "HARD": 1,
    "INTERMEDIATE": 0,
    "WET": 0,
    "UNKNOWN": 0,
}


# Funções auxiliares
def validar_colunas(df, colunas_obrigatorias, nome_base):
    colunas_ausentes = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"As seguintes colunas estão ausentes em {nome_base}: "
            f"{colunas_ausentes}"
        )


def selecionar_colunas_existentes(df, colunas):
    return [coluna for coluna in colunas if coluna in df.columns]


def obter_colunas_circuito(df):
    # pega todas as colunas que vieram do one-hot de circuito
    return [
        coluna for coluna in df.columns
        if coluna.startswith("circuito_")
    ]


def criar_coluna_circuito_derivada(df):
    # reconstrói circuito_derivado a partir das colunas one-hot
    df = df.copy()

    colunas_circuito = obter_colunas_circuito(df)

    if not colunas_circuito:
        raise ValueError(
            "Nenhuma coluna de circuito one-hot encontrada. "
            "Esperado colunas iniciando com 'circuito_'."
        )

    # reconstrói o nome do circuito a partir do one-hot - pega o que tem valor 1
    df["circuito_derivado"] = (
        df[colunas_circuito]
        .idxmax(axis=1)
        .str.replace("circuito_", "", regex=False)
    )

    return df, colunas_circuito


def normalizar_composto(valor):
    if pd.isna(valor):
        return np.nan

    return str(valor).strip().upper()


def moda_segura(series):
    # retorna a moda ignorando nulos - se vazio retorna nan
    valores = series.dropna()

    if valores.empty:
        return np.nan

    return valores.mode().iloc[0]


def imputar_tempos_por_mediana_circuito_ano(df):
    # imputa tempos de volta com mediana do circuito/ano, com fallback global
    df = df.copy()

    time_cols = selecionar_colunas_existentes(df, TIME_COLUMNS)

    resumo = {}

    for coluna in time_cols:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

        nulos_antes = df[coluna].isna().sum()

        # primeiro tenta mediana do circuito no ano
        mediana_circuito_ano = df.groupby(
            ["season", "circuito_derivado"]
        )[coluna].transform("median")

        df[coluna] = df[coluna].fillna(mediana_circuito_ano)

        # se ainda sobrou nulo, usa mediana do ano
        mediana_ano = df.groupby("season")[coluna].transform("median")
        df[coluna] = df[coluna].fillna(mediana_ano)

        # fallback final: mediana global
        mediana_global = df[coluna].median()

        if pd.isna(mediana_global):
            mediana_global = 0

        df[coluna] = df[coluna].fillna(mediana_global)

        nulos_depois = df[coluna].isna().sum()

        resumo[coluna] = {
            "nulos_antes": int(nulos_antes),
            "nulos_depois": int(nulos_depois),
            "mediana_global_fallback": float(mediana_global),
        }

    return df, resumo


def imputar_composto_por_moda_corrida(df):
    # imputa composto de pneu com a moda da corrida (season + round)
    df = df.copy()

    resumo = {}

    # garante que a coluna compound_normalizado existe antes de imputar
    if "compound_normalizado" not in df.columns:
        if "fastf1_first_compound" in df.columns:
            df["compound_normalizado"] = df["fastf1_first_compound"].apply(normalizar_composto)
        elif "fastf1_main_compound" in df.columns:
            df["compound_normalizado"] = df["fastf1_main_compound"].apply(normalizar_composto)
        else:
            raise ValueError(
                "Nenhuma coluna de composto encontrada. "
                "Esperado compound_normalizado, fastf1_first_compound ou fastf1_main_compound."
            )

    df["compound_normalizado"] = df["compound_normalizado"].apply(normalizar_composto)

    nulos_antes = df["compound_normalizado"].isna().sum()

    # pega a moda do composto em cada corrida e usa pra preencher os nulos
    moda_corrida = df.groupby(
        ["season", "round"]
    )["compound_normalizado"].transform(moda_segura)

    df["compound_normalizado"] = df["compound_normalizado"].fillna(moda_corrida)

    # se ainda tiver nulo, usa a moda geral
    moda_global = moda_segura(df["compound_normalizado"])

    if pd.isna(moda_global):
        moda_global = "UNKNOWN"

    df["compound_normalizado"] = df["compound_normalizado"].fillna(moda_global)

    # recalcula o ordinal depois da imputação
    df["compound_ordinal"] = (
        df["compound_normalizado"]
        .map(COMPOUND_ORDINAL_MAP)
        .fillna(0)
        .astype(int)
    )

    nulos_depois = df["compound_normalizado"].isna().sum()

    resumo["compound_normalizado"] = {
        "nulos_antes": int(nulos_antes),
        "nulos_depois": int(nulos_depois),
        "moda_global_fallback": moda_global,
    }

    return df, resumo


def detectar_colunas_qualifying(df):
    # detecta colunas de qualifying no dataset, se existirem
    palavras_chave = [
        "qualifying",
        "qualify",
        "quali",
        "q1",
        "q2",
        "q3",
        "qualifying_position",
    ]

    colunas = []

    for coluna in df.columns:
        coluna_lower = coluna.lower()

        if any(palavra in coluna_lower for palavra in palavras_chave):
            if pd.api.types.is_numeric_dtype(df[coluna]) or coluna_lower in ["q1", "q2", "q3"]:
                colunas.append(coluna)

    return sorted(set(colunas))

def remover_colunas_normalizadas_antigas(df):
    # remove _zscore e _minmax pra recalcular do zero depois da imputação
    df = df.copy()

    # joga fora as colunas _zscore e _minmax pra recalcular do zero
    colunas_para_remover = [
        coluna for coluna in df.columns
        if coluna.endswith("_zscore") or coluna.endswith("_minmax")
    ]

    df = df.drop(columns=colunas_para_remover, errors="ignore")

    return df

def imputar_qualifying_knn(df_2024, df_2025):
    # KNN Imputer nas colunas de qualifying - fit na 2024, aplica na 2025
    df_2024 = df_2024.copy()
    df_2025 = df_2025.copy()

    qualifying_cols = detectar_colunas_qualifying(df_2024)

    resumo = {
        "colunas_qualifying": qualifying_cols,
        "aplicado": False,
    }

    # se não tiver coluna de qualifying, pula tudo
    if not qualifying_cols:
        return df_2024, df_2025, resumo

    for coluna in qualifying_cols:
        df_2024[coluna] = pd.to_numeric(df_2024[coluna], errors="coerce")
        df_2025[coluna] = pd.to_numeric(df_2025[coluna], errors="coerce")

    nulos_antes_2024 = df_2024[qualifying_cols].isna().sum().to_dict()
    nulos_antes_2025 = df_2025[qualifying_cols].isna().sum().to_dict()

    # features auxiliares que ajudam o KNN a estimar o qualifying
    features_auxiliares = [
        "season",
        "round",
        "grid_position",
        "laps",
        "compound_ordinal",
        "fastf1_avg_lap_time",
        "fastf1_best_lap_time",
    ]

    features_auxiliares = selecionar_colunas_existentes(df_2024, features_auxiliares)

    colunas_knn = sorted(set(qualifying_cols + features_auxiliares))

    for coluna in colunas_knn:
        df_2024[coluna] = pd.to_numeric(df_2024[coluna], errors="coerce")
        df_2025[coluna] = pd.to_numeric(df_2025[coluna], errors="coerce")

    # preenche as auxiliares com mediana antes de passar pro KNN
    for coluna in features_auxiliares:
        mediana = df_2024[coluna].median()

        if pd.isna(mediana):
            mediana = 0

        df_2024[coluna] = df_2024[coluna].fillna(mediana)
        df_2025[coluna] = df_2025[coluna].fillna(mediana)

    # KNN fitado na 2024, aplicado nas duas - 5 vizinhos
    imputer = KNNImputer(n_neighbors=5)

    imputer.fit(df_2024[colunas_knn])

    imputado_2024 = imputer.transform(df_2024[colunas_knn])
    imputado_2025 = imputer.transform(df_2025[colunas_knn])

    df_2024[colunas_knn] = imputado_2024
    df_2025[colunas_knn] = imputado_2025

    nulos_depois_2024 = df_2024[qualifying_cols].isna().sum().to_dict()
    nulos_depois_2025 = df_2025[qualifying_cols].isna().sum().to_dict()

    resumo = {
        "colunas_qualifying": qualifying_cols,
        "aplicado": True,
        "nulos_antes_2024": nulos_antes_2024,
        "nulos_depois_2024": nulos_depois_2024,
        "nulos_antes_2025": nulos_antes_2025,
        "nulos_depois_2025": nulos_depois_2025,
    }

    return df_2024, df_2025, resumo


def recalcular_normalizacao(df_2024, df_2025):
    # refaz Z-score e MinMax depois da imputacao - parametros sempre da base 2024
    df_2024 = df_2024.copy()
    df_2025 = df_2025.copy()

    zscore_cols = selecionar_colunas_existentes(df_2024, ZSCORE_COLUMNS)
    minmax_cols = selecionar_colunas_existentes(df_2024, MINMAX_COLUMNS)

    for coluna in zscore_cols + minmax_cols:
        df_2024[coluna] = pd.to_numeric(df_2024[coluna], errors="coerce")
        df_2025[coluna] = pd.to_numeric(df_2025[coluna], errors="coerce")

        mediana = df_2024[coluna].median()

        if pd.isna(mediana):
            mediana = 0

        df_2024[coluna] = df_2024[coluna].fillna(mediana)
        df_2025[coluna] = df_2025[coluna].fillna(mediana)

    scaler_z = StandardScaler()

    if zscore_cols:
        scaler_z.fit(df_2024[zscore_cols])

        z_2024 = scaler_z.transform(df_2024[zscore_cols])
        z_2025 = scaler_z.transform(df_2025[zscore_cols])

        for i, coluna in enumerate(zscore_cols):
            df_2024[f"{coluna}_zscore"] = z_2024[:, i]
            df_2025[f"{coluna}_zscore"] = z_2025[:, i]

    scaler_minmax = MinMaxScaler()

    if minmax_cols:
        scaler_minmax.fit(df_2024[minmax_cols])

        mm_2024 = scaler_minmax.transform(df_2024[minmax_cols])
        mm_2025 = scaler_minmax.transform(df_2025[minmax_cols])

        for i, coluna in enumerate(minmax_cols):
            df_2024[f"{coluna}_minmax"] = mm_2024[:, i]
            df_2025[f"{coluna}_minmax"] = mm_2025[:, i]

    return df_2024, df_2025, zscore_cols, minmax_cols


# carrega os arquivos
df_2024 = pd.read_csv(INPUT_FILE_2018_2024)
df_2025 = pd.read_csv(INPUT_FILE_2018_2025)

print("Arquivos carregados com sucesso.")
print(f"Base 2018-2024: {df_2024.shape}")
print(f"Base 2018-2025: {df_2025.shape}")


# valida colunas mínimas antes de começar
colunas_obrigatorias = [
    "season",
    "round",
    "grid_position",
    "laps",
    "compound_ordinal",
]

validar_colunas(df_2024, colunas_obrigatorias, "historico_normalizado_2018_2024.csv")
validar_colunas(df_2025, colunas_obrigatorias, "historico_normalizado_2018_2025.csv")


# recria coluna de circuito a partir das colunas one-hot
df_2024, colunas_circuito_2024 = criar_coluna_circuito_derivada(df_2024)
df_2025, colunas_circuito_2025 = criar_coluna_circuito_derivada(df_2025)

print("\nColuna circuito_derivado criada com sucesso.")


# imputa tempos pela mediana do circuito naquele ano
df_2024, resumo_tempos_2024 = imputar_tempos_por_mediana_circuito_ano(df_2024)
df_2025, resumo_tempos_2025 = imputar_tempos_por_mediana_circuito_ano(df_2025)

print("\nImputação de tempos concluída.")


# imputa composto pela moda da corrida
df_2024, resumo_composto_2024 = imputar_composto_por_moda_corrida(df_2024)
df_2025, resumo_composto_2025 = imputar_composto_por_moda_corrida(df_2025)

print("\nImputação de composto concluída.")


# imputa qualifying por KNN, se existir - normalmente não tem e pula
df_2024, df_2025, resumo_qualifying = imputar_qualifying_knn(df_2024, df_2025)

if resumo_qualifying["aplicado"]:
    print("\nImputação KNN de qualifying aplicada.")
else:
    print("\nNenhuma coluna de qualifying encontrada. Etapa KNN não aplicada.")


# remove as colunas normalizadas antigas e recalcula tudo
df_2024 = remover_colunas_normalizadas_antigas(df_2024)
df_2025 = remover_colunas_normalizadas_antigas(df_2025)
df_2024, df_2025, zscore_cols, minmax_cols = recalcular_normalizacao(df_2024, df_2025)

print("\nNormalização recalculada após imputação.")


# salva os arquivos finais
df_2024.to_csv(
    OUTPUT_FILE_2018_2024,
    index=False,
    encoding="utf-8-sig"
)

df_2025.to_csv(
    OUTPUT_FILE_2018_2025,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos salvos com sucesso:")
print(OUTPUT_FILE_2018_2024)
print(OUTPUT_FILE_2018_2025)

print("\nEtapa 05 finalizada com sucesso.")
