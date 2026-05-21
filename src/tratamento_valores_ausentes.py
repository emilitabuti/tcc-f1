from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 05 - Tratamento de valores ausentes
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_normalizado_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_normalizado_2018_2025.csv"


# Arquivos de saída
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2025.csv"

REPORT_FILE = PROCESSED_DIR / "relatorio_05_tratamento_valores_ausentes.txt"
METHODOLOGY_FILE = DOCS_DIR / "metodologia_tratamento_valores_ausentes.md"


# Colunas de tempo de volta / setores
# Mediana do circuito naquele ano
TIME_COLUMNS = [
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_avg_sector1",
    "fastf1_avg_sector2",
    "fastf1_avg_sector3",
]


# Colunas que serão recalculadas com Z-score
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


# Colunas que serão recalculadas com MinMaxScaler
MINMAX_COLUMNS = [
    "grid_position",
    "laps",
]


# Mapeamento ordinal do composto
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
    return [
        coluna for coluna in df.columns
        if coluna.startswith("circuito_")
    ]


def criar_coluna_circuito_derivada(df):
    """
    Como depois do One-Hot o race_name/circuit_id vira várias colunas,
    esta função reconstrói uma coluna auxiliar chamada circuito_derivado.
    """
    df = df.copy()

    colunas_circuito = obter_colunas_circuito(df)

    if not colunas_circuito:
        raise ValueError(
            "Nenhuma coluna de circuito one-hot encontrada. "
            "Esperado colunas iniciando com 'circuito_'."
        )

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
    valores = series.dropna()

    if valores.empty:
        return np.nan

    return valores.mode().iloc[0]


def imputar_tempos_por_mediana_circuito_ano(df):
    """
    Regra:
    Tempos de volta e setores recebem a mediana do circuito naquele ano.
    Fallback:
    1. Mediana do circuito naquele ano
    2. Mediana do ano
    3. Mediana global
    """
    df = df.copy()

    time_cols = selecionar_colunas_existentes(df, TIME_COLUMNS)

    resumo = {}

    for coluna in time_cols:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

        nulos_antes = df[coluna].isna().sum()

        mediana_circuito_ano = df.groupby(
            ["season", "circuito_derivado"]
        )[coluna].transform("median")

        df[coluna] = df[coluna].fillna(mediana_circuito_ano)

        mediana_ano = df.groupby("season")[coluna].transform("median")
        df[coluna] = df[coluna].fillna(mediana_ano)

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
    """
    Regra:
    Composto de pneu recebe a moda da corrida.
    Corrida = season + round.
    """
    df = df.copy()

    resumo = {}

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

    moda_corrida = df.groupby(
        ["season", "round"]
    )["compound_normalizado"].transform(moda_segura)

    df["compound_normalizado"] = df["compound_normalizado"].fillna(moda_corrida)

    moda_global = moda_segura(df["compound_normalizado"])

    if pd.isna(moda_global):
        moda_global = "UNKNOWN"

    df["compound_normalizado"] = df["compound_normalizado"].fillna(moda_global)

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
    """
    Detecta colunas de qualifying, caso existam.
    No seu dataset atual, pelas colunas que você mandou, não há qualifying.
    Então o código pula essa etapa se não encontrar.
    """
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
    """
    Remove colunas antigas de normalização para evitar inconsistência
    após a imputação.
    """
    df = df.copy()

    colunas_para_remover = [
        coluna for coluna in df.columns
        if coluna.endswith("_zscore") or coluna.endswith("_minmax")
    ]

    df = df.drop(columns=colunas_para_remover, errors="ignore")

    return df

def imputar_qualifying_knn(df_2024, df_2025):
    """
    Aplica KNN Imputer nas colunas de qualifying, se elas existirem.
    O KNN é ajustado na base 2018-2024 e aplicado também na 2018-2025.
    """
    df_2024 = df_2024.copy()
    df_2025 = df_2025.copy()

    qualifying_cols = detectar_colunas_qualifying(df_2024)

    resumo = {
        "colunas_qualifying": qualifying_cols,
        "aplicado": False,
    }

    if not qualifying_cols:
        return df_2024, df_2025, resumo

    for coluna in qualifying_cols:
        df_2024[coluna] = pd.to_numeric(df_2024[coluna], errors="coerce")
        df_2025[coluna] = pd.to_numeric(df_2025[coluna], errors="coerce")

    nulos_antes_2024 = df_2024[qualifying_cols].isna().sum().to_dict()
    nulos_antes_2025 = df_2025[qualifying_cols].isna().sum().to_dict()

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

    # Preencher colunas auxiliares com mediana para ajudar o KNN
    for coluna in features_auxiliares:
        mediana = df_2024[coluna].median()

        if pd.isna(mediana):
            mediana = 0

        df_2024[coluna] = df_2024[coluna].fillna(mediana)
        df_2025[coluna] = df_2025[coluna].fillna(mediana)

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
    """
    Recalcula Z-score e MinMax após a imputação.
    """
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


def salvar_metodologia():
    texto = """# Tratamento de valores ausentes

## Objetivo

Esta etapa tem como objetivo tratar valores ausentes antes do uso da base em modelos de Machine Learning.

## Regras adotadas

### Tempos de volta

As colunas de tempos de volta e setores foram imputadas pela mediana do circuito naquele ano.

Foram consideradas as seguintes colunas:

- fastf1_avg_lap_time
- fastf1_best_lap_time
- fastf1_avg_sector1
- fastf1_avg_sector2
- fastf1_avg_sector3

Quando não havia mediana disponível para o circuito naquele ano, foram aplicados fallbacks:

1. Mediana do ano
2. Mediana global da coluna
3. Valor 0, caso não houvesse nenhuma mediana disponível

### Composto de pneu

O composto de pneu foi imputado pela moda da corrida, considerando a combinação de `season` e `round`.

Após a imputação, a variável `compound_ordinal` foi recalculada conforme a regra:

- SOFT = 3
- MEDIUM = 2
- HARD = 1
- INTERMEDIATE/WET/UNKNOWN = 0

### Qualifying

Para variáveis de qualifying, foi prevista imputação por KNN.

Caso não existam colunas de qualifying na base, a etapa é registrada como não aplicada.

Quando aplicada, a imputação por KNN usa apenas variáveis disponíveis antes ou durante a corrida. Variáveis pós-corrida, como posição final e pontos, são excluídas para evitar vazamento de informação.

## Reprocessamento da normalização

Após a imputação, as colunas normalizadas foram recalculadas para manter consistência entre os valores originais e os valores padronizados.

Os parâmetros de normalização foram ajustados com base na base 2018-2024 e aplicados também à base 2018-2025.

## Arquivos gerados

- historico_imputado_normalizado_2018_2024.csv
- historico_imputado_normalizado_2018_2025.csv
"""

    with open(METHODOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


# 1. Carregar arquivos
df_2024 = pd.read_csv(INPUT_FILE_2018_2024)
df_2025 = pd.read_csv(INPUT_FILE_2018_2025)

print("Arquivos carregados com sucesso.")
print(f"Base 2018-2024: {df_2024.shape}")
print(f"Base 2018-2025: {df_2025.shape}")


# 2. Validar colunas mínimas
colunas_obrigatorias = [
    "season",
    "round",
    "grid_position",
    "laps",
    "compound_ordinal",
]

validar_colunas(df_2024, colunas_obrigatorias, "historico_normalizado_2018_2024.csv")
validar_colunas(df_2025, colunas_obrigatorias, "historico_normalizado_2018_2025.csv")


# 3. Recriar coluna de circuito a partir do one-hot
df_2024, colunas_circuito_2024 = criar_coluna_circuito_derivada(df_2024)
df_2025, colunas_circuito_2025 = criar_coluna_circuito_derivada(df_2025)

print("\nColuna circuito_derivado criada com sucesso.")


# 4. Imputar tempos por mediana do circuito naquele ano
df_2024, resumo_tempos_2024 = imputar_tempos_por_mediana_circuito_ano(df_2024)
df_2025, resumo_tempos_2025 = imputar_tempos_por_mediana_circuito_ano(df_2025)

print("\nImputação de tempos concluída.")


# 5. Imputar composto pela moda da corrida
df_2024, resumo_composto_2024 = imputar_composto_por_moda_corrida(df_2024)
df_2025, resumo_composto_2025 = imputar_composto_por_moda_corrida(df_2025)

print("\nImputação de composto concluída.")


# 6. Imputar qualifying por KNN, se existir
df_2024, df_2025, resumo_qualifying = imputar_qualifying_knn(df_2024, df_2025)

if resumo_qualifying["aplicado"]:
    print("\nImputação KNN de qualifying aplicada.")
else:
    print("\nNenhuma coluna de qualifying encontrada. Etapa KNN não aplicada.")


# 7. Recalcular normalização
df_2024 = remover_colunas_normalizadas_antigas(df_2024)
df_2025 = remover_colunas_normalizadas_antigas(df_2025)
df_2024, df_2025, zscore_cols, minmax_cols = recalcular_normalizacao(df_2024, df_2025)

print("\nNormalização recalculada após imputação.")


# 8. Salvar arquivos finais
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


# 9. Salvar relatório
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 05 TRATAMENTO DE VALORES AUSENTES\n")
    f.write("=" * 70 + "\n\n")

    f.write("ARQUIVOS DE ENTRADA\n")
    f.write("-" * 70 + "\n")
    f.write(f"{INPUT_FILE_2018_2024}\n")
    f.write(f"{INPUT_FILE_2018_2025}\n\n")

    f.write("ARQUIVOS DE SAÍDA\n")
    f.write("-" * 70 + "\n")
    f.write(f"{OUTPUT_FILE_2018_2024}\n")
    f.write(f"{OUTPUT_FILE_2018_2025}\n\n")

    f.write("TEMPOS DE VOLTA - MEDIANA DO CIRCUITO NO ANO\n")
    f.write("-" * 70 + "\n")
    f.write("Base 2018-2024:\n")
    f.write(str(resumo_tempos_2024))
    f.write("\n\nBase 2018-2025:\n")
    f.write(str(resumo_tempos_2025))
    f.write("\n\n")

    f.write("COMPOSTO DE PNEU - MODA DA CORRIDA\n")
    f.write("-" * 70 + "\n")
    f.write("Base 2018-2024:\n")
    f.write(str(resumo_composto_2024))
    f.write("\n\nBase 2018-2025:\n")
    f.write(str(resumo_composto_2025))
    f.write("\n\n")

    f.write("QUALIFYING - KNN IMPUTER\n")
    f.write("-" * 70 + "\n")
    f.write(str(resumo_qualifying))
    f.write("\n\n")

    f.write("NORMALIZAÇÃO RECALCULADA\n")
    f.write("-" * 70 + "\n")
    f.write("Colunas Z-score:\n")
    for coluna in zscore_cols:
        f.write(f"- {coluna} -> {coluna}_zscore\n")

    f.write("\nColunas MinMaxScaler:\n")
    for coluna in minmax_cols:
        f.write(f"- {coluna} -> {coluna}_minmax\n")

salvar_metodologia()

print("\nRelatório salvo em:")
print(REPORT_FILE)

print("\nDocumentação metodológica salva em:")
print(METHODOLOGY_FILE)

print("\nEtapa 05 finalizada com sucesso.")
