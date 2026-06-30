from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 06 - Tratamento de outliers
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2025.csv"


# Arquivos de saída
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_outliers_tratados_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_outliers_tratados_2018_2025.csv"

OUTLIERS_REMOVIDOS_2018_2024 = PROCESSED_DIR / "outliers_removidos_2018_2024.csv"
OUTLIERS_REMOVIDOS_2018_2025 = PROCESSED_DIR / "outliers_removidos_2018_2025.csv"


# colunas de tempo que serão avaliadas pra outlier
# critério: acima de 3 desvios padrão da média do circuito
OUTLIER_COLUMNS = [
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_avg_sector1",
    "fastf1_avg_sector2",
    "fastf1_avg_sector3",
]


# colunas pra recalcular Z-score depois de remover os espurios
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


# colunas pra recalcular MinMax
MINMAX_COLUMNS = [
    "grid_position",
    "laps",
]


# se o status contem alguma dessas palavras, o outlier provavelmente tem explicacao mecanica
FALHA_MECANICA_KEYWORDS = [
    "engine",
    "gearbox",
    "transmission",
    "clutch",
    "hydraulics",
    "electrical",
    "electronics",
    "ers",
    "power unit",
    "brakes",
    "brake",
    "suspension",
    "radiator",
    "oil",
    "fuel",
    "turbo",
    "exhaust",
    "mechanical",
    "overheating",
    "puncture",
    "tyre",
    "wheel",
]

# compostos de chuva - corrida molhada justifica tempos fora do padrao seco
WET_COMPOUNDS = {
    "WET",
    "INTERMEDIATE",
}


# Funcoes auxiliares
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
    # reconstrói circuito_derivado a partir das colunas one-hot
    df = df.copy()

    # se já existe, não precisa recriar
    if "circuito_derivado" in df.columns:
        return df

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

    return df


def preparar_safety_car_flag(df):
    # garante que safety_car_flag existe - cria com 0 se nao tiver
    df = df.copy()

    # ainda nao tem dado real de safety car, entao coloca 0 por enquanto
    if "safety_car_flag" not in df.columns:
        df["safety_car_flag"] = 0

    df["safety_car_flag"] = pd.to_numeric(
        df["safety_car_flag"],
        errors="coerce"
    ).fillna(0).astype(int)

    return df


def status_indica_falha_mecanica(status):
    # checa se o status do piloto menciona algum problema mecânico
    if pd.isna(status):
        return False

    status_normalizado = str(status).strip().lower()

    return any(
        palavra in status_normalizado
        for palavra in FALHA_MECANICA_KEYWORDS
    )


def preparar_flags_contexto_corrida(df):
    # cria flags de contexto: wet_compound_flag e corrida_chuva_flag
    df = df.copy()

    # descobre o composto a partir de qualquer coluna disponível
    if "compound_normalizado" in df.columns:
        composto = df["compound_normalizado"].astype(str).str.strip().str.upper()
    elif "fastf1_first_compound" in df.columns:
        composto = df["fastf1_first_compound"].astype(str).str.strip().str.upper()
    else:
        composto = pd.Series("", index=df.index)

    df["wet_compound_flag"] = composto.isin(WET_COMPOUNDS).astype(int)

    # se qualquer piloto na corrida usou pneu de chuva, marca a corrida inteira
    df["corrida_chuva_flag"] = (
        df.groupby(["season", "round"])["wet_compound_flag"]
        .transform("max")
        .fillna(0)
        .astype(int)
    )

    return df


def marcar_outlier_espurio_estrito(df, outlier_cols):
    # marca como espúrio só o que é tecnicamente inválido (nulo ou <= 0)
    colunas_existentes = selecionar_colunas_existentes(df, outlier_cols)
    valores_invalidos = pd.Series(False, index=df.index)

    # tempo de volta nulo ou <= 0 é tecnicamente impossível
    for coluna in colunas_existentes:
        valores = pd.to_numeric(df[coluna], errors="coerce")
        valores_invalidos = valores_invalidos | valores.isna() | (valores <= 0)

    return (df["outlier_flag"] == 1) & valores_invalidos


def marcar_outliers_por_circuito(df):
    # marca outliers com critério de 3 desvios padrão por circuito
    df = df.copy()

    outlier_cols = selecionar_colunas_existentes(df, OUTLIER_COLUMNS)

    for coluna in outlier_cols:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df["outlier_flag"] = 0
    df["outlier_colunas"] = ""

    for coluna in outlier_cols:
        # média e desvio padrão calculados por circuito
        media_circuito = df.groupby("circuito_derivado")[coluna].transform("mean")
        desvio_circuito = df.groupby("circuito_derivado")[coluna].transform("std")

        limite_superior = media_circuito + (3 * desvio_circuito)

        outlier_coluna = (
            df[coluna].notna()
            & desvio_circuito.notna()
            & (desvio_circuito > 0)
            & (df[coluna] > limite_superior)
        )

        df.loc[outlier_coluna, "outlier_flag"] = 1

        # registra quais colunas dispararam o outlier
        df.loc[outlier_coluna, "outlier_colunas"] = (
            df.loc[outlier_coluna, "outlier_colunas"]
            + coluna
            + ";"
        )

    df.loc[df["outlier_flag"] == 0, "outlier_colunas"] = "NONE"

    return df, outlier_cols


def classificar_outliers(df, outlier_cols):
    # classifica outliers em legítimos, espúrios e revisão
    df = df.copy()

    if "dnf_car_flag" not in df.columns:
        df["dnf_car_flag"] = 0

    df["dnf_car_flag"] = pd.to_numeric(
        df["dnf_car_flag"],
        errors="coerce"
    ).fillna(0).astype(int)

    df["status_falha_mecanica_flag"] = df["status"].apply(
        status_indica_falha_mecanica
    ).astype(int)

    df["outlier_tipo"] = "nao_outlier"
    cond_espurio = marcar_outlier_espurio_estrito(df, outlier_cols)

    # outlier legítimo = tem uma explicação real (chuva, falha, safety car)
    cond_legitimo = (
        (df["outlier_flag"] == 1)
        & (
            (df["safety_car_flag"] == 1)
            | (df["dnf_car_flag"] == 1)
            | (df["status_falha_mecanica_flag"] == 1)
            | (df["corrida_chuva_flag"] == 1)
        )
    )

    # outlier revisao = valor extremo mas sem explicacao clara - mantem com flag
    cond_revisao = (
        (df["outlier_flag"] == 1)
        & ~cond_legitimo
        & ~cond_espurio
    )

    df.loc[cond_legitimo, "outlier_tipo"] = "outlier_legitimo"
    df.loc[cond_revisao, "outlier_tipo"] = "outlier_revisao"
    df.loc[cond_espurio, "outlier_tipo"] = "outlier_espurio"

    df["outlier_legitimo_flag"] = (
        df["outlier_tipo"] == "outlier_legitimo"
    ).astype(int)

    df["outlier_revisao_flag"] = (
        df["outlier_tipo"] == "outlier_revisao"
    ).astype(int)

    df["outlier_espurio_flag"] = (
        df["outlier_tipo"] == "outlier_espurio"
    ).astype(int)

    return df


def remover_outliers_espurios(df):
    # remove só os espúrios - legítimos e revisão ficam com flag
    df = df.copy()

    removidos = df[df["outlier_tipo"] == "outlier_espurio"].copy()
    tratado = df[df["outlier_tipo"] != "outlier_espurio"].copy()

    return tratado, removidos


def remover_colunas_normalizadas_antigas(df):
    # joga fora as colunas de normalização antiga pra recalcular do zero
    df = df.copy()

    colunas_para_remover = [
        coluna for coluna in df.columns
        if coluna.endswith("_zscore") or coluna.endswith("_minmax")
    ]

    df = df.drop(columns=colunas_para_remover, errors="ignore")

    return df


def recalcular_normalizacao(df_2024, df_2025):
    # refaz Z-score e MinMax depois de remover os outliers espúrios
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


# valida colunas mínimas antes de qualquer processamento
colunas_obrigatorias = [
    "season",
    "round",
    "grid_position",
    "laps",
    "status",
]

validar_colunas(df_2024, colunas_obrigatorias, "historico_imputado_normalizado_2018_2024.csv")
validar_colunas(df_2025, colunas_obrigatorias, "historico_imputado_normalizado_2018_2025.csv")


# prepara colunas auxiliares - circuito derivado, safety car e flags de chuva
df_2024 = criar_coluna_circuito_derivada(df_2024)
df_2025 = criar_coluna_circuito_derivada(df_2025)

df_2024 = preparar_safety_car_flag(df_2024)
df_2025 = preparar_safety_car_flag(df_2025)

df_2024 = preparar_flags_contexto_corrida(df_2024)
df_2025 = preparar_flags_contexto_corrida(df_2025)

print("\nColunas auxiliares preparadas com sucesso.")


# marca quais registros são outlier (> 3 desvios da média do circuito)
df_2024, outlier_cols_2024 = marcar_outliers_por_circuito(df_2024)
df_2025, outlier_cols_2025 = marcar_outliers_por_circuito(df_2025)

print("\nOutliers marcados com sucesso.")


# classifica cada outlier: legítimo, espúrio ou pra revisão
df_2024 = classificar_outliers(df_2024, outlier_cols_2024)
df_2025 = classificar_outliers(df_2025, outlier_cols_2025)

print("\nOutliers classificados com sucesso.")


# remove so os espurios - os legitimos ficam na base com flag
df_2024_tratado, outliers_removidos_2024 = remover_outliers_espurios(df_2024)
df_2025_tratado, outliers_removidos_2025 = remover_outliers_espurios(df_2025)

print("\nOutliers espúrios removidos.")
print(f"Removidos 2018-2024: {len(outliers_removidos_2024)}")
print(f"Removidos 2018-2025: {len(outliers_removidos_2025)}")


# recalcula normalização depois da remoção
df_2024_tratado = remover_colunas_normalizadas_antigas(df_2024_tratado)
df_2025_tratado = remover_colunas_normalizadas_antigas(df_2025_tratado)

df_2024_tratado, df_2025_tratado, zscore_cols, minmax_cols = recalcular_normalizacao(
    df_2024_tratado,
    df_2025_tratado
)

print("\nNormalização recalculada após tratamento de outliers.")


# salva os arquivos finais - tratados e os removidos separadamente
df_2024_tratado.to_csv(
    OUTPUT_FILE_2018_2024,
    index=False,
    encoding="utf-8-sig"
)

df_2025_tratado.to_csv(
    OUTPUT_FILE_2018_2025,
    index=False,
    encoding="utf-8-sig"
)

outliers_removidos_2024.to_csv(
    OUTLIERS_REMOVIDOS_2018_2024,
    index=False,
    encoding="utf-8-sig"
)

outliers_removidos_2025.to_csv(
    OUTLIERS_REMOVIDOS_2018_2025,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos salvos com sucesso:")
print(OUTPUT_FILE_2018_2024)
print(OUTPUT_FILE_2018_2025)
print(OUTLIERS_REMOVIDOS_2018_2024)
print(OUTLIERS_REMOVIDOS_2018_2025)

print("\nEtapa 06 finalizada com sucesso.")
