from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 06 - Tratamento de outliers
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_imputado_normalizado_2018_2025.csv"


# Arquivos de saída
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_outliers_tratados_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_outliers_tratados_2018_2025.csv"

OUTLIERS_REMOVIDOS_2018_2024 = PROCESSED_DIR / "outliers_removidos_2018_2024.csv"
OUTLIERS_REMOVIDOS_2018_2025 = PROCESSED_DIR / "outliers_removidos_2018_2025.csv"

REPORT_FILE = PROCESSED_DIR / "relatorio_06_tratamento_outliers.txt"
METHODOLOGY_FILE = DOCS_DIR / "metodologia_tratamento_outliers.md"


# Colunas avaliadas para outlier
# Critério: acima de 3 desvios padrão da média por circuito

OUTLIER_COLUMNS = [
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_avg_sector1",
    "fastf1_avg_sector2",
    "fastf1_avg_sector3",
]


# Colunas para recalcular Z-score depois da remoção

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


# Colunas para recalcular MinMax

MINMAX_COLUMNS = [
    "grid_position",
    "laps",
]


# Palavras que indicam outlier legítimo por falha mecânica

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

WET_COMPOUNDS = {
    "WET",
    "INTERMEDIATE",
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
    Reconstrói uma coluna auxiliar de circuito a partir das colunas One-Hot.
    """
    df = df.copy()

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
    """
    Garante que exista a coluna safety_car_flag.

    Se ela não existir, cria com 0.
    Isso evita erro no código, mas significa que ainda não há informação real
    de Safety Car integrada na base.
    """
    df = df.copy()

    if "safety_car_flag" not in df.columns:
        df["safety_car_flag"] = 0

    df["safety_car_flag"] = pd.to_numeric(
        df["safety_car_flag"],
        errors="coerce"
    ).fillna(0).astype(int)

    return df


def status_indica_falha_mecanica(status):
    if pd.isna(status):
        return False

    status_normalizado = str(status).strip().lower()

    return any(
        palavra in status_normalizado
        for palavra in FALHA_MECANICA_KEYWORDS
    )


def preparar_flags_contexto_corrida(df):
    """
    Cria flags de contexto que ajudam a diferenciar evento real de erro de dado.
    Uma corrida com qualquer composto WET/INTERMEDIATE é tratada como corrida de chuva.
    """
    df = df.copy()

    if "compound_normalizado" in df.columns:
        composto = df["compound_normalizado"].astype(str).str.strip().str.upper()
    elif "fastf1_first_compound" in df.columns:
        composto = df["fastf1_first_compound"].astype(str).str.strip().str.upper()
    else:
        composto = pd.Series("", index=df.index)

    df["wet_compound_flag"] = composto.isin(WET_COMPOUNDS).astype(int)

    df["corrida_chuva_flag"] = (
        df.groupby(["season", "round"])["wet_compound_flag"]
        .transform("max")
        .fillna(0)
        .astype(int)
    )

    return df


def marcar_outlier_espurio_estrito(df, outlier_cols):
    """
    Marca apenas valores tecnicamente inválidos como espúrios.
    Valores extremos plausíveis ficam para revisão e não são removidos automaticamente.
    """
    colunas_existentes = selecionar_colunas_existentes(df, outlier_cols)
    valores_invalidos = pd.Series(False, index=df.index)

    for coluna in colunas_existentes:
        valores = pd.to_numeric(df[coluna], errors="coerce")
        valores_invalidos = valores_invalidos | valores.isna() | (valores <= 0)

    return (df["outlier_flag"] == 1) & valores_invalidos


def marcar_outliers_por_circuito(df):
    """
    Marca outliers usando critério:
    valor acima de 3 desvios padrão da média por circuito.
    """
    df = df.copy()

    outlier_cols = selecionar_colunas_existentes(df, OUTLIER_COLUMNS)

    for coluna in outlier_cols:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df["outlier_flag"] = 0
    df["outlier_colunas"] = ""

    for coluna in outlier_cols:
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

        df.loc[outlier_coluna, "outlier_colunas"] = (
            df.loc[outlier_coluna, "outlier_colunas"]
            + coluna
            + ";"
        )

    df.loc[df["outlier_flag"] == 0, "outlier_colunas"] = "NONE"

    return df, outlier_cols


def classificar_outliers(df, outlier_cols):
    """
    Classifica outlier como:
    - nao_outlier
    - outlier_legitimo
    - outlier_revisao
    - outlier_espurio

    Regras:
    - Se não é outlier: nao_outlier
    - Se safety_car_flag = 1: outlier_legitimo
    - Se dnf_car_flag = 1: outlier_legitimo
    - Se status indica falha mecânica: outlier_legitimo
    - Se corrida teve WET/INTERMEDIATE: outlier_legitimo
    - Se valor é tecnicamente inválido: outlier_espurio
    - Caso contrário: outlier_revisao
    """
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

    cond_legitimo = (
        (df["outlier_flag"] == 1)
        & (
            (df["safety_car_flag"] == 1)
            | (df["dnf_car_flag"] == 1)
            | (df["status_falha_mecanica_flag"] == 1)
            | (df["corrida_chuva_flag"] == 1)
        )
    )

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
    """
    Remove apenas outliers espúrios.
    Mantém outliers legítimos com flag.
    """
    df = df.copy()

    removidos = df[df["outlier_tipo"] == "outlier_espurio"].copy()
    tratado = df[df["outlier_tipo"] != "outlier_espurio"].copy()

    return tratado, removidos


def remover_colunas_normalizadas_antigas(df):
    """
    Remove colunas antigas de normalização para recalcular após a remoção de outliers.
    """
    df = df.copy()

    colunas_para_remover = [
        coluna for coluna in df.columns
        if coluna.endswith("_zscore") or coluna.endswith("_minmax")
    ]

    df = df.drop(columns=colunas_para_remover, errors="ignore")

    return df


def recalcular_normalizacao(df_2024, df_2025):
    """
    Recalcula Z-score e MinMax após tratamento de outliers.
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
    texto = """# Tratamento de outliers

## Objetivo

Esta etapa tem como objetivo identificar e tratar valores extremos nas variáveis de tempo de volta e setores.

## Critério adotado

Foi adotado o critério de valores acima de 3 desvios padrão da média por circuito.

Foram avaliadas as seguintes colunas:

- fastf1_avg_lap_time
- fastf1_best_lap_time
- fastf1_avg_sector1
- fastf1_avg_sector2
- fastf1_avg_sector3

## Classificação dos outliers

Os outliers foram classificados em tres grupos:

### Outliers legítimos

São valores extremos que podem ser explicados por eventos reais da corrida, como Safety Car, falha mecânica ou corrida com pneus WET/INTERMEDIATE.

Esses registros foram mantidos na base e marcados com flags.

### Outliers espúrios

São valores extremos tecnicamente inválidos, como tempos ausentes ou menores/iguais a zero.

Esses registros foram removidos da base tratada.

### Outliers para revisão

São valores extremos plausíveis, mas sem evidência suficiente para remoção automática.

Esses registros foram mantidos na base com flag, seguindo a decisão metodológica de não descartar eventos reais de corrida sem confirmação.

## Observação sobre Safety Car

A base atual não possui uma variável real de Safety Car integrada. Por isso, quando a coluna `safety_car_flag` não existe, ela é criada com valor 0.

Caso dados de Safety Car sejam integrados posteriormente, essa flag poderá ser usada para preservar outliers legítimos associados a esse evento.

## Reprocessamento da normalização

Após a remoção dos outliers espúrios, as colunas normalizadas foram recalculadas.

Os parâmetros de normalização foram ajustados com base na base 2018-2024 e aplicados também à base 2018-2025.

## Arquivos gerados

- historico_outliers_tratados_2018_2024.csv
- historico_outliers_tratados_2018_2025.csv
- outliers_removidos_2018_2024.csv
- outliers_removidos_2018_2025.csv
"""

    with open(METHODOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


# ============================================================
# 1. Carregar arquivos
# ============================================================

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
    "status",
]

validar_colunas(df_2024, colunas_obrigatorias, "historico_imputado_normalizado_2018_2024.csv")
validar_colunas(df_2025, colunas_obrigatorias, "historico_imputado_normalizado_2018_2025.csv")


# 3. Preparar colunas auxiliares
df_2024 = criar_coluna_circuito_derivada(df_2024)
df_2025 = criar_coluna_circuito_derivada(df_2025)

df_2024 = preparar_safety_car_flag(df_2024)
df_2025 = preparar_safety_car_flag(df_2025)

df_2024 = preparar_flags_contexto_corrida(df_2024)
df_2025 = preparar_flags_contexto_corrida(df_2025)

print("\nColunas auxiliares preparadas com sucesso.")


# 4. Marcar outliers por circuito
df_2024, outlier_cols_2024 = marcar_outliers_por_circuito(df_2024)
df_2025, outlier_cols_2025 = marcar_outliers_por_circuito(df_2025)

print("\nOutliers marcados com sucesso.")


# 5. Classificar outliers
df_2024 = classificar_outliers(df_2024, outlier_cols_2024)
df_2025 = classificar_outliers(df_2025, outlier_cols_2025)

print("\nOutliers classificados com sucesso.")


# 6. Remover apenas outliers espúrios
df_2024_tratado, outliers_removidos_2024 = remover_outliers_espurios(df_2024)
df_2025_tratado, outliers_removidos_2025 = remover_outliers_espurios(df_2025)

print("\nOutliers espúrios removidos.")
print(f"Removidos 2018-2024: {len(outliers_removidos_2024)}")
print(f"Removidos 2018-2025: {len(outliers_removidos_2025)}")


# 7. Recalcular normalização
df_2024_tratado = remover_colunas_normalizadas_antigas(df_2024_tratado)
df_2025_tratado = remover_colunas_normalizadas_antigas(df_2025_tratado)

df_2024_tratado, df_2025_tratado, zscore_cols, minmax_cols = recalcular_normalizacao(
    df_2024_tratado,
    df_2025_tratado
)

print("\nNormalização recalculada após tratamento de outliers.")



# 8. Salvar arquivos finais
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


# 9. Salvar relatório
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 06 TRATAMENTO DE OUTLIERS\n")
    f.write("=" * 70 + "\n\n")

    f.write("ARQUIVOS DE ENTRADA\n")
    f.write("-" * 70 + "\n")
    f.write(f"{INPUT_FILE_2018_2024}\n")
    f.write(f"{INPUT_FILE_2018_2025}\n\n")

    f.write("ARQUIVOS DE SAÍDA\n")
    f.write("-" * 70 + "\n")
    f.write(f"{OUTPUT_FILE_2018_2024}\n")
    f.write(f"{OUTPUT_FILE_2018_2025}\n")
    f.write(f"{OUTLIERS_REMOVIDOS_2018_2024}\n")
    f.write(f"{OUTLIERS_REMOVIDOS_2018_2025}\n\n")

    f.write("COLUNAS AVALIADAS PARA OUTLIER\n")
    f.write("-" * 70 + "\n")
    for coluna in outlier_cols_2024:
        f.write(f"- {coluna}\n")

    f.write("\nRESUMO 2018-2024\n")
    f.write("-" * 70 + "\n")
    f.write(str(df_2024["outlier_tipo"].value_counts()))
    f.write("\n")
    f.write(f"Linhas antes: {len(df_2024)}\n")
    f.write(f"Linhas depois: {len(df_2024_tratado)}\n")
    f.write(f"Outliers removidos: {len(outliers_removidos_2024)}\n")

    f.write("\nRESUMO 2018-2025\n")
    f.write("-" * 70 + "\n")
    f.write(str(df_2025["outlier_tipo"].value_counts()))
    f.write("\n")
    f.write(f"Linhas antes: {len(df_2025)}\n")
    f.write(f"Linhas depois: {len(df_2025_tratado)}\n")
    f.write(f"Outliers removidos: {len(outliers_removidos_2025)}\n")

    f.write("\nNORMALIZAÇÃO RECALCULADA\n")
    f.write("-" * 70 + "\n")
    f.write("Colunas Z-score:\n")
    for coluna in zscore_cols:
        f.write(f"- {coluna} -> {coluna}_zscore\n")

    f.write("\nColunas MinMaxScaler:\n")
    for coluna in minmax_cols:
        f.write(f"- {coluna} -> {coluna}_minmax\n")


# 10. Salvar documentação metodológica
salvar_metodologia()

print("\nRelatório salvo em:")
print(REPORT_FILE)

print("\nDocumentação metodológica salva em:")
print(METHODOLOGY_FILE)

print("\nEtapa 06 finalizada com sucesso.")
