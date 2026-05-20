from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 04 - Normalização das variáveis numéricas
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_encoded_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_encoded_2018_2025.csv"


# Arquivos de saída
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_normalizado_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_normalizado_2018_2025.csv"

REPORT_FILE = PROCESSED_DIR / "relatorio_04_normalizacao.txt"
METHODOLOGY_FILE = DOCS_DIR / "metodologia_normalizacao.md"



# Colunas para normalização
# Z-score para variáveis numéricas contínuas
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

# MinMaxScaler para GridPosition e Laps
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


def selecionar_colunas_existentes(df, colunas):
    #Retorna apenas as colunas que existem no DataFrame.
    return [coluna for coluna in colunas if coluna in df.columns]


def preencher_nulos_com_mediana(df_treino, df_aplicacao, colunas):
    #Preenche valores nulos usando a mediana da base de treino 2018-2024. Isso evita vazamento de informação da base 2018-2025 para o cálculo dos parâmetros de normalização.
 
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


def aplicar_normalizacao(df_2018_2024, df_2018_2025):
    #Aplica:
    # StandardScaler / Z-score nas variáveis contínuas
    # MinMaxScaler em grid_position e laps

    #Importante:
    #Os scalers são ajustados na base 2018-2024 e aplicados também na base 2018-2025.
 
    df_2018_2024 = df_2018_2024.copy()
    df_2018_2025 = df_2018_2025.copy()

    # Garantir que colunas obrigatórias existem
    validar_colunas(
        df_2018_2024,
        ["grid_position", "laps"],
        "historico_encoded_2018_2024.csv"
    )

    validar_colunas(
        df_2018_2025,
        ["grid_position", "laps"],
        "historico_encoded_2018_2025.csv"
    )

    # Selecionar somente colunas existentes
    zscore_cols = selecionar_colunas_existentes(df_2018_2024, ZSCORE_COLUMNS)
    minmax_cols = selecionar_colunas_existentes(df_2018_2024, MINMAX_COLUMNS)

    # Converter para numérico
    for coluna in zscore_cols + minmax_cols:
        df_2018_2024[coluna] = pd.to_numeric(
            df_2018_2024[coluna],
            errors="coerce"
        )

        df_2018_2025[coluna] = pd.to_numeric(
            df_2018_2025[coluna],
            errors="coerce"
        )

    # Preencher nulos usando mediana da base 2018-2024
    df_2018_2024, df_2018_2025, medianas = preencher_nulos_com_mediana(
        df_2018_2024,
        df_2018_2025,
        zscore_cols + minmax_cols
    )

    # Z-score
    standard_scaler = StandardScaler()

    if zscore_cols:
        standard_scaler.fit(df_2018_2024[zscore_cols])

        zscore_2024 = standard_scaler.transform(df_2018_2024[zscore_cols])
        zscore_2025 = standard_scaler.transform(df_2018_2025[zscore_cols])

        for i, coluna in enumerate(zscore_cols):
            df_2018_2024[f"{coluna}_zscore"] = zscore_2024[:, i]
            df_2018_2025[f"{coluna}_zscore"] = zscore_2025[:, i]

    # MinMaxScaler
    minmax_scaler = MinMaxScaler()

    if minmax_cols:
        minmax_scaler.fit(df_2018_2024[minmax_cols])

        minmax_2024 = minmax_scaler.transform(df_2018_2024[minmax_cols])
        minmax_2025 = minmax_scaler.transform(df_2018_2025[minmax_cols])

        for i, coluna in enumerate(minmax_cols):
            df_2018_2024[f"{coluna}_minmax"] = minmax_2024[:, i]
            df_2018_2025[f"{coluna}_minmax"] = minmax_2025[:, i]

    return df_2018_2024, df_2018_2025, zscore_cols, minmax_cols, medianas


def salvar_metodologia(zscore_cols, minmax_cols):
    # Salva documentação metodológica da etapa de normalização.

    texto = f"""# Normalização das variáveis numéricas

## Objetivo

Esta etapa tem como objetivo padronizar variáveis numéricas para uso em modelos de Machine Learning.

A normalização evita que variáveis com escalas muito diferentes tenham impacto desproporcional no treinamento do modelo.

## Z-score

Foi aplicado Z-score, também conhecido como padronização, nas variáveis numéricas contínuas.

A fórmula geral é:

z = (x - média) / desvio padrão

As colunas normalizadas por Z-score foram:

{chr(10).join([f"- `{coluna}`" for coluna in zscore_cols])}

Para preservar os dados originais, foram criadas novas colunas com o sufixo `_zscore`.

## MinMaxScaler

Foi aplicado MinMaxScaler nas variáveis `grid_position` e `laps`.

A fórmula geral é:

x_normalizado = (x - mínimo) / (máximo - mínimo)

As colunas normalizadas por MinMaxScaler foram:

{chr(10).join([f"- `{coluna}`" for coluna in minmax_cols])}

Para preservar os dados originais, foram criadas novas colunas com o sufixo `_minmax`.

## Critério metodológico

Os parâmetros de normalização foram ajustados com base na base histórica de 2018 a 2024 e aplicados também à base 2018 a 2025.

Essa decisão evita vazamento de informação da base com 2025 para o processo de ajuste dos scalers.

## Arquivos gerados

- `historico_normalizado_2018_2024.csv`
- `historico_normalizado_2018_2025.csv`

A base principal recomendada para treinamento inicial do modelo é a versão 2018-2024.
"""

    with open(METHODOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


# ============================================================
# 1. Carregar arquivos da etapa 03
# ============================================================

historico_2018_2024 = pd.read_csv(INPUT_FILE_2018_2024)
historico_2018_2025 = pd.read_csv(INPUT_FILE_2018_2025)

print("Arquivos carregados com sucesso.")
print(f"Histórico encoded 2018-2024: {historico_2018_2024.shape}")
print(f"Histórico encoded 2018-2025: {historico_2018_2025.shape}")


# ============================================================
# 2. Aplicar normalização
# ============================================================

normalizado_2018_2024, normalizado_2018_2025, zscore_cols, minmax_cols, medianas = aplicar_normalizacao(
    historico_2018_2024,
    historico_2018_2025
)

print("\nNormalização aplicada com sucesso.")

print("\nColunas com Z-score:")
print(zscore_cols)

print("\nColunas com MinMaxScaler:")
print(minmax_cols)

print("\nDimensões finais:")
print(f"2018-2024: {normalizado_2018_2024.shape}")
print(f"2018-2025: {normalizado_2018_2025.shape}")


# ============================================================
# 3. Conferência das novas colunas
# ============================================================

zscore_novas_colunas = [f"{coluna}_zscore" for coluna in zscore_cols]
minmax_novas_colunas = [f"{coluna}_minmax" for coluna in minmax_cols]

print("\nNovas colunas Z-score:")
print(zscore_novas_colunas)

print("\nNovas colunas MinMax:")
print(minmax_novas_colunas)


# ============================================================
# 4. Salvar arquivos finais
# ============================================================

normalizado_2018_2024.to_csv(
    OUTPUT_FILE_2018_2024,
    index=False,
    encoding="utf-8-sig"
)

normalizado_2018_2025.to_csv(
    OUTPUT_FILE_2018_2025,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos salvos com sucesso:")
print(OUTPUT_FILE_2018_2024)
print(OUTPUT_FILE_2018_2025)


# ============================================================
# 5. Salvar relatório
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 04 NORMALIZAÇÃO\n")
    f.write("=" * 60 + "\n\n")

    f.write("ARQUIVOS DE ENTRADA\n")
    f.write("-" * 60 + "\n")
    f.write(f"{INPUT_FILE_2018_2024}\n")
    f.write(f"{INPUT_FILE_2018_2025}\n\n")

    f.write("ARQUIVOS DE SAÍDA\n")
    f.write("-" * 60 + "\n")
    f.write(f"{OUTPUT_FILE_2018_2024}\n")
    f.write(f"{OUTPUT_FILE_2018_2025}\n\n")

    f.write("NORMALIZAÇÃO APLICADA\n")
    f.write("-" * 60 + "\n")
    f.write("Z-score aplicado nas variáveis numéricas contínuas.\n")
    f.write("MinMaxScaler aplicado em grid_position e laps.\n\n")

    f.write("COLUNAS COM Z-SCORE\n")
    f.write("-" * 60 + "\n")
    for coluna in zscore_cols:
        f.write(f"- {coluna} -> {coluna}_zscore\n")

    f.write("\nCOLUNAS COM MINMAXSCALER\n")
    f.write("-" * 60 + "\n")
    for coluna in minmax_cols:
        f.write(f"- {coluna} -> {coluna}_minmax\n")

    f.write("\nMEDIANAS USADAS PARA PREENCHER NULOS\n")
    f.write("-" * 60 + "\n")
    for coluna, mediana in medianas.items():
        f.write(f"{coluna}: {mediana}\n")

    f.write("\nDIMENSÕES\n")
    f.write("-" * 60 + "\n")
    f.write(f"Base 2018-2024 antes: {historico_2018_2024.shape}\n")
    f.write(f"Base 2018-2024 depois: {normalizado_2018_2024.shape}\n")
    f.write(f"Base 2018-2025 antes: {historico_2018_2025.shape}\n")
    f.write(f"Base 2018-2025 depois: {normalizado_2018_2025.shape}\n")


print("\nRelatório salvo em:")
print(REPORT_FILE)


# ============================================================
# 6. Salvar documentação metodológica
# ============================================================

salvar_metodologia(zscore_cols, minmax_cols)

print("\nDocumentação metodológica salva em:")
print(METHODOLOGY_FILE)

print("\nEtapa 04 finalizada com sucesso.")