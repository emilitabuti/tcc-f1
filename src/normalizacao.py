from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 04 - Normalização das variáveis numéricas
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models" / "preprocessing"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_encoded_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_encoded_2018_2025.csv"
INPUT_BASE_LIMPA_2018_2024 = PROCESSED_DIR / "base_historica_encoded_2018_2024.csv"
INPUT_BASE_LIMPA_2018_2025 = PROCESSED_DIR / "base_historica_encoded_2018_2025.csv"


# Arquivos de saída
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_normalizado_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_normalizado_2018_2025.csv"
OUTPUT_BASE_LIMPA_2018_2024 = PROCESSED_DIR / "base_historica_normalizado_2018_2024.csv"
OUTPUT_BASE_LIMPA_2018_2025 = PROCESSED_DIR / "base_historica_normalizado_2018_2025.csv"

STANDARD_SCALER_HISTORICO = MODELS_DIR / "standard_scaler_historico.joblib"
MINMAX_SCALER_HISTORICO = MODELS_DIR / "minmax_scaler_historico.joblib"
STANDARD_SCALER_BASE_LIMPA = MODELS_DIR / "standard_scaler_base_historica.joblib"
MINMAX_SCALER_BASE_LIMPA = MODELS_DIR / "minmax_scaler_base_historica.joblib"

REPORT_FILE = PROCESSED_DIR / "relatorio_04_normalizacao.txt"
METHODOLOGY_FILE = DOCS_DIR / "metodologia_normalizacao.md"



# Colunas para normalização
# Z-score para variáveis numéricas contínuas
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


def repo_relative(path):
    # Registra caminhos portáveis no relatório, independentemente da máquina.
    return path.relative_to(BASE_DIR).as_posix()


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


def aplicar_normalizacao(df_2018_2024, df_2018_2025, nome_base_2024,
                         nome_base_2025, standard_scaler_path,
                         minmax_scaler_path):
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
        nome_base_2024
    )

    validar_colunas(
        df_2018_2025,
        ["grid_position", "laps"],
        nome_base_2025
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
        joblib.dump(standard_scaler, standard_scaler_path)

        zscore_2024 = standard_scaler.transform(df_2018_2024[zscore_cols])
        zscore_2025 = standard_scaler.transform(df_2018_2025[zscore_cols])

        for i, coluna in enumerate(zscore_cols):
            df_2018_2024[f"{coluna}_zscore"] = zscore_2024[:, i]
            df_2018_2025[f"{coluna}_zscore"] = zscore_2025[:, i]

    # MinMaxScaler
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


def salvar_metodologia(resultados):
    # Salva documentação metodológica da etapa de normalização.
    zscore_cols = resultados[0]["zscore_cols"]
    minmax_cols = resultados[0]["minmax_cols"]

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

Foram gerados artefatos `.joblib` com os scalers ajustados em 2018-2024 para permitir reaplicação reprodutível do mesmo pré-processamento.

## Arquivos gerados

- `historico_normalizado_2018_2024.csv`
- `historico_normalizado_2018_2025.csv`
- `base_historica_normalizado_2018_2024.csv`
- `base_historica_normalizado_2018_2025.csv`

A base principal recomendada para treinamento inicial do modelo é a versão 2018-2024.
"""

    with open(METHODOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


# ============================================================
# 1. Processar arquivos da etapa 03
# ============================================================

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


# ============================================================
# 5. Salvar relatório
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 04 NORMALIZAÇÃO\n")
    f.write("=" * 60 + "\n\n")

    f.write("ARQUIVOS DE ENTRADA\n")
    f.write("-" * 60 + "\n")
    for resultado in resultados:
        f.write(f"{repo_relative(resultado['input_2024'])}\n")
        f.write(f"{repo_relative(resultado['input_2025'])}\n")
    f.write("\n")

    f.write("ARQUIVOS DE SAÍDA\n")
    f.write("-" * 60 + "\n")
    for resultado in resultados:
        f.write(f"{repo_relative(resultado['output_2024'])}\n")
        f.write(f"{repo_relative(resultado['output_2025'])}\n")
    f.write("\n")

    f.write("ARTEFATOS DE SCALER\n")
    f.write("-" * 60 + "\n")
    for resultado in resultados:
        f.write(f"{repo_relative(resultado['standard_scaler'])}\n")
        f.write(f"{repo_relative(resultado['minmax_scaler'])}\n")
    f.write("\n")

    f.write("NORMALIZAÇÃO APLICADA\n")
    f.write("-" * 60 + "\n")
    f.write("Z-score aplicado nas variáveis numéricas contínuas.\n")
    f.write("MinMaxScaler aplicado em grid_position e laps.\n\n")

    for resultado in resultados:
        f.write(f"{resultado['rotulo'].upper()}\n")
        f.write("-" * 60 + "\n")
        f.write("Colunas com Z-score:\n")
        for coluna in resultado["zscore_cols"]:
            f.write(f"- {coluna} -> {coluna}_zscore\n")

        f.write("\nColunas com MinMaxScaler:\n")
        for coluna in resultado["minmax_cols"]:
            f.write(f"- {coluna} -> {coluna}_minmax\n")

        f.write("\nMedianas usadas para preencher nulos:\n")
        for coluna, mediana in resultado["medianas"].items():
            f.write(f"{coluna}: {mediana}\n")

        f.write("\nDimensões:\n")
        f.write(f"Base 2018-2024 antes: {resultado['inicial_2024']}\n")
        f.write(f"Base 2018-2024 depois: {resultado['final_2024']}\n")
        f.write(f"Base 2018-2025 antes: {resultado['inicial_2025']}\n")
        f.write(f"Base 2018-2025 depois: {resultado['final_2025']}\n\n")


print("\nRelatório salvo em:")
print(REPORT_FILE)


# ============================================================
# 6. Salvar documentação metodológica
# ============================================================

salvar_metodologia(resultados)

print("\nDocumentação metodológica salva em:")
print(METHODOLOGY_FILE)

print("\nEtapa 04 finalizada com sucesso.")
