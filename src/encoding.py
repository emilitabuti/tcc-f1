from pathlib import Path
import pandas as pd
import numpy as np
 
 
# 03 - Encoding das variáveis categóricas
BASE_DIR = Path(__file__).resolve().parents[1]
 
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
 
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
 
 
# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_dnf_excluded_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_dnf_excluded_2018_2025.csv"
 
 
# Arquivos de saída
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_encoded_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_encoded_2018_2025.csv"
 
REPORT_FILE = PROCESSED_DIR / "relatorio_03_encoding.txt"
METHODOLOGY_FILE = DOCS_DIR / "metodologia_encoding.md"
 
 
# Mapeamento ordinal dos compostos de pneu
# Soft > Medium > Hard
COMPOUND_ORDINAL_MAP = {
    "SOFT": 3,
    "MEDIUM": 2,
    "HARD": 1,
 
    # Compostos de chuva/intermediário não entram na ordem seco,
    # mas são preservados com valores próprios para não virar nulo.
    "INTERMEDIATE": 0,
    "WET": 0,
    "UNKNOWN": 0,
}
 
 
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
 
 
def normalizar_composto(valor):
    #Padroniza o nome do composto de pneu.
    
    if pd.isna(valor):
        return "UNKNOWN"
 
    return str(valor).strip().upper()
 
 
def escolher_coluna_circuito(df):
    #Define qual coluna será usada para representar o circuito.
 
    #Preferência:
    #1. circuit_id, se existir
    #2. race_name, se circuit_id não existir
    
    if "circuit_id" in df.columns:
        return "circuit_id"
 
    if "race_name" in df.columns:
        return "race_name"
 
    raise ValueError(
        "Nenhuma coluna de circuito encontrada. "
        "Esperado: circuit_id ou race_name."
    )
 
 
def aplicar_encoding(df, nome_base):
    # Aplica:
    # One-Hot Encoding para circuito/corrida
    # One-Hot Encoding para construtor
    # Label Encoding ordinal para composto de pneu
    
    df = df.copy()
 
    # Validação mínima
    colunas_obrigatorias = [
        "season",
        "round",
        "driver_id",
        "constructor_id",
    ]
 
    validar_colunas(df, colunas_obrigatorias, nome_base)
 
    # Escolher coluna de circuito
    coluna_circuito = escolher_coluna_circuito(df)
 
    # Definir coluna de composto
    # Prioridade: primeiro composto usado na corrida
    if "fastf1_first_compound" in df.columns:
        coluna_composto = "fastf1_first_compound"
    elif "fastf1_main_compound" in df.columns:
        coluna_composto = "fastf1_main_compound"
    else:
        raise ValueError(
            "Nenhuma coluna de composto encontrada. "
            "Esperado: fastf1_first_compound ou fastf1_main_compound."
        )
 
    # Normalizar composto
    df["compound_normalizado"] = df[coluna_composto].apply(normalizar_composto)
 
    # Label Encoding ordinal para composto
    df["compound_ordinal"] = (
        df["compound_normalizado"]
        .map(COMPOUND_ORDINAL_MAP)
        .fillna(0)
        .astype(int)
    )
 
    # One-Hot Encoding para circuito/corrida e construtor
    df_encoded = pd.get_dummies(
        df,
        columns=[coluna_circuito, "constructor_id"],
        prefix=["circuito", "constructor"],
        dtype=int
    )
 
    return df_encoded, coluna_circuito, coluna_composto
 
 
def salvar_metodologia(coluna_circuito, coluna_composto):
    # Salva documentação metodológica da etapa de encoding.
    texto = f"""# Encoding das variáveis categóricas

## Objetivo

Esta etapa tem como objetivo transformar variáveis categóricas em representações numéricas, permitindo sua utilização por modelos de Machine Learning.

## One-Hot Encoding

Foi aplicado One-Hot Encoding para variáveis categóricas sem ordem natural.

As variáveis utilizadas foram:

- `{coluna_circuito}`: representação do circuito ou corrida.
- `constructor_id`: identificação da equipe/construtor.

O One-Hot Encoding cria uma coluna binária para cada categoria. Dessa forma, evita-se que o modelo interprete categorias nominais como se tivessem uma ordem numérica.

## Label Encoding ordinal para composto de pneu

Para o composto de pneu foi utilizado Label Encoding ordinal, pois os compostos de pista seca possuem uma relação técnica de dureza.

A coluna utilizada foi:

- `{coluna_composto}`

A regra aplicada foi:

- SOFT = 3
- MEDIUM = 2
- HARD = 1
- INTERMEDIATE/WET/UNKNOWN = 0

A ordem adotada segue a relação:

SOFT > MEDIUM > HARD

Compostos intermediários, de chuva ou ausentes foram mantidos com valor 0, pois não seguem a mesma escala ordinal dos compostos de pista seca.

## Arquivos gerados

Esta etapa gera duas bases:

- `historico_encoded_2018_2024.csv`
- `historico_encoded_2018_2025.csv`

A base principal recomendada para treinamento inicial do modelo é a versão 2018-2024.
"""

    with open(METHODOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


# 1. Carregar arquivos da etapa 02
historico_2018_2024 = pd.read_csv(INPUT_FILE_2018_2024)
historico_2018_2025 = pd.read_csv(INPUT_FILE_2018_2025)

print("Arquivos carregados com sucesso.")
print(f"Histórico DNF Excluded 2018-2024: {historico_2018_2024.shape}")
print(f"Histórico DNF Excluded 2018-2025: {historico_2018_2025.shape}")


# 2. Aplicar encoding
encoded_2018_2024, coluna_circuito_2024, coluna_composto_2024 = aplicar_encoding(
    historico_2018_2024,
    "historico_dnf_excluded_2018_2024.csv"
)

encoded_2018_2025, coluna_circuito_2025, coluna_composto_2025 = aplicar_encoding(
    historico_2018_2025,
    "historico_dnf_excluded_2018_2025.csv"
)


# 3. Conferir colunas criadas
colunas_circuito_2024 = [
    col for col in encoded_2018_2024.columns
    if col.startswith("circuito_")
]

colunas_constructor_2024 = [
    col for col in encoded_2018_2024.columns
    if col.startswith("constructor_")
]

colunas_circuito_2025 = [
    col for col in encoded_2018_2025.columns
    if col.startswith("circuito_")
]

colunas_constructor_2025 = [
    col for col in encoded_2018_2025.columns
    if col.startswith("constructor_")
]

print("\nEncoding aplicado com sucesso.")

print("\nBase 2018-2024:")
print(f"Coluna usada para circuito: {coluna_circuito_2024}")
print(f"Coluna usada para composto: {coluna_composto_2024}")
print(f"Quantidade de colunas de circuito criadas: {len(colunas_circuito_2024)}")
print(f"Quantidade de colunas de construtor criadas: {len(colunas_constructor_2024)}")
print(f"Dimensão final: {encoded_2018_2024.shape}")

print("\nBase 2018-2025:")
print(f"Coluna usada para circuito: {coluna_circuito_2025}")
print(f"Coluna usada para composto: {coluna_composto_2025}")
print(f"Quantidade de colunas de circuito criadas: {len(colunas_circuito_2025)}")
print(f"Quantidade de colunas de construtor criadas: {len(colunas_constructor_2025)}")
print(f"Dimensão final: {encoded_2018_2025.shape}")



# 4. Conferir compound_ordinal
print("\nDistribuição compound_ordinal - 2018-2024:")
print(encoded_2018_2024["compound_ordinal"].value_counts().sort_index())

print("\nDistribuição compound_ordinal - 2018-2025:")
print(encoded_2018_2025["compound_ordinal"].value_counts().sort_index())


# 5. Salvar arquivos finais
encoded_2018_2024.to_csv(
    OUTPUT_FILE_2018_2024,
    index=False,
    encoding="utf-8-sig"
)

encoded_2018_2025.to_csv(
    OUTPUT_FILE_2018_2025,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos salvos com sucesso:")
print(OUTPUT_FILE_2018_2024)
print(OUTPUT_FILE_2018_2025)


# 6. Salvar relatório
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 03 ENCODING\n")
    f.write("=" * 60 + "\n\n")

    f.write("ARQUIVOS DE ENTRADA\n")
    f.write("-" * 60 + "\n")
    f.write(f"{INPUT_FILE_2018_2024}\n")
    f.write(f"{INPUT_FILE_2018_2025}\n\n")

    f.write("ARQUIVOS DE SAÍDA\n")
    f.write("-" * 60 + "\n")
    f.write(f"{OUTPUT_FILE_2018_2024}\n")
    f.write(f"{OUTPUT_FILE_2018_2025}\n\n")

    f.write("ENCODING APLICADO\n")
    f.write("-" * 60 + "\n")
    f.write("One-Hot Encoding aplicado para circuito/corrida e constructor_id.\n")
    f.write("Label Encoding ordinal aplicado para composto de pneu.\n\n")

    f.write("REGRA DO COMPOSTO ORDINAL\n")
    f.write("-" * 60 + "\n")
    f.write("SOFT = 3\n")
    f.write("MEDIUM = 2\n")
    f.write("HARD = 1\n")
    f.write("INTERMEDIATE/WET/UNKNOWN = 0\n\n")

    f.write("BASE 2018-2024\n")
    f.write("-" * 60 + "\n")
    f.write(f"Dimensão inicial: {historico_2018_2024.shape}\n")
    f.write(f"Dimensão final: {encoded_2018_2024.shape}\n")
    f.write(f"Coluna usada para circuito: {coluna_circuito_2024}\n")
    f.write(f"Coluna usada para composto: {coluna_composto_2024}\n")
    f.write(f"Colunas de circuito criadas: {len(colunas_circuito_2024)}\n")
    f.write(f"Colunas de construtor criadas: {len(colunas_constructor_2024)}\n\n")

    f.write("BASE 2018-2025\n")
    f.write("-" * 60 + "\n")
    f.write(f"Dimensão inicial: {historico_2018_2025.shape}\n")
    f.write(f"Dimensão final: {encoded_2018_2025.shape}\n")
    f.write(f"Coluna usada para circuito: {coluna_circuito_2025}\n")
    f.write(f"Coluna usada para composto: {coluna_composto_2025}\n")
    f.write(f"Colunas de circuito criadas: {len(colunas_circuito_2025)}\n")
    f.write(f"Colunas de construtor criadas: {len(colunas_constructor_2025)}\n")

print("\nRelatório salvo em:")
print(REPORT_FILE)


# 7. Salvar documentação metodológica
salvar_metodologia(coluna_circuito_2024, coluna_composto_2024)

print("\nDocumentação metodológica salva em:")
print(METHODOLOGY_FILE)

print("\nEtapa 03 finalizada com sucesso.")