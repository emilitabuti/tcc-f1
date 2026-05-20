from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
 
 
# 03 - Encoding das variáveis categóricas
BASE_DIR = Path(__file__).resolve().parents[1]
 
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
 
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
 
 
# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_dnf_excluded_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_dnf_excluded_2018_2025.csv"
INPUT_BASE_LIMPA_2018_2024 = PROCESSED_DIR / "base_historica_dnf_excluded_2018_2024.csv"
INPUT_BASE_LIMPA_2018_2025 = PROCESSED_DIR / "base_historica_dnf_excluded_2018_2025.csv"
 
 
# Arquivos de saída
OUTPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_encoded_2018_2024.csv"
OUTPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_encoded_2018_2025.csv"
OUTPUT_BASE_LIMPA_2018_2024 = PROCESSED_DIR / "base_historica_encoded_2018_2024.csv"
OUTPUT_BASE_LIMPA_2018_2025 = PROCESSED_DIR / "base_historica_encoded_2018_2025.csv"
 
REPORT_FILE = PROCESSED_DIR / "relatorio_03_encoding.txt"
METHODOLOGY_FILE = DOCS_DIR / "metodologia_encoding.md"
 
 
# Colunas de metadado que devem ser preservadas mesmo quando são usadas como
# categoria de entrada do encoder (não serão dropadas do DataFrame final).
COLS_METADADOS = {"constructor_id", "circuit_id"}

# Mapeamento race_name → circuit_id (físico)
# Resolve duplicatas: Styrian/Austrian → red_bull_ring, Brazilian/São Paulo → interlagos, etc.
RACE_NAME_TO_CIRCUIT_ID = {
    "Australian Grand Prix": "albert_park",
    "Bahrain Grand Prix": "bahrain",
    "Chinese Grand Prix": "shanghai",
    "Azerbaijan Grand Prix": "baku",
    "Spanish Grand Prix": "catalunya",
    "Monaco Grand Prix": "monaco",
    "Canadian Grand Prix": "villeneuve",
    "French Grand Prix": "ricard",
    "Austrian Grand Prix": "red_bull_ring",
    "British Grand Prix": "silverstone",
    "German Grand Prix": "hockenheim",
    "Hungarian Grand Prix": "hungaroring",
    "Belgian Grand Prix": "spa",
    "Italian Grand Prix": "monza",
    "Singapore Grand Prix": "marina_bay",
    "Russian Grand Prix": "sochi",
    "Japanese Grand Prix": "suzuka",
    "United States Grand Prix": "americas",
    "Mexican Grand Prix": "rodriguez",
    "Mexico City Grand Prix": "rodriguez",
    "Brazilian Grand Prix": "interlagos",
    "São Paulo Grand Prix": "interlagos",
    "Abu Dhabi Grand Prix": "yas_marina",
    # 2020 COVID extras
    "Styrian Grand Prix": "red_bull_ring",
    "70th Anniversary Grand Prix": "silverstone",
    "Tuscan Grand Prix": "mugello",
    "Eifel Grand Prix": "nurburgring",
    "Turkish Grand Prix": "istanbul",
    "Sakhir Grand Prix": "bahrain_outer",
    # 2021+
    "Dutch Grand Prix": "zandvoort",
    "Emilia Romagna Grand Prix": "imola",
    "Portuguese Grand Prix": "portimao",
    # 2022+
    "Saudi Arabian Grand Prix": "jeddah",
    "Miami Grand Prix": "miami",
    # 2023+
    "Qatar Grand Prix": "losail",
    "Las Vegas Grand Prix": "las_vegas",
}


# Mapeamento ordinal dos compostos de pneu
# Soft > Medium > Hard
COMPOUND_ORDINAL_MAP = {
    "HYPERSOFT": 6,
    "ULTRASOFT": 5,
    "SUPERSOFT": 4,
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
 
 
def repo_relative(path):
    # Registra caminhos portáveis no relatório, independentemente da máquina.
    return path.relative_to(BASE_DIR).as_posix()


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
 
 
def preparar_base_encoding(df, nome_base):
    # Valida e prepara colunas categóricas antes do OneHotEncoder.
    df = df.copy()

    colunas_obrigatorias = [
        "season",
        "round",
        "driver_id",
        "constructor_id",
    ]

    validar_colunas(df, colunas_obrigatorias, nome_base)

    # Derivar circuit_id a partir de race_name (se ainda não existir).
    # Resolve duplicatas de nome (Styrian/Austrian → red_bull_ring, etc.)
    # para que o encoder use o circuito físico, não o nome da corrida.
    if "race_name" in df.columns and "circuit_id" not in df.columns:
        df["circuit_id"] = df["race_name"].map(RACE_NAME_TO_CIRCUIT_ID)
        sem_mapa = df[df["circuit_id"].isnull()]["race_name"].unique()
        if len(sem_mapa) > 0:
            raise ValueError(
                f"race_name sem mapeamento para circuit_id em {nome_base}: {sem_mapa}"
            )
 
    # Escolher coluna de circuito
    coluna_circuito = escolher_coluna_circuito(df)
 
    # Definir coluna de composto
    # Prioridade: primeiro composto usado na corrida
    if "fastf1_first_compound" in df.columns:
        coluna_composto = "fastf1_first_compound"
    elif "fastf1_main_compound" in df.columns:
        coluna_composto = "fastf1_main_compound"
    else:
        coluna_composto = None
 
    # Normalizar composto
    if coluna_composto is None:
        df["compound_normalizado"] = "UNKNOWN"
    else:
        df["compound_normalizado"] = df[coluna_composto].apply(normalizar_composto)
 
    # Label Encoding ordinal para composto
    df["compound_ordinal"] = (
        df["compound_normalizado"]
        .map(COMPOUND_ORDINAL_MAP)
        .fillna(0)
        .astype(int)
    )

    return df, coluna_circuito, coluna_composto


def criar_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def aplicar_onehot(df, encoder, colunas_categoricas):
    encoded_array = encoder.transform(df[colunas_categoricas])
    encoded_columns = encoder.get_feature_names_out()
    coluna_circuito = colunas_categoricas[0]
    encoded_columns = [
        col.replace(f"{coluna_circuito}_", "circuito_", 1)
           .replace("constructor_id_", "constructor_", 1)
        for col in encoded_columns
    ]
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_columns,
        index=df.index,
    ).astype(int)
 
    # Preserva colunas de metadado (constructor_id, circuit_id) mesmo quando
    # foram usadas como entrada do encoder — são úteis em Feature Engineering.
    colunas_para_dropar = [c for c in colunas_categoricas if c not in COLS_METADADOS]
    df_sem_categoricas = df.drop(columns=colunas_para_dropar)
    return pd.concat([df_sem_categoricas, encoded_df], axis=1)
 
 
def aplicar_encoding_par(df_2024, df_2025, nome_base_2024, nome_base_2025):
    # Fit do encoder na base 2018-2024 e transform nas duas versões.
    df_2024, coluna_circuito_2024, coluna_composto_2024 = preparar_base_encoding(
        df_2024,
        nome_base_2024,
    )
    df_2025, coluna_circuito_2025, coluna_composto_2025 = preparar_base_encoding(
        df_2025,
        nome_base_2025,
    )

    if coluna_circuito_2024 != coluna_circuito_2025:
        raise ValueError(
            "As bases 2018-2024 e 2018-2025 escolheram colunas de circuito "
            f"diferentes: {coluna_circuito_2024} vs {coluna_circuito_2025}"
        )

    colunas_categoricas = [coluna_circuito_2024, "constructor_id"]
    encoder = criar_onehot_encoder()
    encoder.fit(df_2024[colunas_categoricas])

    encoded_2024 = aplicar_onehot(df_2024, encoder, colunas_categoricas)
    encoded_2025 = aplicar_onehot(df_2025, encoder, colunas_categoricas)

    return (
        encoded_2024,
        encoded_2025,
        coluna_circuito_2024,
        coluna_composto_2024,
        coluna_composto_2025,
    )
 
 
def processar_base(input_2024, input_2025, output_2024, output_2025, rotulo):
    historico_2018_2024 = pd.read_csv(input_2024)
    historico_2018_2025 = pd.read_csv(input_2025)

    print(f"\nArquivos carregados com sucesso ({rotulo}).")
    print(f"{rotulo} 2018-2024: {historico_2018_2024.shape}")
    print(f"{rotulo} 2018-2025: {historico_2018_2025.shape}")

    (
        encoded_2018_2024,
        encoded_2018_2025,
        coluna_circuito_2024,
        coluna_composto_2024,
        coluna_composto_2025,
    ) = aplicar_encoding_par(
        historico_2018_2024,
        historico_2018_2025,
        input_2024.name,
        input_2025.name,
    )
    coluna_circuito_2025 = coluna_circuito_2024

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

    print(f"\nEncoding aplicado com sucesso ({rotulo}).")

    print("\nBase 2018-2024:")
    print(f"Coluna usada para circuito: {coluna_circuito_2024}")
    print(f"Coluna usada para composto: {coluna_composto_2024 or 'UNKNOWN'}")
    print(f"Quantidade de colunas de circuito criadas: {len(colunas_circuito_2024)}")
    print(f"Quantidade de colunas de construtor criadas: {len(colunas_constructor_2024)}")
    print(f"Dimensão final: {encoded_2018_2024.shape}")

    print("\nBase 2018-2025:")
    print(f"Coluna usada para circuito: {coluna_circuito_2025}")
    print(f"Coluna usada para composto: {coluna_composto_2025 or 'UNKNOWN'}")
    print(f"Quantidade de colunas de circuito criadas: {len(colunas_circuito_2025)}")
    print(f"Quantidade de colunas de construtor criadas: {len(colunas_constructor_2025)}")
    print(f"Dimensão final: {encoded_2018_2025.shape}")

    print(f"\nDistribuição compound_ordinal - {rotulo} - 2018-2024:")
    print(encoded_2018_2024["compound_ordinal"].value_counts().sort_index())

    print(f"\nDistribuição compound_ordinal - {rotulo} - 2018-2025:")
    print(encoded_2018_2025["compound_ordinal"].value_counts().sort_index())

    encoded_2018_2024.to_csv(
        output_2024,
        index=False,
        encoding="utf-8-sig"
    )

    encoded_2018_2025.to_csv(
        output_2025,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArquivos salvos com sucesso ({rotulo}):")
    print(output_2024)
    print(output_2025)

    return {
        "rotulo": rotulo,
        "input_2024": input_2024,
        "input_2025": input_2025,
        "output_2024": output_2024,
        "output_2025": output_2025,
        "inicial_2024": historico_2018_2024.shape,
        "inicial_2025": historico_2018_2025.shape,
        "final_2024": encoded_2018_2024.shape,
        "final_2025": encoded_2018_2025.shape,
        "coluna_circuito_2024": coluna_circuito_2024,
        "coluna_circuito_2025": coluna_circuito_2025,
        "coluna_composto_2024": coluna_composto_2024,
        "coluna_composto_2025": coluna_composto_2025,
        "colunas_circuito_2024": len(colunas_circuito_2024),
        "colunas_circuito_2025": len(colunas_circuito_2025),
        "colunas_constructor_2024": len(colunas_constructor_2024),
        "colunas_constructor_2025": len(colunas_constructor_2025),
    }


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

O encoder é ajustado na base 2018-2024 e reaplicado na base 2018-2025 com `handle_unknown="ignore"`, evitando quebra do pipeline caso apareçam categorias novas em dados futuros.

## Label Encoding ordinal para composto de pneu

Para o composto de pneu foi utilizado Label Encoding ordinal, pois os compostos de pista seca possuem uma relação técnica de dureza.

A coluna utilizada foi:

- `{coluna_composto}`

A regra aplicada foi:

- HYPERSOFT = 6
- ULTRASOFT = 5
- SUPERSOFT = 4
- SOFT = 3
- MEDIUM = 2
- HARD = 1
- INTERMEDIATE/WET/UNKNOWN = 0

A ordem adotada segue a relação:

SOFT > MEDIUM > HARD

Compostos intermediários, de chuva ou ausentes foram mantidos com valor 0, pois não seguem a mesma escala ordinal dos compostos de pista seca. Compostos secos antigos da era 2018, como HyperSoft, UltraSoft e SuperSoft, foram preservados na ordem ordinal por serem pneus de pista seca.

## Arquivos gerados

Esta etapa gera quatro bases:

- `historico_encoded_2018_2024.csv`
- `historico_encoded_2018_2025.csv`
- `base_historica_encoded_2018_2024.csv`
- `base_historica_encoded_2018_2025.csv`

A base principal recomendada para treinamento inicial do modelo é a versão 2018-2024.
"""

    with open(METHODOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


# 1. Processar arquivos da etapa 02
resultados = []

resultados.append(processar_base(
    INPUT_FILE_2018_2024,
    INPUT_FILE_2018_2025,
    OUTPUT_FILE_2018_2024,
    OUTPUT_FILE_2018_2025,
    "Histórico enriquecido com FastF1"
))

resultados.append(processar_base(
    INPUT_BASE_LIMPA_2018_2024,
    INPUT_BASE_LIMPA_2018_2025,
    OUTPUT_BASE_LIMPA_2018_2024,
    OUTPUT_BASE_LIMPA_2018_2025,
    "Base histórica limpa"
))


# 2. Salvar relatório
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 03 ENCODING\n")
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

    f.write("ENCODING APLICADO\n")
    f.write("-" * 60 + "\n")
    f.write("One-Hot Encoding aplicado para circuito/corrida e constructor_id.\n")
    f.write("Encoder ajustado em 2018-2024 e reaplicado em 2018-2025 com handle_unknown='ignore'.\n")
    f.write("Label Encoding ordinal aplicado para composto de pneu.\n\n")

    f.write("REGRA DO COMPOSTO ORDINAL\n")
    f.write("-" * 60 + "\n")
    f.write("HYPERSOFT = 6\n")
    f.write("ULTRASOFT = 5\n")
    f.write("SUPERSOFT = 4\n")
    f.write("SOFT = 3\n")
    f.write("MEDIUM = 2\n")
    f.write("HARD = 1\n")
    f.write("INTERMEDIATE/WET/UNKNOWN = 0\n\n")

    for resultado in resultados:
        f.write(f"{resultado['rotulo'].upper()} - 2018-2024\n")
        f.write("-" * 60 + "\n")
        f.write(f"Dimensão inicial: {resultado['inicial_2024']}\n")
        f.write(f"Dimensão final: {resultado['final_2024']}\n")
        f.write(f"Coluna usada para circuito: {resultado['coluna_circuito_2024']}\n")
        f.write(
            "Coluna usada para composto: "
            f"{resultado['coluna_composto_2024'] or 'UNKNOWN'}\n"
        )
        f.write(f"Colunas de circuito criadas: {resultado['colunas_circuito_2024']}\n")
        f.write(f"Colunas de construtor criadas: {resultado['colunas_constructor_2024']}\n\n")

        f.write(f"{resultado['rotulo'].upper()} - 2018-2025\n")
        f.write("-" * 60 + "\n")
        f.write(f"Dimensão inicial: {resultado['inicial_2025']}\n")
        f.write(f"Dimensão final: {resultado['final_2025']}\n")
        f.write(f"Coluna usada para circuito: {resultado['coluna_circuito_2025']}\n")
        f.write(
            "Coluna usada para composto: "
            f"{resultado['coluna_composto_2025'] or 'UNKNOWN'}\n"
        )
        f.write(f"Colunas de circuito criadas: {resultado['colunas_circuito_2025']}\n")
        f.write(f"Colunas de construtor criadas: {resultado['colunas_constructor_2025']}\n\n")

print("\nRelatório salvo em:")
print(REPORT_FILE)


# 3. Salvar documentação metodológica
salvar_metodologia(
    resultados[0]["coluna_circuito_2024"],
    resultados[0]["coluna_composto_2024"] or "UNKNOWN"
)

print("\nDocumentação metodológica salva em:")
print(METHODOLOGY_FILE)

print("\nEtapa 03 finalizada com sucesso.")
