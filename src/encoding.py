from pathlib import Path
import json

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder


# 03 - Encoding das variáveis categóricas
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "preprocessing"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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

ENCODER_HISTORICO = MODELS_DIR / "onehot_encoder_historico_fastf1.joblib"
SCHEMA_HISTORICO = MODELS_DIR / "schema_encoding_historico_fastf1.json"
ENCODER_BASE_LIMPA = MODELS_DIR / "onehot_encoder_base_historica.joblib"
SCHEMA_BASE_LIMPA = MODELS_DIR / "schema_encoding_base_historica.json"


# constructor_id e circuit_id ficam no df mesmo depois do encoding - sao uteis depois
COLS_METADADOS = {"constructor_id", "circuit_id"}

# resolve nomes de corrida duplicados pro mesmo circuito físico (ex: Styrian = red_bull_ring)
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


# ordinal pra composto de pneu - Soft é mais macio, Hard é mais duro
# chuva/intermediário ficam com 0 porque não entram na mesma escala
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


def paths_encoding_por_rotulo(rotulo):
    # escolhe qual encoder/schema salvar dependendo do tipo de base
    if "FastF1" in rotulo:
        return ENCODER_HISTORICO, SCHEMA_HISTORICO

    return ENCODER_BASE_LIMPA, SCHEMA_BASE_LIMPA


def normalizar_composto(valor):
    #Padroniza o nome do composto de pneu.

    if pd.isna(valor):
        return "UNKNOWN"

    return str(valor).strip().upper()


def escolher_coluna_circuito(df):
    # prefere circuit_id, mas cai pra race_name se não existir

    if "circuit_id" in df.columns:
        return "circuit_id"

    if "race_name" in df.columns:
        return "race_name"

    raise ValueError(
        "Nenhuma coluna de circuito encontrada. "
        "Esperado: circuit_id ou race_name."
    )


def preparar_base_encoding(df, nome_base):
    # valida colunas e prepara circuito, composto e encoding ordinal
    df = df.copy()

    colunas_obrigatorias = [
        "season",
        "round",
        "driver_id",
        "constructor_id",
    ]

    validar_colunas(df, colunas_obrigatorias, nome_base)

    # deriva circuit_id a partir de race_name quando ainda nao existe
    if "race_name" in df.columns and "circuit_id" not in df.columns:
        df["circuit_id"] = df["race_name"].map(RACE_NAME_TO_CIRCUIT_ID)
        sem_mapa = df[df["circuit_id"].isnull()]["race_name"].unique()
        if len(sem_mapa) > 0:
            raise ValueError(
                f"race_name sem mapeamento para circuit_id em {nome_base}: {sem_mapa}"
            )

    coluna_circuito = escolher_coluna_circuito(df)

    # pega o primeiro composto usado - se nao tiver, usa o mais frequente
    if "fastf1_first_compound" in df.columns:
        coluna_composto = "fastf1_first_compound"
    elif "fastf1_main_compound" in df.columns:
        coluna_composto = "fastf1_main_compound"
    else:
        coluna_composto = None

    if coluna_composto is None:
        df["compound_normalizado"] = "UNKNOWN"
    else:
        df["compound_normalizado"] = df[coluna_composto].apply(normalizar_composto)

    # converte composto pra numero ordinal
    df["compound_ordinal"] = (
        df["compound_normalizado"]
        .map(COMPOUND_ORDINAL_MAP)
        .fillna(0)
        .astype(int)
    )

    return df, coluna_circuito, coluna_composto


def criar_onehot_encoder():
    # tenta sparse_output primeiro (sklearn novo), cai pra sparse no antigo
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def aplicar_onehot(df, encoder, colunas_categoricas):
    # transforma as colunas categóricas e renomeia pra ficar no padrão
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

    # mantem constructor_id e circuit_id no df mesmo tendo feito encoding deles
    colunas_para_dropar = [c for c in colunas_categoricas if c not in COLS_METADADOS]
    df_sem_categoricas = df.drop(columns=colunas_para_dropar)
    return pd.concat([df_sem_categoricas, encoded_df], axis=1)


def aplicar_encoding_par(df_2024, df_2025, nome_base_2024, nome_base_2025):
    # fit do encoder na base 2024 e aplica nas duas - evita vazamento de info
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
        encoder,
        colunas_categoricas,
        coluna_circuito_2024,
        coluna_composto_2024,
        coluna_composto_2025,
    )


def salvar_encoder_schema(
    encoder,
    colunas_categoricas,
    encoded_2018_2024,
    encoded_2018_2025,
    rotulo,
):
    # persiste o encoder e um json com as categorias e colunas pra reprodução futura
    encoder_path, schema_path = paths_encoding_por_rotulo(rotulo)
    joblib.dump(encoder, encoder_path)

    schema = {
        "rotulo": rotulo,
        "encoder_path": repo_relative(encoder_path),
        "colunas_categoricas": colunas_categoricas,
        "categorias": {
            coluna: [str(valor) for valor in categorias]
            for coluna, categorias in zip(colunas_categoricas, encoder.categories_)
        },
        "colunas_saida_2018_2024": list(encoded_2018_2024.columns),
        "colunas_saida_2018_2025": list(encoded_2018_2025.columns),
        "compound_ordinal_map": COMPOUND_ORDINAL_MAP,
        "handle_unknown": "ignore",
        "observacao": (
            "Encoder ajustado na base 2018-2024 e reaplicado na base 2018-2025. "
            "Usar este schema para manter ordem e compatibilidade das colunas em dados futuros."
        ),
    }

    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return encoder_path, schema_path


def processar_base(input_2024, input_2025, output_2024, output_2025, rotulo):
    # lê, aplica encoding e salva os arquivos de saída
    historico_2018_2024 = pd.read_csv(input_2024)
    historico_2018_2025 = pd.read_csv(input_2025)

    print(f"\nArquivos carregados com sucesso ({rotulo}).")
    print(f"{rotulo} 2018-2024: {historico_2018_2024.shape}")
    print(f"{rotulo} 2018-2025: {historico_2018_2025.shape}")

    (
        encoded_2018_2024,
        encoded_2018_2025,
        encoder,
        colunas_categoricas,
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

    encoder_path, schema_path = salvar_encoder_schema(
        encoder,
        colunas_categoricas,
        encoded_2018_2024,
        encoded_2018_2025,
        rotulo,
    )

    print(f"\nArquivos salvos com sucesso ({rotulo}):")
    print(output_2024)
    print(output_2025)
    print(encoder_path)
    print(schema_path)

    return {
        "rotulo": rotulo,
        "input_2024": input_2024,
        "input_2025": input_2025,
        "output_2024": output_2024,
        "output_2025": output_2025,
        "encoder_path": encoder_path,
        "schema_path": schema_path,
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


# processa os arquivos da etapa 02
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

print("\nEtapa 03 finalizada com sucesso.")
