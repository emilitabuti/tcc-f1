from pathlib import Path
import pandas as pd
import numpy as np


#Tratamento de DNFs
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2025.csv"
INPUT_BASE_LIMPA_2018_2024 = PROCESSED_DIR / "base_historica_limpa_2018_2024.csv"
INPUT_BASE_LIMPA_2018_2025 = PROCESSED_DIR / "base_historica_limpa_2018_2025.csv"


# Arquivos de saida
OUTPUT_CLASSIFICADO_2018_2024 = PROCESSED_DIR / "historico_dnf_classificado_2018_2024.csv"
OUTPUT_DNF_EXCLUDED_2018_2024 = PROCESSED_DIR / "historico_dnf_excluded_2018_2024.csv"

OUTPUT_CLASSIFICADO_2018_2025 = PROCESSED_DIR / "historico_dnf_classificado_2018_2025.csv"
OUTPUT_DNF_EXCLUDED_2018_2025 = PROCESSED_DIR / "historico_dnf_excluded_2018_2025.csv"

OUTPUT_BASE_CLASSIFICADA_2018_2024 = PROCESSED_DIR / "base_historica_dnf_classificado_2018_2024.csv"
OUTPUT_BASE_DNF_EXCLUDED_2018_2024 = PROCESSED_DIR / "base_historica_dnf_excluded_2018_2024.csv"

OUTPUT_BASE_CLASSIFICADA_2018_2025 = PROCESSED_DIR / "base_historica_dnf_classificado_2018_2025.csv"
OUTPUT_BASE_DNF_EXCLUDED_2018_2025 = PROCESSED_DIR / "base_historica_dnf_excluded_2018_2025.csv"


# pilotos que terminaram ou foram classificados (mesmo com volta a menos)
STATUS_CLASSIFICADO_EXATOS = {
    "finished",
    "lapped",
}

# Exemplo de status de classificados com voltas atras:
# +1 Lap, +2 Laps, +3 Laps etc.



# abandonos por culpa do piloto - acidente, rodada, colisao
DNF_PILOTO_KEYWORDS = [
    "accident",
    "collision",
    "spun off",
    "spun-off",
    "spin",
    "crash",
    "damage",
]


# abandonos por falha do carro
DNF_CARRO_KEYWORDS = [
    "engine",
    "gearbox",
    "transmission",
    "clutch",
    "hydraulics",
    "electrical",
    "electronics",
    "ers",
    "power unit",
    "power loss",
    "brakes",
    "brake",
    "suspension",
    "steering",
    "radiator",
    "oil",
    "oil leak",
    "water pressure",
    "water leak",
    "water pump",
    "cooling system",
    "fuel",
    "fuel pressure",
    "fuel pump",
    "fuel leak",
    "out of fuel",
    "turbo",
    "exhaust",
    "mechanical",
    "overheating",
    "puncture",
    "tyre",
    "wheel",
    "wheel nut",
    "driveshaft",
    "differential",
    "battery",
    "undertray",
    "front wing",
    "rear wing",
    "vibrations",
]


# outros casos que não encaixam nas categorias acima
DNF_OUTROS_KEYWORDS = [
    "did not start",
    "dns",
    "withdrew",
    "withdrawn",
    "illness",
    "excluded",
    "disqualified",
    "retired",
    "not classified",
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
    # Registra caminhos portaveis no relatorio, independentemente da maquina.
    return path.relative_to(BASE_DIR).as_posix()


def normalizar_status(status):
    #Padroniza o texto do status para facilitar a classificação.

    if pd.isna(status):
        return ""

    return str(status).strip().lower()


def is_status_classificado(status_normalizado):
    #Identifica pilotos que terminaram a corrida ou foram classificados, mesmo com uma ou mais voltas atrás.
    if status_normalizado in STATUS_CLASSIFICADO_EXATOS:
        return True

    # Exemplo: +1 Lap, +2 Laps, +10 Laps
    if status_normalizado.startswith("+") and "lap" in status_normalizado:
        return True

    return False


def contem_keyword(status_normalizado, keywords):
    #Verifica se algum termo da lista aparece no status
    return any(keyword in status_normalizado for keyword in keywords)


def classificar_dnf(status):
    # olha o status e decide em qual categoria o piloto cai
    status_normalizado = normalizar_status(status)

    if status_normalizado == "":
        return "dnf_outros"

    if is_status_classificado(status_normalizado):
        return "classificado"

    if contem_keyword(status_normalizado, DNF_PILOTO_KEYWORDS):
        return "dnf_piloto"

    if contem_keyword(status_normalizado, DNF_CARRO_KEYWORDS):
        return "dnf_carro"

    if contem_keyword(status_normalizado, DNF_OUTROS_KEYWORDS):
        return "dnf_outros"

    # Qualquer outro status nao classificado como "Finished" ou "+x Laps"
    # será tratado como DNF indefinido/outros.
    return "dnf_outros"


def aplicar_tratamento_dnf(df, nome_base):
    # aplica a classificação e cria as flags de DNF
    df = df.copy()

    validar_colunas(
        df,
        ["season", "round", "driver_id", "status"],
        nome_base
    )

    df["status_normalizado"] = df["status"].apply(normalizar_status)
    df["dnf_categoria"] = df["status"].apply(classificar_dnf)

    # cria flags individuais pra cada tipo de DNF
    df["is_dnf"] = df["dnf_categoria"].isin(
        ["dnf_piloto", "dnf_carro", "dnf_outros"]
    )
    df["dnf_flag"] = df["is_dnf"].astype(int)
    df["dnf_driver_flag"] = (df["dnf_categoria"] == "dnf_piloto").astype(int)
    df["dnf_car_flag"] = (df["dnf_categoria"] == "dnf_carro").astype(int)
    df["dnf_other_flag"] = (df["dnf_categoria"] == "dnf_outros").astype(int)

    # a base DNF Excluded fica so com pilotos classificados
    df_dnf_excluded = df[df["dnf_categoria"] == "classificado"].copy()

    return df, df_dnf_excluded


def resumo_dnf(df):
    #Gera resumo da quantidade de registros por categoria DNF.
    return df["dnf_categoria"].value_counts().sort_index()


def processar_base(input_2024, input_2025, output_classificado_2024,
                   output_excluded_2024, output_classificado_2025,
                   output_excluded_2025, rotulo):
    # lê os dois arquivos e aplica o tratamento de DNF nos dois
    historico_2018_2024 = pd.read_csv(input_2024)
    historico_2018_2025 = pd.read_csv(input_2025)

    print(f"\nArquivos carregados com sucesso ({rotulo}).")
    print(f"{rotulo} 2018-2024: {historico_2018_2024.shape}")
    print(f"{rotulo} 2018-2025: {historico_2018_2025.shape}")

    classificado_2018_2024, dnf_excluded_2018_2024 = aplicar_tratamento_dnf(
        historico_2018_2024,
        input_2024.name
    )

    classificado_2018_2025, dnf_excluded_2018_2025 = aplicar_tratamento_dnf(
        historico_2018_2025,
        input_2025.name
    )

    print(f"\nResumo DNF - {rotulo} - 2018 a 2024:")
    print(resumo_dnf(classificado_2018_2024))

    print(f"\nResumo DNF - {rotulo} - 2018 a 2025:")
    print(resumo_dnf(classificado_2018_2025))

    print(f"\nLinhas após DNF Excluded ({rotulo}):")
    print(f"2018-2024: {len(dnf_excluded_2018_2024)}")
    print(f"2018-2025: {len(dnf_excluded_2018_2025)}")

    classificado_2018_2024.to_csv(
        output_classificado_2024,
        index=False,
        encoding="utf-8-sig"
    )

    dnf_excluded_2018_2024.to_csv(
        output_excluded_2024,
        index=False,
        encoding="utf-8-sig"
    )

    classificado_2018_2025.to_csv(
        output_classificado_2025,
        index=False,
        encoding="utf-8-sig"
    )

    dnf_excluded_2018_2025.to_csv(
        output_excluded_2025,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArquivos salvos com sucesso ({rotulo}):")
    print(output_classificado_2024)
    print(output_excluded_2024)
    print(output_classificado_2025)
    print(output_excluded_2025)

    return {
        "rotulo": rotulo,
        "input_2024": input_2024,
        "input_2025": input_2025,
        "classificado_2018_2024": classificado_2018_2024,
        "dnf_excluded_2018_2024": dnf_excluded_2018_2024,
        "classificado_2018_2025": classificado_2018_2025,
        "dnf_excluded_2018_2025": dnf_excluded_2018_2025,
        "outputs": [
            output_classificado_2024,
            output_excluded_2024,
            output_classificado_2025,
            output_excluded_2025,
        ],
    }


# processa as bases da etapa 01
resultados = []

resultados.append(processar_base(
    INPUT_FILE_2018_2024,
    INPUT_FILE_2018_2025,
    OUTPUT_CLASSIFICADO_2018_2024,
    OUTPUT_DNF_EXCLUDED_2018_2024,
    OUTPUT_CLASSIFICADO_2018_2025,
    OUTPUT_DNF_EXCLUDED_2018_2025,
    "Histórico enriquecido com FastF1"
))

resultados.append(processar_base(
    INPUT_BASE_LIMPA_2018_2024,
    INPUT_BASE_LIMPA_2018_2025,
    OUTPUT_BASE_CLASSIFICADA_2018_2024,
    OUTPUT_BASE_DNF_EXCLUDED_2018_2024,
    OUTPUT_BASE_CLASSIFICADA_2018_2025,
    OUTPUT_BASE_DNF_EXCLUDED_2018_2025,
    "Base histórica limpa"
))

print("\nEtapa 02 finalizada com sucesso.")
