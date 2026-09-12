"""
Etapa 3 - Classificacao de DNF.

Adiciona a coluna `status_categoria` na base tratada.

Categorias:
- classificado
- dnf_piloto
- dnf_mecanico
- dnf_outro

Nenhuma linha e removida nesta etapa.
Os arquivos de log servem apenas para auditoria e evidencia.
As features historicas de DNF serao calculadas posteriormente,
sempre utilizando apenas corridas anteriores.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(BASE_DIR, "dados")
PASTA_PROCESSADOS = os.path.join(PASTA_DADOS, "processados")
PASTA_LOGS = os.path.join(PASTA_DADOS, "logs")

ARQUIVO_ENTRADA = os.path.join(PASTA_PROCESSADOS, "base_consolidada_2018_2025.csv")
ARQUIVO_SAIDA = os.path.join(PASTA_PROCESSADOS, "base_tratada_2018_2025.csv")
ARQUIVO_LOG_DNF_PILOTO = os.path.join(PASTA_LOGS, "log_dnf_piloto_2018_2025.csv")
ARQUIVO_LOG_DNF_MECANICO = os.path.join(PASTA_LOGS, "log_dnf_mecanico_2018_2025.csv")
ARQUIVO_LOG_DNF_OUTRO = os.path.join(PASTA_LOGS, "log_dnf_outro_2018_2025.csv")

STATUS_CLASSIFICADO = {"finished", "lapped"}

PALAVRAS_DNF_PILOTO = [
    "accident",
    "collision",
    "spun off",
    "spin",
    "crash",
]

PALAVRAS_DNF_MECANICO = [
    "engine",
    "gearbox",
    "transmission",
    "clutch",
    "hydraulics",
    "electrical",
    "electronics",
    "brakes",
    "brake",
    "suspension",
    "steering",
    "throttle",
    "technical",
    "mechanical",
    "overheating",
    "oil leak",
    "water leak",
    "water pressure",
    "fuel pressure",
    "fuel pump",
    "fuel system",
    "fuel leak",
    "power unit",
    "ers",
    "turbo",
    "radiator",
    "cooling system",
    "exhaust",
    "driveshaft",
    "differential",
    "ignition",
    "battery",
    "wheel nut",
    "wheel",
]

PALAVRAS_DNF_OUTRO = [
    "retired",
    "disqualified",
    "did not start",
    "didn't start",
    "did not qualify",
    "withdrew",
    "withdrawn",
    "excluded",
    "not classified",
    "damage",
    "puncture",
    "tyre",
    "vibrations",
    "illness",
    "rear wing",
    "front wing",
    "undertray",
    "debris",
]


def normalizar_status(status):
    if pd.isna(status):
        return ""
    return str(status).strip().lower()


def classificado(status):
    if status in STATUS_CLASSIFICADO:
        return True
    return status.startswith("+") and "lap" in status


def contem_alguma_palavra(status, palavras):
    return any(palavra in status for palavra in palavras)


def classificar_categoria(status):
    status = normalizar_status(status)

    if status == "":
        return "dnf_outro"
    if classificado(status):
        return "classificado"

    if contem_alguma_palavra(status, PALAVRAS_DNF_PILOTO):
        return "dnf_piloto"
    if contem_alguma_palavra(status, PALAVRAS_DNF_MECANICO):
        return "dnf_mecanico"
    if contem_alguma_palavra(status, PALAVRAS_DNF_OUTRO):
        return "dnf_outro"

    return "dnf_outro"


if __name__ == "__main__":
    print("Lendo base:", ARQUIVO_ENTRADA)

    if not os.path.exists(ARQUIVO_ENTRADA):
        raise FileNotFoundError(f"Arquivo nao encontrado: {ARQUIVO_ENTRADA}")

    base = pd.read_csv(ARQUIVO_ENTRADA)

    if "status" not in base.columns:
        raise KeyError("Coluna 'status' nao encontrada na base.")

    base["status_categoria"] = base["status"].apply(classificar_categoria)

    resumo = (
        base["status_categoria"]
        .value_counts()
        .reindex(["classificado", "dnf_piloto", "dnf_mecanico", "dnf_outro"], fill_value=0)
    )

    print()
    print("Resumo da classificacao:")
    print(resumo)

    percentuais = (
        base["status_categoria"]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(["classificado", "dnf_piloto", "dnf_mecanico", "dnf_outro"], fill_value=0)
        .round(2)
    )

    print()
    print("Percentuais:")
    print(percentuais.astype(str) + "%")

    os.makedirs(PASTA_PROCESSADOS, exist_ok=True)
    os.makedirs(PASTA_LOGS, exist_ok=True)

    base.to_csv(ARQUIVO_SAIDA, index=False)

    log_dnf_piloto = base[base["status_categoria"] == "dnf_piloto"].copy()
    log_dnf_piloto.to_csv(ARQUIVO_LOG_DNF_PILOTO, index=False)

    log_dnf_mecanico = base[base["status_categoria"] == "dnf_mecanico"].copy()
    log_dnf_mecanico.to_csv(ARQUIVO_LOG_DNF_MECANICO, index=False)

    log_dnf_outro = base[base["status_categoria"] == "dnf_outro"].copy()
    log_dnf_outro.to_csv(ARQUIVO_LOG_DNF_OUTRO, index=False)

    print()
    print("ETAPA 3 CONCLUIDA")
    print("Nenhuma linha foi removida.")
    print("Base tratada:", ARQUIVO_SAIDA)
    print("Log DNF piloto:", ARQUIVO_LOG_DNF_PILOTO)
    print("Log DNF mecanico:", ARQUIVO_LOG_DNF_MECANICO)
    print("Log DNF outro:", ARQUIVO_LOG_DNF_OUTRO)