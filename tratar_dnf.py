"""
Classifica cada linha da base em uma das categorias:
- "classificado": o piloto terminou a corrida 
- "dnf_piloto": abandonou por culpa do piloto (bateu, rodou, saiu da pista)
- "dnf_mecanico": abandonou por falha do carro, ou o motivo nao ficou claro
  (ex: "Retired", "Disqualified", "Did not start"). 

Separa a base em duas: quem entra no treino do modelo
("classificado" + "dnf_piloto") e quem fica de fora ("dnf_mecanico", que e um evento
aleatorio).
"""

import os
import pandas as pd


# pasta onde esta a base consolidada
PASTA_DADOS = "dados"
PASTA_PROCESSADOS = os.path.join(PASTA_DADOS, "processados")

ARQUIVO_ENTRADA = os.path.join(PASTA_PROCESSADOS, "base_consolidada_2018_2025.csv")

ARQUIVO_SAIDA_TREINO = os.path.join(PASTA_PROCESSADOS, "base_dnf_treino_2018_2025.csv")
ARQUIVO_SAIDA_EXCLUIDOS = os.path.join(PASTA_PROCESSADOS, "base_dnf_excluidos_2018_2025.csv")


# quem terminou a corrida
STATUS_CLASSIFICADO = {"finished", "lapped"}

# abandono por culpa do piloto (bateu, rodou, saiu da pista,...)
PALAVRAS_DNF_PILOTO = [
    "accident",
    "collision",
    "spun off",
    "spin",
    "crash",
    "damage",
]

def normalizar_status(status):
    """deixa o texto do status em minusculo e sem espacos nas pontas"""
    return str(status).strip().lower()


def classificado(status_norm):
    """verifica se o piloto terminou a corrida"""
    if status_norm in STATUS_CLASSIFICADO:
        return True

    # ex: "+1 Lap", "+2 Laps"
    if status_norm.startswith("+") and "lap" in status_norm:
        return True

    return False


def contem_alguma_palavra(status_norm, lista_palavras):
    """verifica se o status contem alguma das palavras da lista"""
    return any(palavra in status_norm for palavra in lista_palavras)


def classificar_categoria(status):
    """decide a categoria da linha: classificado, dnf_piloto ou dnf_mecanico"""

    status_norm = normalizar_status(status)

    if classificado(status_norm):
        return "classificado"

    if contem_alguma_palavra(status_norm, PALAVRAS_DNF_PILOTO):
        return "dnf_piloto"

    # dnf_mecanico: falha mecanica (motor, cambio, freio,...) e os casos
    # que nao ficou claro (ex: "Retired", "Disqualified", "Did not start", "Withdrew")
    return "dnf_mecanico"


if __name__ == "__main__":

    print("Lendo base consolidada...")
    base = pd.read_csv(ARQUIVO_ENTRADA)

    print("Classificando cada linha (terminou / dnf piloto / dnf mecanico)...")
    base["status_categoria"] = base["status"].apply(classificar_categoria)

    print()
    print(base["status_categoria"].value_counts())

    # so entra no treino quem terminou a corrida ou abandonou por culpa do piloto
    # abandono mecanico/motivo desconhecido fica de fora do treino do modelo
    entra_no_treino = base["status_categoria"].isin(["classificado", "dnf_piloto"])

    base_treino = base[entra_no_treino].copy()
    base_excluidos = base[~entra_no_treino].copy()

    os.makedirs(PASTA_PROCESSADOS, exist_ok=True)

    base_treino.to_csv(ARQUIVO_SAIDA_TREINO, index=False)
    base_excluidos.to_csv(ARQUIVO_SAIDA_EXCLUIDOS, index=False)

    print()
    print("Linhas que entram no treino:", len(base_treino))
    print("Linhas excluidas (dnf mecanico/motivo desconhecido):", len(base_excluidos))
    print()
    print("Base de treino salva em:", ARQUIVO_SAIDA_TREINO)
    print("Base excluida salva em:", ARQUIVO_SAIDA_EXCLUIDOS)
