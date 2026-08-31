"""
Trata os valores ausentes (NaN):

- pitstops: NaN quer dizer que o piloto bateu/quebrou antes de fazer
  qualquer parada, ou seja, o numero de paradas dele e 0, e nao "desconhecido".
- Q2_s e Q3_s: NaN aqui e normal, so quem passa pra fase seguinte da
  classificacao tem tempo registrado nelas. Antes de preencher, guarda numa
  coluna se o piloto chegou ou nao naquela fase (pode ser um sinal util pro
  modelo).
- tempos de volta, clima e Q1_s: dado real que faltou (corrida sem sessao
  registrada no fastf1, ou piloto que nao chegou a fazer nenhuma volta
  cronometrada). Preenche com a mediana hierarquica: primeiro tenta a
  mediana do mesmo circuito na mesma temporada, se nao tiver usa a mediana
  da temporada inteira, e se nao usa a mediana geral da base.
- tire_compound_predominante e choveu: poucos casos (mesma corrida sem dado
  do fastf1), preenche com o valor mais comum da base toda.
- qualifying_position: 3 casos, piloto nao fez tempo na classificacao mas
  correu (largou de algum lugar), preenche com a posicao de largada dele.
"""

import os
import pandas as pd


PASTA_DADOS = "dados"
PASTA_PROCESSADOS = os.path.join(PASTA_DADOS, "processados")

ARQUIVO_ENTRADA = os.path.join(PASTA_PROCESSADOS, "base_dnf_treino_2018_2025.csv")
ARQUIVO_SAIDA = os.path.join(PASTA_PROCESSADOS, "base_tratada_2018_2025.csv")


# colunas de pitstop: NaN = piloto nao chegou a parar (bateu/quebrou antes)
COLUNAS_PITSTOP = ["num_pitstops", "tempo_total_pitstop", "tempo_medio_pitstop"]

# colunas numericas que sao dado real faltando, tratadas com mediana hierarquica
COLUNAS_MEDIANA_HIERARQUICA = [
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_num_voltas",
    "fastf1_num_stints",
    "fastf1_tyre_life_media",
    "Q1_s",
    "Q2_s",
    "Q3_s",
    "temp_ar_media",
    "temp_pista_media",
    "umidade_media",
    "vento_media",
]


def preencher_pitstops_com_zero(base):
    """se nao tem registro de pitstop, e pq o piloto nao chegou a parar"""

    for coluna in COLUNAS_PITSTOP:
        base[coluna] = base[coluna].fillna(0)

    return base


def marcar_fase_da_classificacao(base):
    """guarda numa coluna 0/1 se o piloto chegou na Q2 e na Q3, antes de
    preencher os tempos que faltam"""

    base["chegou_no_q2"] = base["Q2_s"].notna().astype(int)
    base["chegou_no_q3"] = base["Q3_s"].notna().astype(int)

    return base


def preencher_mediana_hierarquica(base, coluna):
    """preenche o NaN de uma coluna numerica em 3 tentativas: mediana do
    circuito na mesma temporada -> mediana da temporada inteira -> mediana
    geral da base"""

    mediana_circuito_temporada = base.groupby(["circuit_id", "season"])[coluna].transform("median")
    base[coluna] = base[coluna].fillna(mediana_circuito_temporada)

    mediana_temporada = base.groupby("season")[coluna].transform("median")
    base[coluna] = base[coluna].fillna(mediana_temporada)

    base[coluna] = base[coluna].fillna(base[coluna].median())

    return base


def preencher_categoricas_com_valor_mais_comum(base):
    """poucos casos (mesma corrida sem dado do fastf1), preenche com o
    valor mais frequente da base toda"""

    for coluna in ["tire_compound_predominante", "choveu"]:
        valor_mais_comum = base[coluna].mode().iloc[0]
        base[coluna] = base[coluna].fillna(valor_mais_comum)

    return base


def preencher_qualifying_position_com_grid(base):
    """os pilotos que nao fizeram tempo na classificacao mas
    correram largaram de algum lugar, entao usa a posicao de largada
    como aproximacao"""

    base["qualifying_position"] = base["qualifying_position"].fillna(base["grid_position"])

    return base


if __name__ == "__main__":

    print("Lendo base depois do tratamento de DNF...")
    base = pd.read_csv(ARQUIVO_ENTRADA)

    print("\nNaN antes do tratamento:")
    print(base.isna().sum()[base.isna().sum() > 0])

    base = preencher_pitstops_com_zero(base)
    base = marcar_fase_da_classificacao(base)

    for coluna in COLUNAS_MEDIANA_HIERARQUICA:
        base = preencher_mediana_hierarquica(base, coluna)

    base = preencher_categoricas_com_valor_mais_comum(base)
    base = preencher_qualifying_position_com_grid(base)

    print("\nNaN depois do tratamento:")
    faltando = base.isna().sum()
    print(faltando[faltando > 0] if faltando.sum() > 0 else "nenhum :)")

    os.makedirs(PASTA_PROCESSADOS, exist_ok=True)
    base.to_csv(ARQUIVO_SAIDA, index=False)

    print("\nBase tratada salva em:", ARQUIVO_SAIDA)
    print("Linhas:", len(base), "| Colunas:", len(base.columns))
