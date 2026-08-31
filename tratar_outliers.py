"""
Trata os outliers dos tempos de volta (fastf1_avg_lap_time e
fastf1_best_lap_time) e dos tempos de pitstop (tempo_total_pitstop e
tempo_medio_pitstop):

- Um valor e considerado outlier quando fica muito acima do normal (mais
  que 3 desvios padrao acima da mediana):
  - tempo de volta: compara com a mediana do mesmo circuito (ao longo dos
    anos).
  - pitstop: compara com a mediana da mesma corrida (so os pilotos daquele
    dia). 
- So remove a linha se o valor for invalido (erro real de
  sensor/medicao). Nessa base (que ja passou pelo tratamento de DNF e de
  valores ausentes) isso nao acontece, mas o codigo confere mesmo assim:
  - tempo de volta: nulo, zero ou negativo.
  - pitstop: nulo ou negativo.
- Todo o resto (outlier com explicacao, tipo chuva ou o piloto bater, ou
  sem explicacao) e mantido na base. So marca em duas colunas (uma pro tempo de 
  volta e outra pro pitstop) qual foi a situacao de cada linha:
  - "normal": valor dentro do esperado.
  - "outlier_explicado": valor extremo, mas com motivo conhecido (chuva
    ou o piloto bateu/saiu da pista).
  - "outlier_sem_causa": valor extremo sem motivo conhecido - mantido
    mesmo assim.
  - "invalido": valor impossivel - essa linha e removida da base.
"""

import os
import pandas as pd


PASTA_DADOS = "dados"
PASTA_PROCESSADOS = os.path.join(PASTA_DADOS, "processados")

ARQUIVO_ENTRADA = os.path.join(PASTA_PROCESSADOS, "base_tratada_2018_2025.csv")

ARQUIVO_SAIDA = os.path.join(PASTA_PROCESSADOS, "base_outliers_marcados_2018_2025.csv")
ARQUIVO_SAIDA_INVALIDOS = os.path.join(PASTA_PROCESSADOS, "base_valores_invalidos_removidos_2018_2025.csv")


# colunas de tempo de volta: outlier comparando com o mesmo circuito
COLUNAS_TEMPO_VOLTA = ["fastf1_avg_lap_time", "fastf1_best_lap_time"]
GRUPO_TEMPO_VOLTA = ["circuit_id"]

# colunas de pitstop: outlier comparando com a mesma corrida
COLUNAS_PITSTOP = ["tempo_total_pitstop", "tempo_medio_pitstop"]
GRUPO_PITSTOP = ["season", "round"]


def marcar_valor_invalido(base, coluna, permite_zero):
    """um valor e invalido se for nulo ou negativo. tempo de volta tambem nao 
    pode ser zero, mas pitstop pode"""

    if permite_zero:
        return base[coluna].isna() | (base[coluna] < 0)

    return base[coluna].isna() | (base[coluna] <= 0)


def marcar_outlier(base, coluna, colunas_grupo):
    """um valor e outlier se fica acima de 3 desvios padrao da mediana
    do grupo (circuito ou corrida)"""

    mediana_grupo = base.groupby(colunas_grupo)[coluna].transform("median")
    desvio_grupo = base.groupby(colunas_grupo)[coluna].transform("std")

    limite = mediana_grupo + 3 * desvio_grupo

    return base[coluna] > limite


def tem_explicacao(base):
    """o outlier tem explicacao se a corrida teve chuva ou se o
    piloto bateu/saiu da pista"""

    return base["choveu"] | (base["status_categoria"] == "dnf_piloto")


def marcar_situacao(base, coluna_situacao, colunas, colunas_grupo, explicado):
    """cria a coluna de situacao (normal/outlier_explicado/outlier_sem_causa)"""

    eh_outlier = pd.Series(False, index=base.index)
    for coluna in colunas:
        eh_outlier = eh_outlier | marcar_outlier(base, coluna, colunas_grupo)

    base[coluna_situacao] = "normal"
    base.loc[eh_outlier & explicado, coluna_situacao] = "outlier_explicado"
    base.loc[eh_outlier & ~explicado, coluna_situacao] = "outlier_sem_causa"

    return base, eh_outlier


if __name__ == "__main__":

    print("Lendo base tratada...")
    base = pd.read_csv(ARQUIVO_ENTRADA)

    print("Conferindo se existe algum valor invalido...")
    invalido = pd.Series(False, index=base.index)
    for coluna in COLUNAS_TEMPO_VOLTA:
        invalido = invalido | marcar_valor_invalido(base, coluna, permite_zero=False)
    for coluna in COLUNAS_PITSTOP:
        invalido = invalido | marcar_valor_invalido(base, coluna, permite_zero=True)

    base_invalidos = base[invalido].copy()
    base = base[~invalido].copy()

    base_invalidos["situacao_tempo_volta"] = "invalido"
    base_invalidos["situacao_pitstop"] = "invalido"

    print("Valores invalidos removidos:", invalido.sum())

    explicado = tem_explicacao(base)

    print("Procurando outliers nos tempos de volta (comparando com o circuito)...")
    base, eh_outlier_tempo = marcar_situacao(
        base, "situacao_tempo_volta", COLUNAS_TEMPO_VOLTA, GRUPO_TEMPO_VOLTA, explicado
    )

    print("Procurando outliers no pitstop (comparando com a mesma corrida)...")
    base, eh_outlier_pitstop = marcar_situacao(
        base, "situacao_pitstop", COLUNAS_PITSTOP, GRUPO_PITSTOP, explicado
    )

    print()
    print("Outliers no tempo de volta:", eh_outlier_tempo.sum())
    print("  - outlier_explicado (chuva/batida):", (eh_outlier_tempo & explicado).sum())
    print("  - outlier_sem_causa:", (eh_outlier_tempo & ~explicado).sum())
    print()
    print("Outliers no pitstop:", eh_outlier_pitstop.sum())
    print("  - outlier_explicado (chuva/batida):", (eh_outlier_pitstop & explicado).sum())
    print("  - outlier_sem_causa:", (eh_outlier_pitstop & ~explicado).sum())
    print("(nenhum foi removido, so marcado - continuam na base)")

    os.makedirs(PASTA_PROCESSADOS, exist_ok=True)

    base.to_csv(ARQUIVO_SAIDA, index=False)
    base_invalidos.to_csv(ARQUIVO_SAIDA_INVALIDOS, index=False)

    print()
    print("Linhas:", len(base))
    print("Base com outliers marcados salva em:", ARQUIVO_SAIDA)
    print("Valores invalidos removidos salvos em:", ARQUIVO_SAIDA_INVALIDOS)
