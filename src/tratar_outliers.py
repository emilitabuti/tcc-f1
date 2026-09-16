"""ETAPA 5 - ANALISE TEMPORAL DE OUTLIERS.

Objetivos:
- identificar valores extremos nos tempos de volta e pitstop;
- utilizar somente corridas anteriores para calcular os limites;
- nao remover nenhuma linha da base;
- nao alterar os valores identificados como outliers;
- registrar os outliers encontrados em um log;
- manter as classificacoes de outlier apenas para auditoria.

Importante: para uma corrida t, nenhum calculo usa dados da propria corrida t
nem de corridas futuras. As informacoes observadas na corrida atual, como chuva
ou DNF do piloto, sao usadas somente no log para explicar o outlier.
"""

import os
import pandas as pd

PASTA_DADOS = "dados"
PASTA_PROCESSADOS = os.path.join(PASTA_DADOS, "processados")
ARQUIVO_ENTRADA = os.path.join(PASTA_PROCESSADOS, "base_tratada_2018_2025.csv")
ARQUIVO_SAIDA = ARQUIVO_ENTRADA
PASTA_LOGS = os.path.join(PASTA_DADOS, "logs")
ARQUIVO_LOG_OUTLIERS = os.path.join(PASTA_LOGS, "log_outliers_2018_2025.csv")

COLUNAS_TEMPO_VOLTA = ["fastf1_avg_lap_time", "fastf1_best_lap_time"]
COLUNAS_PITSTOP = ["tempo_total_pitstop", "tempo_medio_pitstop"]
MIN_AMOSTRAS = 5
NUM_DESVIOS = 3

COLUNAS_LOG = [
    "season", "round", "race_name", "circuit_id", "driver_id", "constructor_id",
    "coluna", "valor", "mediana_historica", "desvio_historico", "limite_outlier",
    "n_historico", "criterio_historico", "situacao", "classificacao_auditoria",
    "explicacao_observada",
]


def validar_base(base):
    """Confere se a Etapa 4 entregou dados consistentes."""
    colunas_obrigatorias = [
        "season", "round", "circuit_id", "driver_id", "num_pitstops",
        "status", "choveu", *COLUNAS_TEMPO_VOLTA, *COLUNAS_PITSTOP,
    ]

    faltando = [coluna for coluna in colunas_obrigatorias if coluna not in base.columns]
    if faltando:
        raise ValueError("Colunas obrigatorias ausentes: " + ", ".join(faltando))

    colunas_analisadas = COLUNAS_TEMPO_VOLTA + COLUNAS_PITSTOP + ["num_pitstops"]
    nan_encontrado = base[colunas_analisadas].isna().sum()
    nan_encontrado = nan_encontrado[nan_encontrado > 0]
    if len(nan_encontrado) > 0:
        print("\nNaN encontrados:")
        print(nan_encontrado)
        raise ValueError("Existem valores ausentes nas colunas analisadas. Revise a Etapa 4.")

    for coluna in COLUNAS_TEMPO_VOLTA:
        if (base[coluna] <= 0).any():
            raise ValueError(f"Foram encontrados valores <= 0 em {coluna}. Revise a Etapa 4.")

    if (base["num_pitstops"] < 0).any():
        raise ValueError("Foram encontrados valores negativos em num_pitstops.")

    for coluna in COLUNAS_PITSTOP:
        if (base[coluna] < 0).any():
            raise ValueError(f"Foram encontrados valores negativos em {coluna}.")

    for coluna in COLUNAS_PITSTOP:
        mascara = (base["num_pitstops"] > 0) & (base[coluna] <= 0)
        if mascara.any():
            raise ValueError(f"Existem registros com pitstop realizado e {coluna} <= 0. Revise a Etapa 4.")

    for coluna in COLUNAS_PITSTOP:
        mascara = (base["num_pitstops"] == 0) & (base[coluna] > 0)
        if mascara.any():
            raise ValueError(f"Existem registros com num_pitstops = 0 e {coluna} > 0. Revise a Etapa 4.")


def mascara_corridas_anteriores(base, season, round_):
    """Retorna somente registros de corridas anteriores."""
    return (base["season"] < season) | ((base["season"] == season) & (base["round"] < round_))


def calcular_estatisticas(valores):
    """Calcula mediana, desvio padrao e limite de outlier."""
    valores = valores.dropna().astype(float)
    n = len(valores)
    if n < MIN_AMOSTRAS:
        return None

    mediana = valores.median()
    desvio = valores.std(ddof=1)
    if pd.isna(desvio):
        return None

    return {"mediana": mediana, "desvio": desvio, "limite": mediana + NUM_DESVIOS * desvio, "n": n}


def estatisticas_tempo_volta(historico, coluna, circuit_id):
    """Compara o tempo de volta com o mesmo circuito em corridas anteriores."""
    mesmo_circuito = historico[historico["circuit_id"] == circuit_id]
    estatisticas = calcular_estatisticas(mesmo_circuito[coluna])
    if estatisticas is None:
        return None
    estatisticas["criterio"] = "mesmo_circuito_corridas_anteriores"
    return estatisticas


def estatisticas_pitstop_medio(historico, coluna, circuit_id, season):
    """Hierarquia temporal para tempo medio de pitstop."""
    historico = historico[historico["num_pitstops"] > 0]

    valores = historico.loc[historico["circuit_id"] == circuit_id, coluna]
    estatisticas = calcular_estatisticas(valores)
    if estatisticas is not None:
        estatisticas["criterio"] = "mesmo_circuito_corridas_anteriores"
        return estatisticas

    valores = historico.loc[historico["season"] == season, coluna]
    estatisticas = calcular_estatisticas(valores)
    if estatisticas is not None:
        estatisticas["criterio"] = "mesma_temporada_rounds_anteriores"
        return estatisticas

    estatisticas = calcular_estatisticas(historico[coluna])
    if estatisticas is not None:
        estatisticas["criterio"] = "todas_corridas_anteriores"
        return estatisticas
    return None


def estatisticas_pitstop_total(historico, coluna, circuit_id, season, num_pitstops):
    """Compara o tempo total de pitstop com a mesma quantidade de paradas."""
    historico = historico[(historico["num_pitstops"] > 0) & (historico["num_pitstops"] == num_pitstops)]

    valores = historico.loc[historico["circuit_id"] == circuit_id, coluna]
    estatisticas = calcular_estatisticas(valores)
    if estatisticas is not None:
        estatisticas["criterio"] = "mesmo_circuito_mesmo_num_pitstops"
        return estatisticas

    valores = historico.loc[historico["season"] == season, coluna]
    estatisticas = calcular_estatisticas(valores)
    if estatisticas is not None:
        estatisticas["criterio"] = "mesma_temporada_mesmo_num_pitstops"
        return estatisticas

    estatisticas = calcular_estatisticas(historico[coluna])
    if estatisticas is not None:
        estatisticas["criterio"] = "historico_completo_mesmo_num_pitstops"
        return estatisticas
    return None


def obter_explicacao_observada(linha):
    """Usa informacoes da corrida atual apenas para explicar o outlier no log."""
    choveu = bool(linha.get("choveu", 0))
    status = str(linha.get("status", "")).strip().lower()
    dnf_piloto = any(
        palavra in status
        for palavra in ["accident", "collision", "spun off", "spin", "crash"]
    )

    if choveu and dnf_piloto:
        return "outlier_explicado", "chuva e dnf_piloto"
    if choveu:
        return "outlier_explicado", "chuva"
    if dnf_piloto:
        return "outlier_explicado", "dnf_piloto"
    return "outlier_sem_causa", "sem causa observada"


def registrar_outlier(log_outliers, linha, coluna, estatisticas):
    """Registra detalhes do outlier identificado."""
    classificacao_auditoria, explicacao_observada = obter_explicacao_observada(linha)
    log_outliers.append({
        "season": linha.get("season"),
        "round": linha.get("round"),
        "race_name": linha.get("race_name"),
        "circuit_id": linha.get("circuit_id"),
        "driver_id": linha.get("driver_id"),
        "constructor_id": linha.get("constructor_id"),
        "coluna": coluna,
        "valor": linha[coluna],
        "mediana_historica": estatisticas["mediana"],
        "desvio_historico": estatisticas["desvio"],
        "limite_outlier": estatisticas["limite"],
        "n_historico": estatisticas["n"],
        "criterio_historico": estatisticas["criterio"],
        "situacao": "outlier_temporal",
        "classificacao_auditoria": classificacao_auditoria,
        "explicacao_observada": explicacao_observada,
    })


if __name__ == "__main__":
    print("=" * 70)
    print("ETAPA 5 - ANALISE TEMPORAL DE OUTLIERS")
    print("=" * 70)

    if not os.path.exists(ARQUIVO_ENTRADA):
        raise FileNotFoundError(f"Arquivo nao encontrado: {ARQUIVO_ENTRADA}")

    base = pd.read_csv(ARQUIVO_ENTRADA)
    print()
    print("Linhas:", len(base))

    colunas_numericas = ["season", "round", "num_pitstops", *COLUNAS_TEMPO_VOLTA, *COLUNAS_PITSTOP]
    for coluna in colunas_numericas:
        base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    print()
    print("Validando resultado da Etapa 4...")
    validar_base(base)
    print("Base valida para analise de outliers.")

    base = base.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    base["situacao_tempo_volta"] = "historico_insuficiente"
    base["situacao_pitstop"] = "historico_insuficiente"
    base.loc[base["num_pitstops"] == 0, "situacao_pitstop"] = "sem_pitstop"

    log_outliers = []
    corridas = base[["season", "round"]].drop_duplicates().sort_values(["season", "round"])

    for _, corrida in corridas.iterrows():
        season = corrida["season"]
        round_ = corrida["round"]
        mascara_corrida = (base["season"] == season) & (base["round"] == round_)
        indices_corrida = base.index[mascara_corrida]
        historico = base[mascara_corridas_anteriores(base, season, round_)]

        for indice in indices_corrida:
            linha = base.loc[indice]
            circuit_id = linha["circuit_id"]
            num_pitstops = linha["num_pitstops"]

            encontrou_historico_volta = False
            encontrou_outlier_volta = False
            for coluna in COLUNAS_TEMPO_VOLTA:
                estatisticas = estatisticas_tempo_volta(historico, coluna, circuit_id)
                if estatisticas is None:
                    continue
                encontrou_historico_volta = True
                valor = linha[coluna]
                if valor > estatisticas["limite"]:
                    encontrou_outlier_volta = True
                    registrar_outlier(log_outliers, linha, coluna, estatisticas)

            if encontrou_outlier_volta:
                base.at[indice, "situacao_tempo_volta"] = "outlier_temporal"
            elif encontrou_historico_volta:
                base.at[indice, "situacao_tempo_volta"] = "normal"

            if num_pitstops == 0:
                continue

            encontrou_historico_pit = False
            encontrou_outlier_pit = False

            estatisticas = estatisticas_pitstop_total(historico, "tempo_total_pitstop", circuit_id, season, num_pitstops)
            if estatisticas is not None:
                encontrou_historico_pit = True
                valor = linha["tempo_total_pitstop"]
                if valor > estatisticas["limite"]:
                    encontrou_outlier_pit = True
                    registrar_outlier(log_outliers, linha, "tempo_total_pitstop", estatisticas)

            estatisticas = estatisticas_pitstop_medio(historico, "tempo_medio_pitstop", circuit_id, season)
            if estatisticas is not None:
                encontrou_historico_pit = True
                valor = linha["tempo_medio_pitstop"]
                if valor > estatisticas["limite"]:
                    encontrou_outlier_pit = True
                    registrar_outlier(log_outliers, linha, "tempo_medio_pitstop", estatisticas)

            if encontrou_outlier_pit:
                base.at[indice, "situacao_pitstop"] = "outlier_temporal"
            elif encontrou_historico_pit:
                base.at[indice, "situacao_pitstop"] = "normal"

    log_df = pd.DataFrame(log_outliers, columns=COLUNAS_LOG)

    print()
    print("=" * 70)
    print("TEMPO DE VOLTA")
    print("=" * 70)
    print(base["situacao_tempo_volta"].value_counts())

    print()
    print("=" * 70)
    print("PITSTOP")
    print("=" * 70)
    print(base["situacao_pitstop"].value_counts())

    print()
    print("=" * 70)
    print("OUTLIERS REGISTRADOS")
    print("=" * 70)
    print("Total:", len(log_df))

    if not log_df.empty:
        print()
        print("Por coluna:")
        print(log_df["coluna"].value_counts())
        print()
        print("Classificacao posterior para auditoria:")
        print(log_df["classificacao_auditoria"].value_counts())
        print()
        print("Explicacoes observadas:")
        print(log_df["explicacao_observada"].value_counts())

    os.makedirs(PASTA_PROCESSADOS, exist_ok=True)
    base.to_csv(ARQUIVO_SAIDA, index=False)
    log_df.to_csv(ARQUIVO_LOG_OUTLIERS, index=False)

    print()
    print("=" * 70)
    print("ETAPA 5 CONCLUIDA")
    print("=" * 70)
    print()
    print("Base tratada:")
    print(ARQUIVO_SAIDA)
    print()
    print("Log de outliers:")
    print(ARQUIVO_LOG_OUTLIERS)
    print()
    print("Linhas:", len(base), "| Colunas:", len(base.columns))
    print()
    print("Nenhuma linha foi removida e nenhum valor de outlier foi alterado.")
