"""
Junta todos os csv brutos da pasta dados/ numa unica tabela, com uma linha
por (temporada, corrida, piloto).
"""

import os
import pandas as pd


# pasta onde estao os dados brutos
PASTA_DADOS = "dados"

# pasta onde vamos salvar a base consolidada
PASTA_SAIDA = os.path.join(PASTA_DADOS, "processados")
os.makedirs(PASTA_SAIDA, exist_ok=True)


# a api marca o Sakhir GP (2020, layout "Outer Circuit") com o mesmo
# circuit_id do Bahrain GP normal ("bahrain"), mas o circuitos_manual.csv
# trata como um circuito a parte ("bahrain_outer") porque o layout da pista
# foi diferente. corrige na mao so esse caso especial, que nao tem
# como resolver so com o circuit_id que vem da api.
CORRIGIR_CIRCUITO_POR_CORRIDA = {
    "Sakhir Grand Prix": "bahrain_outer",
}


def carregar_csv(nome_arquivo):
    """le um csv de dentro da pasta de dados brutos"""
    caminho = os.path.join(PASTA_DADOS, nome_arquivo)
    return pd.read_csv(caminho)


def tempo_para_segundos(coluna):
    """converte uma coluna de tempo (formato "0 days 00:01:32.123") para segundos"""
    return pd.to_timedelta(coluna, errors="coerce").dt.total_seconds()


def duracao_pitstop_para_segundos(valor):
    """converte a duracao do pitstop pra segundos."""
    valor = str(valor)

    if ":" in valor:
        minutos, segundos = valor.split(":")
        return int(minutos) * 60 + float(segundos)

    try:
        return float(valor)
    except ValueError:
        return None



# RESULTADOS + CIRCUITO
def montar_resultados_com_circuito():
    """pega os resultados das corridas e descobre o circuito de cada uma

    o circuito vem do calendario_circuitos_2018_2025.csv, que traz o
    circuit_id oficial de cada (season, round) direto da api.
    """

    print("Montando resultados com circuito...")

    resultados = carregar_csv("resultados_2018_2025.csv")
    calendario = carregar_csv("calendario_circuitos_2018_2025.csv")

    calendario = calendario[["season", "round", "circuit_id"]]
    resultados = resultados.merge(calendario, on=["season", "round"], how="left")

    # corrige o caso especial do Sakhir GP (CORRIGIR_CIRCUITO_POR_CORRIDA)
    for race_name, circuito_correto in CORRIGIR_CIRCUITO_POR_CORRIDA.items():
        resultados.loc[resultados["race_name"] == race_name, "circuit_id"] = circuito_correto

    # avisa se sobrou alguma corrida sem circuito mapeado
    sem_circuito = resultados[resultados["circuit_id"].isna()]
    if len(sem_circuito) > 0:
        print("Aviso: corridas sem circuito mapeado ->", sem_circuito["race_name"].unique())

    return resultados


# CIRCUITOS (localizacao da api + dados manuais de altitude, curvas,...)
def montar_circuitos():
    """junta a localizacao (vinda da api) com os dados manuais de cada circuito"""

    print("Montando tabela de circuitos...")

    circuitos_api = carregar_csv("circuitos_2018_2025.csv")
    circuitos_manual = carregar_csv("circuitos_manual.csv")

    # so usa lat/long/country da api, o resto (nome, altitude,...) vem do manual
    localizacao = circuitos_api[["circuit_id", "lat", "long", "country"]]

    circuitos = circuitos_manual.merge(localizacao, on="circuit_id", how="left")

    return circuitos


# PIT STOPS (agregado por corrida e piloto)
def montar_pitstops_agregado():
    """conta quantas paradas cada piloto fez numa corrida e o tempo gasto nelas"""

    print("Agregando pit stops...")

    pitstops = carregar_csv("pitstops_2018_2025.csv")
    pitstops["duration"] = pitstops["duration"].apply(duracao_pitstop_para_segundos)

    agregado = pitstops.groupby(["season", "round", "driver_id"]).agg(
        num_pitstops=("stop", "count"),
        tempo_total_pitstop=("duration", "sum"),
        tempo_medio_pitstop=("duration", "mean"),
    ).reset_index()

    return agregado


# VOLTAS DO FASTF1 (agregado por corrida e piloto)
def montar_laps_agregado(codigo_para_driver_id):
    """resume as voltas de cada piloto numa corrida: tempo medio, melhor volta,..."""

    print("Agregando voltas (fastf1)...")

    laps = carregar_csv("fastf1_laps_2018_2025.csv")

    # troca o codigo de 3 letras do fastf1 pelo driver_id usado no resto da base
    laps["driver_id"] = laps["Driver"].map(codigo_para_driver_id)
    laps["LapTime_s"] = tempo_para_segundos(laps["LapTime"])

    agregado = laps.groupby(["season", "round", "driver_id"]).agg(
        fastf1_avg_lap_time=("LapTime_s", "mean"),
        fastf1_best_lap_time=("LapTime_s", "min"),
        fastf1_num_voltas=("LapNumber", "count"),
        fastf1_num_stints=("Stint", "nunique"),
        fastf1_tyre_life_media=("TyreLife", "mean"),
    ).reset_index()

    # composto de pneu mais usado por cada piloto na corrida
    composto_mais_usado = (
        laps.groupby(["season", "round", "driver_id"])["Compound"]
        .agg(lambda serie: serie.mode().iloc[0] if not serie.mode().empty else "UNKNOWN")
        .reset_index()
        .rename(columns={"Compound": "tire_compound_predominante"})
    )

    agregado = agregado.merge(composto_mais_usado, on=["season", "round", "driver_id"], how="left")

    return agregado


# QUALIFYING (fastf1)
def montar_qualifying(codigo_para_driver_id):
    """pega a posicao de largada e os tempos de classificacao de cada piloto"""

    print("Montando qualifying (fastf1)...")

    quali = carregar_csv("fastf1_qualifying_2018_2025.csv")

    quali["driver_id"] = quali["Driver"].map(codigo_para_driver_id)
    quali["Q1_s"] = tempo_para_segundos(quali["Q1"])
    quali["Q2_s"] = tempo_para_segundos(quali["Q2"])
    quali["Q3_s"] = tempo_para_segundos(quali["Q3"])
    quali = quali.rename(columns={"position": "qualifying_position"})

    colunas = ["season", "round", "driver_id", "qualifying_position", "Q1_s", "Q2_s", "Q3_s"]
    return quali[colunas]


# CLIMA (fastf1)
def montar_weather_agregado():
    """resume o clima da corrida inteira (uma linha por temporada+round)"""

    print("Agregando clima (fastf1)...")

    weather = carregar_csv("fastf1_weather_2018_2025.csv")

    agregado = weather.groupby(["season", "round"]).agg(
        temp_ar_media=("AirTemp", "mean"),
        temp_pista_media=("TrackTemp", "mean"),
        umidade_media=("Humidity", "mean"),
        vento_media=("WindSpeed", "mean"),
        choveu=("Rainfall", "max"),  # se choveu em algum momento, marca a corrida como chuvosa
    ).reset_index()

    return agregado


# EXECUÇÃO DO PROGRAMA
if __name__ == "__main__":

    pilotos = carregar_csv("pilotos_2018_2025.csv")

    # monta o mapa codigo-de-3-letras -> driver_id a partir da propria coluna
    codigo_para_driver_id = (
        pilotos.dropna(subset=["code"])
        .set_index("code")["driver_id"]
        .to_dict()
    )

    resultados = montar_resultados_com_circuito()
    circuitos = montar_circuitos()
    pitstops_agg = montar_pitstops_agregado()
    laps_agg = montar_laps_agregado(codigo_para_driver_id)
    quali = montar_qualifying(codigo_para_driver_id)
    weather_agg = montar_weather_agregado()

    print("Juntando tudo numa base so...")

    base = resultados.merge(circuitos, on="circuit_id", how="left")
    base = base.merge(pilotos, on="driver_id", how="left")
    base = base.merge(pitstops_agg, on=["season", "round", "driver_id"], how="left")
    base = base.merge(laps_agg, on=["season", "round", "driver_id"], how="left")
    base = base.merge(quali, on=["season", "round", "driver_id"], how="left")
    base = base.merge(weather_agg, on=["season", "round"], how="left")

    caminho_saida = os.path.join(PASTA_SAIDA, "base_consolidada_2018_2025.csv")
    base.to_csv(caminho_saida, index=False)

    print("Base consolidada salva em:", caminho_saida)
    print("Linhas:", len(base), "| Colunas:", len(base.columns))
