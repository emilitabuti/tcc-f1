#junta todos os csv brutos numa unica tabela, com uma linha por (temporada, corrida, piloto)

import os
import pandas as pd

PASTA_DADOS = "dados"

#pasta onde salvamos a base consolidada
PASTA_SAIDA = os.path.join(PASTA_DADOS, "processados")
os.makedirs(PASTA_SAIDA, exist_ok=True)

#a api marca o Sakhir GP (2020, layout "Outer Circuit") com o mesmo
#circuit_id do Bahrain GP normal ("bahrain"), mas o circuitos_manual.csv
#trata como um circuito a parte ("bahrain_outer") porque o layout da pista
#foi diferente. então corrigimos aqui manualmente
CORRIGIR_CIRCUITO_POR_CORRIDA = {
    "Sakhir Grand Prix": "bahrain_outer",
}

def carregar_csv(nome_arquivo):
    caminho = os.path.join(PASTA_DADOS, nome_arquivo)
    return pd.read_csv(caminho)

def validar_unicidade(df, colunas, nome):
    """verifica se a tabela nao tem chaves duplicadas antes do merge"""
    duplicados = df.duplicated(subset=colunas, keep=False)
    if duplicados.any():
        exemplos = df.loc[duplicados, colunas].head(20)
        raise ValueError(
            f"{nome} possui chaves duplicadas em {colunas}:\n"
            + exemplos.to_string(index=False)
        )

def validar_sem_mapeamento_faltante(df, coluna_origem, coluna_mapeada, nome):
    """verifica se algum codigo do FastF1 nao foi convertido para driver_id"""
    faltantes = df[df[coluna_origem].notna() & df[coluna_mapeada].isna()]
    if faltantes.empty:
        return

    codigos = sorted(faltantes[coluna_origem].dropna().unique())
    raise ValueError(
        f"{nome} possui pilotos FastF1 sem mapeamento para driver_id: "
        + ", ".join(codigos)
    )

def validar_base_consolidada(base):
    """valida na base final se tem uma linha por piloto em cada corrida"""
    print("Validando base consolidada...")

    chaves = ["season", "round", "driver_id"]
    colunas_obrigatorias = [
        "season",
        "round",
        "race_name",
        "driver_id",
        "constructor_id",
        "circuit_id",
        "finish_position",
        "grid_position",
    ]

    faltando = [coluna for coluna in colunas_obrigatorias if coluna not in base.columns]
    if faltando:
        raise ValueError("Colunas obrigatorias ausentes: " + ", ".join(faltando))

    duplicados = base.duplicated(subset=chaves, keep=False)
    if duplicados.any():
        exemplos = base.loc[duplicados, chaves + ["race_name"]].head(20)
        raise ValueError(
            "Base consolidada possui linhas duplicadas por season/round/driver_id:\n"
            + exemplos.to_string(index=False)
        )

    for coluna in colunas_obrigatorias:
        ausentes = base[coluna].isna()
        if base[coluna].dtype == object:
            ausentes = ausentes | (base[coluna].astype(str).str.strip() == "")
        if ausentes.any():
            raise ValueError(
                f"Coluna obrigatoria com valores ausentes: {coluna} ({int(ausentes.sum())})"
            )

    finish_position = pd.to_numeric(base["finish_position"], errors="coerce")
    if finish_position.isna().any() or not finish_position.between(1, 25).all():
        raise ValueError("finish_position fora do intervalo esperado de 1 a 25.")

    grid_position = pd.to_numeric(base["grid_position"], errors="coerce")
    if grid_position.isna().any() or not grid_position.between(0, 25).all():
        raise ValueError("grid_position fora do intervalo esperado de 0 a 25.")

    qtd_por_corrida = base.groupby(["season", "round"])["driver_id"].nunique()
    corridas_suspeitas = qtd_por_corrida[(qtd_por_corrida < 15) | (qtd_por_corrida > 25)]
    if not corridas_suspeitas.empty:
        raise ValueError(
            "Corridas com quantidade suspeita de pilotos:\n"
            + corridas_suspeitas.to_string()
        )

    print("Validacao da base consolidada concluida com sucesso.")

def tempo_para_segundos(coluna):
    return pd.to_timedelta(coluna, errors="coerce").dt.total_seconds()

def duracao_pitstop_para_segundos(valor):
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
    """pega os resultados das corridas e descobre o circuito de cada uma"""
    print("Montando resultados com circuito...")

    resultados = carregar_csv("resultados_2018_2025.csv")
    calendario = carregar_csv("calendario_circuitos_2018_2025.csv")

    calendario = calendario[["season", "round", "circuit_id"]]
    validar_unicidade(calendario, ["season", "round"], "calendario_circuitos")
    validar_unicidade(resultados, ["season", "round", "driver_id"], "resultados")
    resultados = resultados.merge(
        calendario,
        on=["season", "round"],
        how="left",
        validate="many_to_one",
    )

    # corrige o caso especial do Sakhir GP (CORRIGIR_CIRCUITO_POR_CORRIDA)
    for race_name, circuito_correto in CORRIGIR_CIRCUITO_POR_CORRIDA.items():
        resultados.loc[resultados["race_name"] == race_name, "circuit_id"] = circuito_correto

    # verifica se sobrou alguma corrida sem circuito mapeado
    sem_circuito = resultados[resultados["circuit_id"].isna()]
    if len(sem_circuito) > 0:
        print("Aviso: corridas sem circuito mapeado ->", sem_circuito["race_name"].unique())

    return resultados

# CIRCUITOS
def montar_circuitos():
    """junta a localizacao (api) com os dados manuais de cada circuito"""
    print("Montando tabela de circuitos...")

    circuitos_api = carregar_csv("circuitos_2018_2025.csv")
    circuitos_manual = carregar_csv("circuitos_manual.csv")

    # usa lat/long/country da api, o resto (nome, altitude,...) vem do manual
    localizacao = circuitos_api[["circuit_id", "lat", "long", "country"]].drop_duplicates(subset=["circuit_id"])
    validar_unicidade(circuitos_manual, ["circuit_id"], "circuitos_manual")
    validar_unicidade(localizacao, ["circuit_id"], "circuitos_api")

    circuitos = circuitos_manual.merge(
        localizacao,
        on="circuit_id",
        how="left",
        validate="one_to_one",
    )

    # a api nao conhece o "bahrain_outer" (id criado so no circuitos_manual.csv
    # pra separar o layout do Sakhir GP 2020), entao ele fica sem lat/long/country
    # depois do merge. corrigimos usando os dados do "bahrain", que e o mesmo
    # circuito, so com o layout da pista diferente.
    bahrain = circuitos[circuitos["circuit_id"] == "bahrain"][["lat", "long", "country"]].dropna()
    if not bahrain.empty:
        referencia = bahrain.iloc[0]
        mascara = circuitos["circuit_id"] == "bahrain_outer"
        circuitos.loc[mascara, "lat"] = circuitos.loc[mascara, "lat"].fillna(referencia["lat"])
        circuitos.loc[mascara, "long"] = circuitos.loc[mascara, "long"].fillna(referencia["long"])
        circuitos.loc[mascara, "country"] = circuitos.loc[mascara, "country"].fillna(referencia["country"])

    return circuitos

# PIT STOPS
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

    validar_unicidade(agregado, ["season", "round", "driver_id"], "pitstops_agregado")
    return agregado

def montar_disponibilidade_pitstops():
    """marca as corridas em que o arquivo de pit stops possui pelo menos um registro"""
    pitstops = carregar_csv("pitstops_2018_2025.csv")
    disponibilidade = pitstops[["season", "round"]].drop_duplicates().copy()
    disponibilidade["pitstop_dado_disponivel"] = 1
    validar_unicidade(disponibilidade, ["season", "round"], "pitstop_disponibilidade")
    return disponibilidade

# VOLTAS DO FASTF1
def montar_laps_agregado(codigo_para_driver_id):
    """agrega as voltas de cada piloto numa corrida: tempo medio, melhor volta,..."""
    print("Agregando voltas (fastf1)...")

    laps = carregar_csv("fastf1_laps_2018_2025.csv")

    # troca o codigo de 3 letras do fastf1 pelo driver_id
    laps["driver_id"] = laps["Driver"].map(codigo_para_driver_id)
    validar_sem_mapeamento_faltante(laps, "Driver", "driver_id", "fastf1_laps")
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

    agregado = agregado.merge(
        composto_mais_usado,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )

    validar_unicidade(agregado, ["season", "round", "driver_id"], "laps_agregado")
    return agregado

# QUALIFYING
def montar_qualifying(codigo_para_driver_id):
    """pega a posicao de largada e os tempos de classificacao de cada piloto"""
    print("Montando qualifying (fastf1)...")

    quali = carregar_csv("fastf1_qualifying_2018_2025.csv")

    quali["driver_id"] = quali["Driver"].map(codigo_para_driver_id)
    validar_sem_mapeamento_faltante(quali, "Driver", "driver_id", "fastf1_qualifying")
    quali["Q1_s"] = tempo_para_segundos(quali["Q1"])
    quali["Q2_s"] = tempo_para_segundos(quali["Q2"])
    quali["Q3_s"] = tempo_para_segundos(quali["Q3"])
    quali = quali.rename(columns={"position": "qualifying_position"})

    colunas = ["season", "round", "driver_id", "qualifying_position", "Q1_s", "Q2_s", "Q3_s"]
    quali = quali[colunas]
    validar_unicidade(quali, ["season", "round", "driver_id"], "qualifying")
    return quali

# CLIMA
def montar_weather_agregado():
    """agrega o clima da corrida inteira (uma linha por temporada e round)"""
    print("Agregando clima (fastf1)...")

    weather = carregar_csv("fastf1_weather_2018_2025.csv")

    agregado = weather.groupby(["season", "round"]).agg(
        temp_ar_media=("AirTemp", "mean"),
        temp_pista_media=("TrackTemp", "mean"),
        umidade_media=("Humidity", "mean"),
        vento_media=("WindSpeed", "mean"),
        choveu=("Rainfall", "max"),
    ).reset_index()

    validar_unicidade(agregado, ["season", "round"], "weather_agregado")
    return agregado

if __name__ == "__main__":
    pilotos = carregar_csv("pilotos_2018_2025.csv")

    # monta o mapa (codigo de 3 letras -> driver_id)
    codigo_para_driver_id = (pilotos.dropna(subset=["code"]).set_index("code")["driver_id"].to_dict())

    resultados = montar_resultados_com_circuito()
    circuitos = montar_circuitos()
    pitstops_agg = montar_pitstops_agregado()
    pitstops_disponibilidade = montar_disponibilidade_pitstops()
    laps_agg = montar_laps_agregado(codigo_para_driver_id)
    quali = montar_qualifying(codigo_para_driver_id)
    weather_agg = montar_weather_agregado()

    print("Juntando tudo numa base so...")

    validar_unicidade(pilotos, ["driver_id"], "pilotos")

    base = resultados.merge(circuitos, on="circuit_id", how="left", validate="many_to_one")
    base = base.merge(pilotos, on="driver_id", how="left", validate="many_to_one")
    base = base.merge(
        pitstops_agg,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    base = base.merge(
        pitstops_disponibilidade,
        on=["season", "round"],
        how="left",
        validate="many_to_one",
    )
    base = base.merge(
        laps_agg,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    base = base.merge(
        quali,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    base = base.merge(
        weather_agg,
        on=["season", "round"],
        how="left",
        validate="many_to_one"
    )

    base["pitstop_dado_disponivel"] = base["pitstop_dado_disponivel"].fillna(0).astype(int)

    validar_base_consolidada(base)

    caminho_saida = os.path.join(PASTA_SAIDA, "base_consolidada_2018_2025.csv")
    base.to_csv(caminho_saida, index=False)

    print("Base consolidada salva em:", caminho_saida)
    print("Linhas:", len(base), "| Colunas:", len(base.columns))
