import os
import json
import time
import requests
import pandas as pd
import fastf1
from fastf1 import RateLimitExceededError

ANO_INICIO = 2018
ANO_FIM = 2025

BASE_URL = "https://api.jolpi.ca/ergast/f1"

PASTA_DADOS = "dados"

#cache local do FastF1 para evitar novos downloads das mesmas sessões
CACHE_DIR = os.path.join(PASTA_DADOS, "cache")

#checkpoint com os rounds já processados
CKPT_FILE = os.path.join(PASTA_DADOS, "fastf1_checkpoint.json")

os.makedirs(PASTA_DADOS, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)


def requisitar(url, tentativas=5):
    #tenta novamente em caso de falha de conexão ou limite da API
    for tentativa in range(tentativas):
        try:
            response = requests.get(url, timeout=30)

        except requests.exceptions.RequestException as erro:
            print("Erro de conexão:", erro)
            time.sleep(10)
            continue

        if response.status_code == 429:
            print("Limite da API. Aguardando...")
            time.sleep(30)
            continue

        if response.status_code != 200:
            print("Erro:", response.status_code)
            return None

        return response.json()

    return None


def paginar(url, limit=100):
    #busca todas as páginas de um endpoint
    paginas = []
    offset = 0

    while True:
        separador = "&" if "?" in url else "?"
        data = requisitar(f"{url}{separador}limit={limit}&offset={offset}")

        if data is None:
            break

        paginas.append(data)

        total = int(data["MRData"]["total"])
        offset += limit

        if offset >= total:
            break

        time.sleep(0.3)

    return paginas


def salvar(df, nome_arquivo):
    #salva o dataframe como csv dentro da pasta de dados
    caminho = os.path.join(PASTA_DADOS, nome_arquivo)
    df.to_csv(caminho, index=False)
    print("Salvo:", caminho)


def salvar_round_incremental(df, nome_arquivo, ano, round_num, chaves_unicidade):
    #permite refazer um round sem acumular versões antigas
    caminho = os.path.join(PASTA_DADOS, nome_arquivo)

    if os.path.exists(caminho):
        existente = pd.read_csv(caminho)
        #substitui somente o round que está sendo reprocessado
        mesmo_round = (existente["season"].astype(int) == int(ano)) & (existente["round"].astype(int) == int(round_num))
        combinado = pd.concat([existente.loc[~mesmo_round], df], ignore_index=True)
    else:
        combinado = df.copy()

    #garante uma única linha por registro esperado
    chaves_presentes = [coluna for coluna in chaves_unicidade if coluna in combinado.columns]
    if len(chaves_presentes) == len(chaves_unicidade):
        combinado = combinado.drop_duplicates(subset=chaves_unicidade, keep="last")

    combinado.to_csv(caminho, index=False)
    print("Salvo:", caminho, "| round:", f"{ano}-{round_num}")


def carregar_checkpoint():
    #le quais rounds do fastf1 já foram processados em execuções anteriores
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f:
            return set(json.load(f))
    return set()

def salvar_checkpoint(concluidos):
    with open(CKPT_FILE, "w") as f:
        json.dump(sorted(concluidos), f)

#RESULTADOS DAS CORRIDAS DE 2018 ATÉ 2025
def buscar_resultados():
    print(f"\nBuscando resultados de {ANO_INICIO} até {ANO_FIM}...")

    dados = []

    for ano in range(ANO_INICIO, ANO_FIM + 1):
        print("Temporada:", ano)
        for pagina in paginar(f"{BASE_URL}/{ano}/results.json"):
            for race in pagina["MRData"]["RaceTable"]["Races"]:
                # cada corrida traz o resultado de cada piloto
                for result in race["Results"]:
                    dados.append({
                        "season": race["season"],
                        "round": race["round"],
                        "race_name": race["raceName"],
                        "driver_id": result["Driver"]["driverId"],
                        "constructor_id": result["Constructor"]["constructorId"],
                        "grid_position": result["grid"],
                        "finish_position": result["position"],
                        "status": result["status"],
                        "points": result["points"],
                        "laps": result.get("laps", "")
                    })

    salvar(pd.DataFrame(dados), "resultados_2018_2025.csv")



#PIT STOPS 2018 ATÉ 2025
def buscar_pitstops():
    print(f"\nBuscando pit stops de {ANO_INICIO} até {ANO_FIM}...")

    dados = []

    for ano in range(ANO_INICIO, ANO_FIM + 1):
        print("Temporada:", ano)

        #busca o calendário da temporada pra saber quantos rounds ela teve
        calendario = requisitar(f"{BASE_URL}/{ano}/races.json?limit=100")
        if not calendario:
            continue
        rounds = [r["round"] for r in calendario["MRData"]["RaceTable"]["Races"]]

        for round_num in rounds:
            for pagina in paginar(f"{BASE_URL}/{ano}/{round_num}/pitstops.json"):
                for race in pagina["MRData"]["RaceTable"]["Races"]:
                    for pit in race["PitStops"]:
                        dados.append({
                            "season": ano,
                            "round": round_num,
                            "race_name": race["raceName"],
                            "driver_id": pit["driverId"],
                            "stop": pit["stop"],
                            "lap": pit["lap"],
                            "duration": pit["duration"]
                        })

            time.sleep(0.5)

    salvar(pd.DataFrame(dados), "pitstops_2018_2025.csv")


#CIRCUITOS UTILIZADOS ENTRE 2018 E 2025
def buscar_circuitos():
    print(f"\nBuscando circuitos de {ANO_INICIO} até {ANO_FIM}...")

    dados = []

    for ano in range(ANO_INICIO, ANO_FIM + 1):
        print("Temporada:", ano)
        data = requisitar(f"{BASE_URL}/{ano}/circuits.json?limit=100")
        if not data:
            continue
        for circuito in data["MRData"]["CircuitTable"]["Circuits"]:
            dados.append({
                "circuit_id": circuito["circuitId"],
                "circuit_name": circuito["circuitName"],
                "lat": circuito["Location"]["lat"],
                "long": circuito["Location"]["long"],
                "country": circuito["Location"]["country"]
            })
        time.sleep(0.5)

    #um circuito pode aparecer em várias temporadas
    df = pd.DataFrame(dados).drop_duplicates(subset="circuit_id")
    salvar(df, "circuitos_2018_2025.csv")


#PILOTOS QUE COMPETIRAM ENTRE 2018 E 2025
def buscar_pilotos():
    print(f"\nBuscando pilotos de {ANO_INICIO} até {ANO_FIM}...")

    dados = []

    for ano in range(ANO_INICIO, ANO_FIM + 1):
        print("Temporada:", ano)
        data = requisitar(f"{BASE_URL}/{ano}/drivers.json?limit=100")
        if not data:
            continue
        for driver in data["MRData"]["DriverTable"]["Drivers"]:
            dados.append({
                "driver_id": driver["driverId"],
                "code": driver.get("code", ""),
                "given_name": driver["givenName"],
                "family_name": driver["familyName"],
                "date_of_birth": driver.get("dateOfBirth", ""),
                "nationality": driver.get("nationality", "")
            })
        time.sleep(0.5)

    #um piloto pode competir em vários anos
    df = pd.DataFrame(dados).drop_duplicates(subset="driver_id")
    salvar(df, "pilotos_2018_2025.csv")



#CALENDARIO: EM QUAL CIRCUITO CADA CORRIDA ACONTECEU
def buscar_calendario_circuitos():
    print(f"\nBuscando calendário de circuitos de {ANO_INICIO} até {ANO_FIM}...")

    dados = []

    for ano in range(ANO_INICIO, ANO_FIM + 1):
        print("Temporada:", ano)
        data = requisitar(f"{BASE_URL}/{ano}/races.json?limit=100")
        if not data:
            continue
        for race in data["MRData"]["RaceTable"]["Races"]:
            dados.append({
                "season": race["season"],
                "round": race["round"],
                "race_name": race["raceName"],
                "circuit_id": race["Circuit"]["circuitId"]
            })

        time.sleep(0.5)

    salvar(pd.DataFrame(dados), "calendario_circuitos_2018_2025.csv")



# FASTF1 - QUALIFYING, VOLTAS E CLIMA DE 2018 ATÉ 2025
def cols_disponiveis(df, colunas):
    return [coluna for coluna in colunas if coluna in df.columns]


def carregar_sessao(ano, round_num, tipo, tentativas=5, espera=900):
    for tentativa in range(tentativas):
        try:
            sessao = fastf1.get_session(ano, round_num, tipo)
            #carrega clima apenas nas sessões de corrida
            sessao.load(laps=True, telemetry=False, weather=(tipo == "R"), messages=False)
            return sessao
        except RateLimitExceededError:
            print(f"Limite da API atingido, aguardando {espera}s antes de tentar de novo...")
            time.sleep(espera)

    #mantém o round pendente quando o limite da API persiste
    raise RateLimitExceededError("limite da api continua ativo após várias tentativas")


def buscar_fastf1():
    print("\nBuscando dados do FastF1...")

    concluidos = carregar_checkpoint()

    for ano in range(ANO_INICIO, ANO_FIM + 1):
        print("\nFastF1 - Temporada:", ano)
        try:
            schedule = fastf1.get_event_schedule(ano, include_testing=False)
        except Exception as erro:
            print("Erro ao buscar calendário:", erro)
            continue

        rounds = schedule["RoundNumber"].dropna().astype(int)

        for round_num in rounds:
            chave = f"{ano}-{round_num}"
             #evita repetir rounds concluídos em execuções anteriores
            if chave in concluidos:
                print("Round:", round_num, "(já processado, pulando)")
                continue

            print("Round:", round_num)

            # qualquer falha mantém o round pendente para uma próxima execução
            limite_atingido = False
            round_falhou = False

            # QUALIFYING
            try:
                session_q = carregar_sessao(ano, round_num, "Q")

                resultados = session_q.results

                colunas = cols_disponiveis(resultados, ["Abbreviation", "Position", "Q1", "Q2", "Q3"])

                df_q = resultados[colunas].copy()

                df_q = df_q.rename(columns={"Abbreviation": "Driver", "Position": "position"})

                df_q["season"] = ano
                df_q["round"] = round_num

                salvar_round_incremental(df_q, "fastf1_qualifying_2018_2025.csv", ano, round_num, ["season", "round", "Driver"],)

            except RateLimitExceededError:
                print("Limite da api esgotado no qualifying, fica pra próxima execução:", ano, round_num)
                limite_atingido = True
                round_falhou = True

            except Exception as erro:
                print("Erro no qualifying:", ano, round_num, erro)
                round_falhou = True

            #CORRIDA/VOLTAS
            try:
                session_r = carregar_sessao(ano, round_num, "R")

                colunas_laps = cols_disponiveis(session_r.laps, ["Driver", "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife", "Stint", "TrackStatus", "FreshTyre", "PitInTime", "PitOutTime"])

                df_laps = session_r.laps[colunas_laps].copy()
                df_laps["season"] = ano
                df_laps["round"] = round_num
                salvar_round_incremental(df_laps, "fastf1_laps_2018_2025.csv", ano, round_num, ["season", "round", "Driver", "LapNumber"],)

                # CLIMA
                if session_r.weather_data is not None and len(session_r.weather_data) > 0:
                    colunas_weather = cols_disponiveis(session_r.weather_data, ["Time", "AirTemp", "Humidity", "Rainfall", "TrackTemp", "WindSpeed"])

                    df_weather = session_r.weather_data[colunas_weather].copy()
                    df_weather["season"] = ano
                    df_weather["round"] = round_num
                    salvar_round_incremental(df_weather, "fastf1_weather_2018_2025.csv", ano, round_num, ["season", "round", "Time"],)

            except RateLimitExceededError:
                print("Limite da api esgotado na corrida, fica pra próxima execução:", ano, round_num)
                limite_atingido = True
                round_falhou = True

            except Exception as erro:
                print("Erro na corrida:", ano, round_num, erro)
                round_falhou = True

            #registra o checkpoint somente quando todas as etapas terminam corretamente
            if not limite_atingido and not round_falhou:
                concluidos.add(chave)
                salvar_checkpoint(concluidos)

            time.sleep(1)

    print("\nFastF1 finalizado (ou pausado pelo limite da api - rode o script de novo mais tarde pra continuar de onde parou).")


# EXECUÇÃO DO PROGRAMA
if __name__ == "__main__":
    buscar_resultados()
    buscar_pitstops()
    buscar_circuitos()
    buscar_pilotos()
    buscar_calendario_circuitos()
    buscar_fastf1()

    print("\nExtração finalizada.")
    print("Arquivos salvos na pasta 'dados'.")