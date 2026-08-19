import os
import json
import time
import requests
import pandas as pd
import fastf1
from fastf1.exceptions import RateLimitExceededError


# endpoint base da api jolpica
BASE_URL = "https://api.jolpi.ca/ergast/f1"

# pasta onde os arquivos csv vão ser salvos
PASTA_DADOS = "dados"

# pasta usada pelo fastf1 pra guardar o cache das sessões
CACHE_DIR = os.path.join(PASTA_DADOS, "cache")

# arquivo que guarda quais rounds do fastf1 já foram baixados com sucesso
CKPT_FILE = os.path.join(PASTA_DADOS, "fastf1_checkpoint.json")

# cria as pastas caso ainda não existam
os.makedirs(PASTA_DADOS, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ativa o cache do fastf1, assim ele não baixa tudo de novo
fastf1.Cache.enable_cache(CACHE_DIR)


def requisitar(url, tentativas=5):
    """faz um GET na url, tentando de novo se der erro de conexão ou limite de requisições (429)"""

    for _ in range(tentativas):

        try:
            # faz a requisição pra api
            response = requests.get(url, timeout=30)

        except requests.exceptions.RequestException as erro:
            # deu erro de conexão, espera um pouco e tenta de novo
            print("Erro de conexão:", erro)
            time.sleep(10)
            continue

        # código 429 significa excesso de requisições
        if response.status_code == 429:
            print("Limite da API. Aguardando...")
            time.sleep(30)
            continue

        # se der outro erro, desiste
        if response.status_code != 200:
            print("Erro:", response.status_code)
            return None

        # deu certo, converte a resposta pra json
        return response.json()

    return None


def paginar(url, limit=100):
    """percorre todas as páginas de um endpoint paginado da jolpica, uma de cada vez"""

    offset = 0

    while True:
        # monta a url da página atual
        separador = "&" if "?" in url else "?"
        data = requisitar(f"{url}{separador}limit={limit}&offset={offset}")

        # se a requisição falhou, encerra
        if data is None:
            return

        # devolve a página
        yield data

        # total de registros que o endpoint tem no total
        total = int(data["MRData"]["total"])

        # avança pra próxima página
        offset += limit
        time.sleep(0.3)

        # para quando tiver percorrido todas as páginas
        if offset >= total:
            return


def salvar(df, nome_arquivo):
    """salva o dataframe como csv dentro da pasta de dados."""
    caminho = os.path.join(PASTA_DADOS, nome_arquivo)
    df.to_csv(caminho, index=False)
    print("Salvo:", caminho)


def salvar_incremental(df, nome_arquivo):
    """acrescenta linhas num csv, criando o cabeçalho só se o arquivo ainda não existir."""
    caminho = os.path.join(PASTA_DADOS, nome_arquivo)
    existe = os.path.exists(caminho)
    df.to_csv(caminho, mode="a", header=not existe, index=False)


def carregar_checkpoint():
    """le quais rounds do fastf1 já foram processados em execuções anteriores."""
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f:
            return set(json.load(f))
    return set()


def salvar_checkpoint(concluidos):
    with open(CKPT_FILE, "w") as f:
        json.dump(sorted(concluidos), f)



# RESULTADOS DAS CORRIDAS DE 2018 ATÉ 2025
def buscar_resultados():

    print("\nBuscando resultados de 2018 até 2025...")

    # lista que vai guardar cada linha de resultado encontrada
    dados = []

    # percorre uma temporada de cada vez, de 2018 até 2025
    for ano in range(2018, 2026):
        print("Temporada:", ano)

        # busca todas as páginas de resultados dessa temporada
        for pagina in paginar(f"{BASE_URL}/{ano}/results.json"):

            # cada página traz uma lista de corridas
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

    # transforma tudo em dataframe e salva em um csv
    salvar(pd.DataFrame(dados), "resultados_2018_2025.csv")



# PIT STOPS 2018 ATÉ 2025
def buscar_pitstops():

    print("\nBuscando pit stops de 2018 até 2025...")

    # lista que vai guardar todos os pit stops encontrados
    dados = []

    # percorre todas as temporadas
    for ano in range(2018, 2026):
        print("Temporada:", ano)

        # busca o calendário da temporada pra saber quantos rounds ela teve
        calendario = requisitar(f"{BASE_URL}/{ano}/races.json?limit=100")
        if not calendario:
            continue

        rounds = [r["round"] for r in calendario["MRData"]["RaceTable"]["Races"]]

        # percorre cada round (gp) da temporada
        for round_num in rounds:

            # busca todas as páginas de pit stops desse round
            for pagina in paginar(f"{BASE_URL}/{ano}/{round_num}/pitstops.json"):
                for race in pagina["MRData"]["RaceTable"]["Races"]:

                    # percorre cada pit stop dessa corrida
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

            # pausa antes de passar pro próximo round
            time.sleep(0.5)

    # transforma tudo em dataframe e salva em um csv
    salvar(pd.DataFrame(dados), "pitstops_2018_2025.csv")



# CIRCUITOS UTILIZADOS ENTRE 2018 E 2025
def buscar_circuitos():

    print("\nBuscando circuitos de 2018 até 2025...")

    # lista que vai guardar os circuitos encontrados
    dados = []

    # faz a busca dos circuitos de cada temporada
    for ano in range(2018, 2026):
        print("Temporada:", ano)

        data = requisitar(f"{BASE_URL}/{ano}/circuits.json?limit=100")
        if not data:
            continue

        # percorre cada circuito retornado e guarda os campos que interessam
        for circuito in data["MRData"]["CircuitTable"]["Circuits"]:
            dados.append({
                "circuit_id": circuito["circuitId"],
                "circuit_name": circuito["circuitName"],
                "lat": circuito["Location"]["lat"],
                "long": circuito["Location"]["long"],
                "country": circuito["Location"]["country"]
            })

        time.sleep(0.5)

    # um circuito pode aparecer em várias temporadas, então remove as repetições pelo id
    df = pd.DataFrame(dados).drop_duplicates(subset="circuit_id")
    salvar(df, "circuitos_2018_2025.csv")



# PILOTOS QUE COMPETIRAM ENTRE 2018 E 2025
def buscar_pilotos():

    print("\nBuscando pilotos de 2018 até 2025...")

    # lista que vai guardar os pilotos encontrados
    dados = []

    # busca os pilotos de cada temporada
    for ano in range(2018, 2026):
        print("Temporada:", ano)

        data = requisitar(f"{BASE_URL}/{ano}/drivers.json?limit=100")
        if not data:
            continue

        # percorre cada piloto retornado e guarda os campos que interessam
        for driver in data["MRData"]["DriverTable"]["Drivers"]:
            dados.append({
                "driver_id": driver["driverId"],
                "given_name": driver["givenName"],
                "family_name": driver["familyName"],
                "date_of_birth": driver.get("dateOfBirth", ""),
                "nationality": driver.get("nationality", "")
            })

        time.sleep(0.5)

    # um piloto pode competir em vários anos, então remove os repetidos
    df = pd.DataFrame(dados).drop_duplicates(subset="driver_id")
    salvar(df, "pilotos_2018_2025.csv")



# FASTF1 - QUALIFYING, VOLTAS E CLIMA DE 2018 ATÉ 2025
def cols_disponiveis(df, colunas):
    """devolve só as colunas da lista que realmente existem no dataframe"""
    return [coluna for coluna in colunas if coluna in df.columns]


def carregar_sessao(ano, round_num, tipo, tentativas=5, espera=900):
    """carrega uma sessão do fastf1, esperando e tentando de novo se bater no limite da api.

    se esgotar as tentativas e o limite continuar valendo, relança o erro pra quem chamou
    saber que esse round não foi concluído (e assim não marcar o checkpoint como feito).
    """
    for tentativa in range(tentativas):
        try:
            sessao = fastf1.get_session(ano, round_num, tipo)
            sessao.load(laps=True, telemetry=False, weather=(tipo == "R"), messages=False)
            return sessao
        except RateLimitExceededError:
            print(f"Limite da API atingido, aguardando {espera}s antes de tentar de novo...")
            time.sleep(espera)

    raise RateLimitExceededError("limite da api continua ativo após várias tentativas")


def buscar_fastf1():

    print("\nBuscando dados do FastF1...")

    # rounds que já foram baixados com sucesso em execuções anteriores
    concluidos = carregar_checkpoint()

    # percorre cada temporada
    for ano in range(2018, 2026):
        print("\nFastF1 - Temporada:", ano)

        try:
            # busca o calendário da temporada
            schedule = fastf1.get_event_schedule(ano, include_testing=False)
        except Exception as erro:
            print("Erro ao buscar calendário:", erro)
            continue

        # pega só os números dos rounds
        rounds = schedule["RoundNumber"].dropna().astype(int)

        # percorre cada gp da temporada
        for round_num in rounds:
            chave = f"{ano}-{round_num}"

            # pula rounds que já foram baixados numa execução anterior
            if chave in concluidos:
                print("Round:", round_num, "(já processado, pulando)")
                continue

            print("Round:", round_num)

            # se bater no limite da api mesmo depois das tentativas, não marca
            # o round como concluído, pra ele ser retomado na próxima execução
            limite_atingido = False

            # QUALIFYING
            try:
                session_q = carregar_sessao(ano, round_num, "Q")

                resultados = session_q.results

                # verifica quais colunas que a gente quer existem
                colunas = cols_disponiveis(resultados, ["Abbreviation", "Position", "Q1", "Q2", "Q3"])

                # cria um dataframe só com essas colunas
                df_q = resultados[colunas].copy()

                # renomeia algumas colunas
                df_q = df_q.rename(columns={"Abbreviation": "Driver", "Position": "position"})

                # adiciona temporada e round pra identificar de onde veio a linha
                df_q["season"] = ano
                df_q["round"] = round_num

                salvar_incremental(df_q, "fastf1_qualifying_2018_2025.csv")

            except RateLimitExceededError:
                print("Limite da api esgotado no qualifying, fica pra próxima execução:", ano, round_num)
                limite_atingido = True

            except Exception as erro:
                print("Erro no qualifying:", ano, round_num, erro)

            # CORRIDA / VOLTAS
            try:
                session_r = carregar_sessao(ano, round_num, "R")

                # colunas de volta que a gente quer salvar
                colunas_laps = cols_disponiveis(session_r.laps, [
                    "Driver", "LapNumber", "LapTime", "Sector1Time", "Sector2Time",
                    "Sector3Time", "Compound", "TyreLife", "Stint", "TrackStatus",
                    "FreshTyre", "PitInTime", "PitOutTime"
                ])

                df_laps = session_r.laps[colunas_laps].copy()
                df_laps["season"] = ano
                df_laps["round"] = round_num
                salvar_incremental(df_laps, "fastf1_laps_2018_2025.csv")

                # CLIMA
                # só salva o clima se a sessão realmente trouxe esses dados
                if session_r.weather_data is not None and len(session_r.weather_data) > 0:
                    colunas_weather = cols_disponiveis(
                        session_r.weather_data,
                        ["AirTemp", "Humidity", "Rainfall", "TrackTemp", "WindSpeed"]
                    )

                    df_weather = session_r.weather_data[colunas_weather].copy()
                    df_weather["season"] = ano
                    df_weather["round"] = round_num
                    salvar_incremental(df_weather, "fastf1_weather_2018_2025.csv")

            except RateLimitExceededError:
                print("Limite da api esgotado na corrida, fica pra próxima execução:", ano, round_num)
                limite_atingido = True

            except Exception as erro:
                print("Erro na corrida:", ano, round_num, erro)

            # só marca como concluído se não foi o limite da api que atrapalhou
            if not limite_atingido:
                concluidos.add(chave)
                salvar_checkpoint(concluidos)

            # pausa entre cada gp pra não sobrecarregar a fonte de dados
            time.sleep(1)

    print("\nFastF1 finalizado (ou pausado pelo limite da api - rode o script de novo mais tarde pra continuar de onde parou).")



# EXECUÇÃO DO PROGRAMA
if __name__ == "__main__":

    # busca os resultados das corridas
    buscar_resultados()

    # busca os pit stops
    buscar_pitstops()

    # busca os circuitos
    buscar_circuitos()

    # busca os pilotos
    buscar_pilotos()

    # busca qualifying, voltas e clima pelo fastf1
    buscar_fastf1()

    print("\nExtração finalizada.")
    print("Arquivos salvos na pasta 'dados'.")