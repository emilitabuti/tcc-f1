# Extração FastF1 - qualifying, corridas e clima, 2018-2025.
# Saídas: fastf1_qualifying_2018_2025.csv, fastf1_laps_2018_2025.csv, fastf1_weather_2018_2025.csv
# Nota: usa fastf1_checkpoint_v2.json para forcar re-extracao com as novas colunas.

import fastf1
import pandas as pd
import os
import json
import time
import socket
import logging
import traceback
from datetime import datetime

# Caminhos
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_RAW    = os.path.join(BASE_DIR, "../data/raw")
CACHE_DIR   = os.path.join(DATA_RAW, "fastf1_cache")
CKPT_FILE   = os.path.join(DATA_RAW, "fastf1_checkpoint_v2.json")
LOG_FILE    = os.path.join(DATA_RAW, "fastf1_extraction.log")
OUT_QUALI   = os.path.join(DATA_RAW, "fastf1_qualifying_2018_2025.csv")
OUT_LAPS    = os.path.join(DATA_RAW, "fastf1_laps_2018_2025.csv")
OUT_WEATHER = os.path.join(DATA_RAW, "fastf1_weather_2018_2025.csv")

YEARS              = list(range(2018, 2026))
SLEEP_BETWEEN_GPS  = 2
MAX_RETRIES        = 3
RETRY_BACKOFF_BASE = 10  # segundos; dobra a cada tentativa: 10s, 20s, 40s

# Setup
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_RAW,  exist_ok=True)

# redireciona pra Jolpica antes de habilitar o cache, senão ele guarda a url errada
_JOLPICA = "https://api.jolpi.ca/ergast/f1"
import fastf1.ergast.interface as _ff1_ergast
import fastf1.ergast.legacy   as _ff1_legacy
_ff1_ergast.BASE_URL = _JOLPICA
_ff1_legacy.base_url = _JOLPICA

fastf1.Cache.enable_cache(CACHE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
logging.getLogger("fastf1").setLevel(logging.WARNING)


# Checkpoint
def load_checkpoint() -> dict:
    # lê o arquivo de progresso pra saber de onde retomar
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(done: dict) -> None:
    with open(CKPT_FILE, "w") as f:
        json.dump(done, f, indent=2)


def ckpt_key(year: int, round_num: int, stype: str) -> str:
    # chave única por sessão: "2022_5_Q" por exemplo
    return f"{year}_{round_num}_{stype}"


# Schedule - carregado UMA vez por temporada
def get_schedule(year: int):
    # tenta pegar o calendário da temporada, com fallback pro Jolpica
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        log.info(f"  Schedule {year} via backend nativo ({len(schedule)} eventos).")
        return schedule
    except Exception as e:
        log.warning(f"  Backend nativo falhou para {year}: {e}")

    try:
        schedule = fastf1.get_event_schedule(
            year, include_testing=False, backend="ergast"
        )
        log.info(f"  Schedule {year} via Ergast/Jolpica ({len(schedule)} eventos).")
        return schedule
    except Exception as e:
        log.error(f"  Todos os backends falharam para {year}: {e}")
        return None


# Carregamento de sessao via Event object
def _is_no_data(exc: Exception) -> bool:
    # distingue "sem dados mesmo" de erro de rede
    msg = str(exc)
    return "session.laps vazio" in msg or "has not been loaded yet" in msg


def load_session_with_retry(event, stype: str, year: int, round_num: int):
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = event.get_session(stype)
            socket.setdefaulttimeout(90)  # evita hang indefinido em downloads
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            socket.setdefaulttimeout(None)
            if session.laps is None or len(session.laps) == 0:
                raise RuntimeError(
                    "session.laps vazio após load - dados não disponíveis no livetiming"
                )
            return session
        except Exception as e:
            last_exc = e
            # se não tem dado mesmo, não adianta ficar tentando
            if _is_no_data(e):
                raise RuntimeError("no_data") from e
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            log.warning(
                f"  Tentativa {attempt}/{MAX_RETRIES} falhou "
                f"({year} R{round_num} {stype}): {e}. Aguardando {wait}s..."
            )
            time.sleep(wait)
    raise RuntimeError(f"Falha após {MAX_RETRIES} tentativas") from last_exc


# Parsers
def parse_qualifying(session, year: int, round_num: int) -> pd.DataFrame:
    # tenta pegar posição e tempos Q1/Q2/Q3 dos results; se não tiver, usa as voltas
    res = session.results
    result_cols = ["Abbreviation", "Position", "Q1", "Q2", "Q3"]
    avail_res = [c for c in result_cols if c in res.columns]

    if not res.empty and len(avail_res) >= 2:
        df = res[avail_res].copy()
        df = df.rename(columns={"Abbreviation": "Driver", "Position": "position"})
        df["season"] = year
        df["round"]  = round_num
        return df

    # fallback: so tem o LapTime, sem os Q1/Q2/Q3 separados
    lap_cols  = ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    available = [c for c in lap_cols if c in session.laps.columns]
    df = session.laps[available].copy()
    df["season"] = year
    df["round"]  = round_num
    return df


def parse_race_laps(session, year: int, round_num: int) -> pd.DataFrame:
    # pega só as colunas que interessam pra modelagem
    cols = [
        "Driver", "LapNumber", "LapTime",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "Compound", "TyreLife", "Stint",
        "TrackStatus", "FreshTyre", "PitInTime", "PitOutTime",
    ]
    available = [c for c in cols if c in session.laps.columns]
    df = session.laps[available].copy()
    df["season"] = year
    df["round"]  = round_num
    return df


def parse_weather(session, year: int, round_num: int):
    # clima bruto - a agregacao mesmo fica na feature engineering
    wd = session.weather_data
    if wd is None or len(wd) == 0:
        return None
    cols      = ["AirTemp", "Humidity", "Rainfall", "TrackTemp", "WindSpeed"]
    available = [c for c in cols if c in wd.columns]
    df = wd[available].copy()
    df["season"] = year
    df["round"]  = round_num
    return df


# Loop principal
def run_extraction():
    done         = load_checkpoint()
    quali_dfs:   list[pd.DataFrame] = []
    laps_dfs:    list[pd.DataFrame] = []
    weather_dfs: list[pd.DataFrame] = []

    # carrega o que ja foi extraido antes pra nao perder
    if os.path.exists(OUT_QUALI):
        quali_dfs.append(pd.read_csv(OUT_QUALI))
        log.info(f"Qualifying parcial carregado: {len(quali_dfs[0])} registros.")
    if os.path.exists(OUT_LAPS):
        laps_dfs.append(pd.read_csv(OUT_LAPS))
        log.info(f"Voltas parcial carregado: {len(laps_dfs[0])} registros.")
    if os.path.exists(OUT_WEATHER):
        weather_dfs.append(pd.read_csv(OUT_WEATHER))
        log.info(f"Clima parcial carregado: {len(weather_dfs[0])} registros.")

    processed = skipped = errors = no_data = 0

    SESSIONS = [
        ("Q", parse_qualifying, quali_dfs, "Qualifying"),
        ("R", parse_race_laps,  laps_dfs,  "Race      "),
    ]

    for year in YEARS:
        log.info(f"\n{'='*55}")
        log.info(f"Temporada {year}")
        log.info(f"{'='*55}")

        schedule = get_schedule(year)
        if schedule is None:
            log.error(f"  Pulando {year} - schedule inacessível.")
            continue

        # itera pelo RoundNumber assim pra evitar bug do dropna() com colunas de string
        for round_num in schedule["RoundNumber"].dropna().astype(int):
            try:
                event = schedule.get_event_by_round(round_num)
            except Exception as e:
                log.error(f"  Não foi possível obter evento R{round_num}: {e}")
                continue
            gp_name = str(event["EventName"])

            for stype, parser, bucket, label in SESSIONS:
                key = ckpt_key(year, round_num, stype)

                # ja foi feito, pula
                if key in done:
                    tag = "[SKIP-ND]" if done[key] == "no_data" else "[SKIP]   "
                    log.info(f"  {tag} {label} {year} R{round_num} {gp_name}")
                    skipped += 1
                    continue

                processed += 1
                try:
                    session = load_session_with_retry(event, stype, year, round_num)
                    df = parser(session, year, round_num)
                    bucket.append(df)

                    # aproveita a sessão de corrida já carregada pra pegar o clima também
                    if stype == "R":
                        weather_df = parse_weather(session, year, round_num)
                        if weather_df is not None:
                            weather_dfs.append(weather_df)

                    done[key] = True
                    save_checkpoint(done)
                    log.info(
                        f"  ok {label} {year} R{round_num} {gp_name} - {len(df)} voltas"
                    )
                except RuntimeError as e:
                    if "no_data" in str(e):
                        no_data += 1
                        done[key] = "no_data"
                        save_checkpoint(done)
                        log.warning(
                            f"  aviso {label} {year} R{round_num} {gp_name}: "
                            f"sem dados no livetiming"
                        )
                    else:
                        errors += 1
                        log.error(
                            f"  erro {label} {year} R{round_num} {gp_name}: {e}"
                        )
                        log.debug(traceback.format_exc())
                except Exception as e:
                    errors += 1
                    log.error(f"  erro {label} {year} R{round_num} {gp_name}: {e}")
                    log.debug(traceback.format_exc())

                time.sleep(SLEEP_BETWEEN_GPS)

            # salva tudo apos cada GP pra nao perder progresso
            if quali_dfs:
                pd.concat(quali_dfs, ignore_index=True).to_csv(OUT_QUALI, index=False)
            if laps_dfs:
                pd.concat(laps_dfs, ignore_index=True).to_csv(OUT_LAPS, index=False)
            if weather_dfs:
                pd.concat(weather_dfs, ignore_index=True).to_csv(OUT_WEATHER, index=False)

    # Resumo final
    log.info(f"\n{'='*55}")
    log.info("Extração concluída!")
    log.info(f"  Processadas : {processed}")
    log.info(f"  Puladas     : {skipped}")
    log.info(f"  Sem dados   : {no_data}")
    log.info(f"  Erros       : {errors}")

    if quali_dfs:
        df_q = pd.concat(quali_dfs, ignore_index=True)
        df_q.to_csv(OUT_QUALI, index=False)
        log.info(f"  -> {OUT_QUALI}  ({len(df_q)} registros)")
    if laps_dfs:
        df_l = pd.concat(laps_dfs, ignore_index=True)
        df_l.to_csv(OUT_LAPS, index=False)
        log.info(f"  -> {OUT_LAPS}  ({len(df_l)} registros)")
    if weather_dfs:
        df_w = pd.concat(weather_dfs, ignore_index=True)
        df_w.to_csv(OUT_WEATHER, index=False)
        log.info(f"  -> {OUT_WEATHER}  ({len(df_w)} registros)")

    if errors > 0:
        log.warning(
            f"\n  {errors} sessão(ões) falharam por erro de rede. "
            f"Rode novamente para retentar."
        )


if __name__ == "__main__":
    log.info(
        f"Iniciando extração FastF1 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    log.info(f"Anos:   {YEARS[0]}-{YEARS[-1]}")
    log.info(f"Cache:  {CACHE_DIR}")
    log.info(f"Saída:  {OUT_QUALI}")
    log.info(f"        {OUT_LAPS}")
    log.info(f"        {OUT_WEATHER}")
    run_extraction()
