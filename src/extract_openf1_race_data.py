"""
Extração OpenF1 — dados de corrida 2025–2026.

Para cada sessão de corrida, extrai via API OpenF1:
  - stints     → compound, tyre_age_at_start, stint_number por piloto
  - weather    → temperatura, umidade, chuva, vento durante a corrida
  - race_control → eventos de safety car / bandeiras
  - session_result → posição final, DNF, número de voltas
  - meetings   → circuit_type por meeting

Saídas:
  data/raw/openf1_stints_2025_2026.csv
    colunas: season, meeting_key, session_key, driver_number,
             stint_number, compound, lap_start, tyre_age_at_start

  data/raw/openf1_weather_2025_2026.csv
    colunas: season, meeting_key, session_key, date,
             air_temperature, humidity, rainfall, track_temperature, wind_speed

  data/raw/openf1_race_control_2025_2026.csv
    colunas: season, meeting_key, session_key, date,
             category, flag, message, driver_number, lap_number

  data/raw/openf1_session_result_2025_2026.csv
    colunas: season, meeting_key, session_key, driver_number,
             position, dnf, number_of_laps

  data/raw/openf1_meetings_2025_2026.csv
    colunas: season, meeting_key, circuit_short_name, circuit_type,
             country_name, location, meeting_name

Fonte: https://api.openf1.org/v1
Rate limit: 3 req/s | 30 req/min → sleep(1.5) entre endpoints por sessão.
"""

import os
import time

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "../data/raw")

OUT_STINTS   = os.path.join(DATA_RAW, "openf1_stints_2025_2026.csv")
OUT_WEATHER  = os.path.join(DATA_RAW, "openf1_weather_2025_2026.csv")
OUT_RC       = os.path.join(DATA_RAW, "openf1_race_control_2025_2026.csv")
OUT_RESULT   = os.path.join(DATA_RAW, "openf1_session_result_2025_2026.csv")
OUT_MEETINGS = os.path.join(DATA_RAW, "openf1_meetings_2025_2026.csv")

os.makedirs(DATA_RAW, exist_ok=True)

BASE_URL = "https://api.openf1.org/v1"
YEARS    = [2025, 2026]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_json(url: str, params: dict, retries: int = 3) -> list | None:
    """GET com retry. Retorna lista de objetos ou None em caso de falha."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"Erro de conexão (tentativa {attempt}): {e}")
            time.sleep(60)
            continue

        if resp.status_code == 429:
            wait = 90 * attempt
            print(f"Erro 429 — aguardando {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code == 404:
            return []

        if resp.status_code != 200:
            print(f"Erro {resp.status_code}: {url} params={params}")
            time.sleep(5)
            continue

        try:
            return resp.json()
        except ValueError:
            print(f"Resposta não-JSON: {resp.text[:200]}")
            return None

    return None


def add_session_meta(records: list, season: int, meeting_key: int, session_key: int):
    """Adiciona season, meeting_key e session_key a cada registro."""
    for r in records:
        r["season"]      = season
        r["meeting_key"] = meeting_key
        r["session_key"] = session_key
    return records


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------
stints_all   = []
weather_all  = []
rc_all       = []
result_all   = []
meetings_all = []

for year in YEARS:
    print(f"\n{'='*50}")
    print(f"Ano {year}")
    print(f"{'='*50}")

    # 1. Meetings do ano
    meetings = get_json(f"{BASE_URL}/meetings", {"year": year}) or []
    for m in meetings:
        meetings_all.append({
            "season":             year,
            "meeting_key":        m.get("meeting_key"),
            "circuit_short_name": m.get("circuit_short_name"),
            "circuit_type":       m.get("circuit_type"),
            "country_name":       m.get("country_name"),
            "location":           m.get("location"),
            "meeting_name":       m.get("meeting_name"),
        })
    time.sleep(1.5)

    # 2. Sessões de corrida do ano
    sessions = get_json(f"{BASE_URL}/sessions", {"year": year, "session_name": "Race"}) or []
    print(f"  Corridas encontradas: {len(sessions)}")

    for sess in tqdm(sessions, desc=f"  {year} corridas"):
        session_key = sess["session_key"]
        meeting_key = sess["meeting_key"]

        # 3. Stints
        stints = get_json(f"{BASE_URL}/stints", {"session_key": session_key}) or []
        stints_parsed = []
        for s in stints:
            stints_parsed.append({
                "season":            year,
                "meeting_key":       meeting_key,
                "session_key":       session_key,
                "driver_number":     s.get("driver_number"),
                "stint_number":      s.get("stint_number"),
                "compound":          s.get("compound"),
                "lap_start":         s.get("lap_start"),
                "tyre_age_at_start": s.get("tyre_age_at_start"),
            })
        stints_all.extend(stints_parsed)
        time.sleep(1.5)

        # 4. Weather
        weather = get_json(f"{BASE_URL}/weather", {"session_key": session_key}) or []
        weather_parsed = []
        for w in weather:
            weather_parsed.append({
                "season":           year,
                "meeting_key":      meeting_key,
                "session_key":      session_key,
                "date":             w.get("date"),
                "air_temperature":  w.get("air_temperature"),
                "humidity":         w.get("humidity"),
                "rainfall":         w.get("rainfall"),
                "track_temperature": w.get("track_temperature"),
                "wind_speed":       w.get("wind_speed"),
            })
        weather_all.extend(weather_parsed)
        time.sleep(1.5)

        # 5. Race control
        rc = get_json(f"{BASE_URL}/race_control", {"session_key": session_key}) or []
        rc_parsed = []
        for r in rc:
            rc_parsed.append({
                "season":        year,
                "meeting_key":   meeting_key,
                "session_key":   session_key,
                "date":          r.get("date"),
                "category":      r.get("category"),
                "flag":          r.get("flag"),
                "message":       r.get("message"),
                "driver_number": r.get("driver_number"),
                "lap_number":    r.get("lap_number"),
            })
        rc_all.extend(rc_parsed)
        time.sleep(1.5)

        # 6. Session result
        results = get_json(f"{BASE_URL}/session_result", {"session_key": session_key}) or []
        result_parsed = []
        for r in results:
            result_parsed.append({
                "season":         year,
                "meeting_key":    meeting_key,
                "session_key":    session_key,
                "driver_number":  r.get("driver_number"),
                "position":       r.get("position"),
                "dnf":            r.get("dnf"),
                "number_of_laps": r.get("number_of_laps"),
            })
        result_all.extend(result_parsed)
        time.sleep(1.5)

# ---------------------------------------------------------------------------
# Salvar CSVs
# ---------------------------------------------------------------------------
for data, path in [
    (stints_all,   OUT_STINTS),
    (weather_all,  OUT_WEATHER),
    (rc_all,       OUT_RC),
    (result_all,   OUT_RESULT),
    (meetings_all, OUT_MEETINGS),
]:
    if data:
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        print(f"\nSalvo: {path}  ({len(df)} registros)")
    else:
        print(f"\nSem dados para: {path}")
