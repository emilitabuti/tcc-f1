"""
Extração OpenF1 — grid de largada 2025.

Para cada corrida de 2025, busca a posição de grid via /starting_grid
usando a session_key da sessão de Qualifying do mesmo meeting.

Saída:
  data/raw/openf1_starting_grid_2025.csv
    colunas: season, meeting_key, race_session_key, qualifying_session_key,
             circuit_short_name, date_start, driver_number,
             grid_position, qualifying_lap_duration

Mapeamento:
  grid_position = campo `position` do endpoint /starting_grid
                  (session_key da Qualifying, não da corrida)
  Join para cruzar com corrida: via meeting_key

Fonte: https://api.openf1.org/v1
Rate limit: 3 req/s | 30 req/min → sleep(1) entre corridas.
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
OUT_FILE = os.path.join(DATA_RAW, "openf1_starting_grid_2025.csv")

os.makedirs(DATA_RAW, exist_ok=True)

BASE_URL = "https://api.openf1.org/v1"

# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------
dados_grid = []

# 1. Buscar todas as sessões de 2025
sessions = requests.get(
    f"{BASE_URL}/sessions",
    params={"year": 2025},
    timeout=30,
).json()

df_sessions = pd.DataFrame(sessions)

# 2. Filtrar apenas sessões de corrida
corridas = df_sessions[df_sessions["session_name"] == "Race"]
print(f"Corridas encontradas em 2025: {len(corridas)}")

# 3. Percorrer todas as corridas
for _, race in tqdm(corridas.iterrows(), total=len(corridas)):
    meeting_key       = race["meeting_key"]
    race_session_key  = race["session_key"]
    circuit_short_name = race.get("circuit_short_name")
    date_start        = race.get("date_start")

    # 4. Buscar todas as sessões do mesmo meeting
    try:
        resp_sessions = requests.get(
            f"{BASE_URL}/sessions",
            params={"meeting_key": meeting_key},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar sessões do meeting {meeting_key}: {e}")
        time.sleep(60)
        continue

    if resp_sessions.status_code == 429:
        print("Erro 429 em /sessions — aguardando 90s...")
        time.sleep(90)
        continue

    if resp_sessions.status_code != 200:
        print(f"Erro {resp_sessions.status_code} em /sessions para meeting_key={meeting_key}")
        continue

    df_meeting = pd.DataFrame(resp_sessions.json())
    if df_meeting.empty:
        continue

    # 5. Buscar a sessão de Qualifying do mesmo meeting
    qualifying = df_meeting[df_meeting["session_name"] == "Qualifying"]
    if qualifying.empty:
        print(f"Sem Qualifying para meeting_key={meeting_key}")
        continue

    qualifying_session_key = qualifying.iloc[0]["session_key"]

    # 6. Buscar starting grid usando a session_key da Qualifying
    try:
        resp_grid = requests.get(
            f"{BASE_URL}/starting_grid",
            params={"session_key": qualifying_session_key},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar /starting_grid para quali {qualifying_session_key}: {e}")
        time.sleep(60)
        continue

    if resp_grid.status_code == 429:
        print("Erro 429 em /starting_grid — aguardando 90s...")
        time.sleep(90)
        continue

    if resp_grid.status_code == 404:
        print(f"Starting grid não disponível para qualifying_session_key={qualifying_session_key}")
        continue

    if resp_grid.status_code != 200:
        print(f"Erro {resp_grid.status_code} em /starting_grid para {qualifying_session_key}")
        continue

    try:
        grid = resp_grid.json()
    except ValueError:
        print(f"Resposta não-JSON em /starting_grid: {resp_grid.text[:300]}")
        continue

    if not grid:
        print(f"Grid vazio para qualifying_session_key={qualifying_session_key}")
        continue

    for item in grid:
        dados_grid.append({
            "season":                  2025,
            "meeting_key":             meeting_key,
            "race_session_key":        race_session_key,
            "qualifying_session_key":  qualifying_session_key,
            "circuit_short_name":      circuit_short_name,
            "date_start":              date_start,
            "driver_number":           item.get("driver_number"),
            "grid_position":           item.get("position"),
            "qualifying_lap_duration": item.get("lap_duration"),
        })

    time.sleep(1)

# 7. Salvar CSV
df_grid = pd.DataFrame(dados_grid)
df_grid.to_csv(OUT_FILE, index=False)
print(f"\nSalvo em {OUT_FILE}")
print(f"Shape: {df_grid.shape}")
print(df_grid.head())
