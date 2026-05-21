"""
Extração Ergast/Jolpica — pit stops 2018–2025.

Saída:
  data/raw/ergast_pitstop_2018_2025.csv  — pit stops por volta/piloto
    colunas: season, round, race_name, driver_id, stop, lap, duration

  data/raw/ergast_pitstop_2018_2025_parcial.csv — checkpoint incremental
    (sobrescrito a cada página; útil para retomar em caso de interrupção)

Fonte: https://api.jolpi.ca/ergast/f1/{ano}/{round}/pitstops.json
Rate limit: sleep(3) entre páginas para evitar 429.
"""

import os
import time

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_RAW  = os.path.join(BASE_DIR, "../data/raw")
OUT_FILE  = os.path.join(DATA_RAW, "ergast_pitstop_2018_2025.csv")
OUT_PARC  = os.path.join(DATA_RAW, "ergast_pitstop_2018_2025_parcial.csv")

os.makedirs(DATA_RAW, exist_ok=True)

BASE_URL   = "https://api.jolpi.ca/ergast/f1"
ANOS       = range(2018, 2026)
MAX_ROUNDS = 24  # número máximo de rounds por temporada

# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------
todos_pitstops = []

for ano in tqdm(ANOS, desc="Temporadas"):
    for round_num in range(1, MAX_ROUNDS + 1):
        limit  = 100
        offset = 0

        while True:
            url = (
                f"{BASE_URL}/{ano}/{round_num}/pitstops.json"
                f"?limit={limit}&offset={offset}"
            )

            try:
                response = requests.get(url, timeout=60)
            except requests.exceptions.RequestException as e:
                print(f"Erro de conexão — aguardando 90s... ({url})\n{e}")
                time.sleep(90)
                continue

            if response.status_code == 429:
                print("Erro 429 — aguardando 180s...")
                time.sleep(180)
                continue

            if response.status_code != 200:
                print(f"Erro {response.status_code}: {url}")
                time.sleep(30)
                continue

            try:
                data = response.json()
            except ValueError:
                print(f"Resposta não-JSON: {response.text[:300]}")
                time.sleep(60)
                continue

            races = data["MRData"]["RaceTable"]["Races"]
            total = int(data["MRData"]["total"])

            if not races:
                break

            for race in races:
                for pit in race["PitStops"]:
                    todos_pitstops.append({
                        "season":    str(ano),
                        "round":     str(round_num),
                        "race_name": race["raceName"],
                        "driver_id": pit["driverId"],
                        "stop":      int(pit["stop"]),
                        "lap":       int(pit["lap"]),
                        "duration":  pit["duration"],
                    })

            # checkpoint incremental
            pd.DataFrame(todos_pitstops).to_csv(OUT_PARC, index=False)

            offset += limit
            time.sleep(3)

            if offset >= total:
                break

df = pd.DataFrame(todos_pitstops)
df.to_csv(OUT_FILE, index=False)
print(f"\nSalvo em {OUT_FILE}")
print(f"Shape: {df.shape}")
