"""
Extração Ergast/Jolpica — resultados de corrida 2018–2025.

Saída:
  data/raw/ergast_2018_2024.csv  — temporadas 2018–2024
    colunas: season, round, race_name, driver_id, constructor_id,
             grid_position, finish_position, status, points, laps

Fonte: https://api.jolpi.ca/ergast/f1  (mirror não-oficial do Ergast)
Rate limit: ~4 req/s. sleep(0.3) mantém ~3 req/s.
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
OUT_FILE = os.path.join(DATA_RAW, "ergast_2018_2024.csv")

os.makedirs(DATA_RAW, exist_ok=True)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
ANOS     = range(2018, 2025)

# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------
dados_corridas = []

for ano in tqdm(ANOS, desc="Temporadas"):
    limit  = 50
    offset = 0

    while True:
        url = f"{BASE_URL}/{ano}/results.json?limit={limit}&offset={offset}"
        response = requests.get(url, timeout=30)

        if response.status_code == 429:
            print(f"Erro 429 — aguardando 90s... ({url})")
            time.sleep(90)
            continue

        if response.status_code != 200:
            print(f"Erro {response.status_code}: {url}")
            time.sleep(5)
            continue

        try:
            data = response.json()
        except ValueError:
            print(f"Resposta não-JSON: {response.text[:300]}")
            time.sleep(5)
            continue

        races = data["MRData"]["RaceTable"]["Races"]
        total = int(data["MRData"]["total"])

        if not races:
            break

        for race in races:
            for result in race["Results"]:
                dados_corridas.append({
                    "season":          race["season"],
                    "round":           race["round"],
                    "race_name":       race["raceName"],
                    "driver_id":       result["Driver"]["driverId"],
                    "constructor_id":  result["Constructor"]["constructorId"],
                    "grid_position":   result["grid"],
                    "finish_position": result["position"],
                    "status":          result["status"],
                    "points":          result["points"],
                    "laps":            result.get("laps", ""),
                })

        offset += limit
        time.sleep(0.3)

        if offset >= total:
            break

df = pd.DataFrame(dados_corridas)
df.to_csv(OUT_FILE, index=False)
print(f"\nSalvo em {OUT_FILE}")
print(f"Shape: {df.shape}")
print(f"Seasons: {sorted(df['season'].unique())}")
