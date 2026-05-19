"""
Extração Ergast/Jolpica — resultados de corrida 2025.

Saída:
  data/raw/ergast_2025_results.csv
    colunas: season, round, race_name, driver_id, constructor_id,
             grid_position, finish_position, status, points, laps

Fonte: https://api.jolpi.ca/ergast/f1/2025/results.json
Rodar periodicamente para manter o CSV atualizado ao longo da temporada.
"""

import os
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "../data/raw")
OUT_FILE = os.path.join(DATA_RAW, "ergast_2025_results.csv")

os.makedirs(DATA_RAW, exist_ok=True)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
ANO      = 2025

# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------
dados_2025 = []
limit      = 50
offset     = 0

while True:
    url = f"{BASE_URL}/{ANO}/results.json?limit={limit}&offset={offset}"

    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão — aguardando 60s... ({url})\n{e}")
        time.sleep(60)
        continue

    if response.status_code == 429:
        print("Erro 429 — aguardando 90s...")
        time.sleep(90)
        continue

    if response.status_code != 200:
        print(f"Erro {response.status_code}: {url}")
        time.sleep(10)
        continue

    try:
        data = response.json()
    except ValueError:
        print(f"Resposta não-JSON: {response.text[:300]}")
        time.sleep(10)
        continue

    races = data["MRData"]["RaceTable"]["Races"]
    total = int(data["MRData"]["total"])

    if not races:
        break

    for race in races:
        for result in race["Results"]:
            dados_2025.append({
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
    time.sleep(0.5)

    if offset >= total:
        break

df = pd.DataFrame(dados_2025)
df.to_csv(OUT_FILE, index=False)
print(f"\nSalvo em {OUT_FILE}")
print(f"Shape: {df.shape}")
print(df.head())
