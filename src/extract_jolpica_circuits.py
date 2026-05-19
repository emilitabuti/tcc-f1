"""
Extração Jolpica — circuitos do calendário F1 2018–2025.

Saída:
  data/raw/jolpica_circuits.csv
    colunas: circuit_id, circuit_name, lat, long, country

Fonte: https://api.jolpi.ca/ergast/f1/circuits.json
Busca todos os circuitos de uma vez (sem paginação necessária para o conjunto completo).
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
OUT_FILE = os.path.join(DATA_RAW, "jolpica_circuits.csv")

os.makedirs(DATA_RAW, exist_ok=True)

BASE_URL = "https://api.jolpi.ca/ergast/f1"

# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------
dados_circuitos = []
limit  = 100
offset = 0

while True:
    url = f"{BASE_URL}/circuits.json?limit={limit}&offset={offset}"

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
        break

    try:
        data = response.json()
    except ValueError:
        print(f"Resposta não-JSON: {response.text[:300]}")
        break

    circuits = data["MRData"]["CircuitTable"]["Circuits"]
    total    = int(data["MRData"]["total"])

    for c in circuits:
        dados_circuitos.append({
            "circuit_id":   c["circuitId"],
            "circuit_name": c["circuitName"],
            "lat":          c["Location"]["lat"],
            "long":         c["Location"]["long"],
            "country":      c["Location"]["country"],
        })

    offset += limit
    time.sleep(0.3)

    if offset >= total:
        break

df = pd.DataFrame(dados_circuitos)
df.to_csv(OUT_FILE, index=False)
print(f"\nSalvo em {OUT_FILE}")
print(f"Shape: {df.shape}")
print(df.head())
