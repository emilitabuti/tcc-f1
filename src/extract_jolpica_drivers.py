# Extração Jolpica - pilotos que competiram em 2018-2025.
# Saida: data/raw/jolpica_drivers.csv
# Fonte: https://api.jolpi.ca/ergast/f1/drivers.json

import os
import time

import pandas as pd
import requests

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "../data/raw")
OUT_FILE = os.path.join(DATA_RAW, "jolpica_drivers.csv")

os.makedirs(DATA_RAW, exist_ok=True)

BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Extração
dados_pilotos = []
limit  = 100
offset = 0

while True:
    url = f"{BASE_URL}/drivers.json?limit={limit}&offset={offset}"

    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão - aguardando 60s... ({url})\n{e}")
        time.sleep(60)
        continue

    if response.status_code == 429:
        print("Erro 429 - aguardando 90s...")
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

    drivers = data["MRData"]["DriverTable"]["Drivers"]
    total   = int(data["MRData"]["total"])

    # monta uma linha por piloto com os campos que vão ser usados depois
    for d in drivers:
        dados_pilotos.append({
            "driver_id":    d["driverId"],
            "given_name":   d["givenName"],
            "family_name":  d["familyName"],
            "date_of_birth": d.get("dateOfBirth", ""),
            "nationality":  d.get("nationality", ""),
        })

    offset += limit
    time.sleep(0.3)

    if offset >= total:
        break

df = pd.DataFrame(dados_pilotos)
df.to_csv(OUT_FILE, index=False)
print(f"\nSalvo em {OUT_FILE}")
print(f"Shape: {df.shape}")
print(df.head())
