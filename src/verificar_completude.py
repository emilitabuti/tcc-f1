"""
Verifica completude dos dados extraídos.

Analisa os CSVs em data/raw/ e reporta:
  - Shape e colunas de cada arquivo
  - Seasons presentes
  - Nulos por coluna
  - Corridas do Ergast sem pit stop correspondente

Saída impressa no terminal (sem arquivos gerados).
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "../data/raw")

ARQUIVOS = {
    "Ergast 2018–2024":            os.path.join(DATA_RAW, "ergast_2018_2024.csv"),
    "Ergast pitstops 2018–2024":   os.path.join(DATA_RAW, "ergast_pitstop_2018_2024.csv"),
    "Ergast 2025":                 os.path.join(DATA_RAW, "ergast_2025_results.csv"),
    "OpenF1 starting grid 2025":   os.path.join(DATA_RAW, "openf1_starting_grid_2025.csv"),
    "FastF1 qualifying 2018–2024": os.path.join(DATA_RAW, "fastf1_qualifying_2018_2024.csv"),
        "FastF1 laps 2018–2024":       os.path.join(DATA_RAW, "fastf1_laps_2018_2024.csv"),
}

# ---------------------------------------------------------------------------
# Análise por arquivo
# ---------------------------------------------------------------------------
dataframes = {}

for nome, caminho in ARQUIVOS.items():
    print(f"\n{'='*55}")
    print(f"  {nome}")
    print(f"{'='*55}")

    if not os.path.exists(caminho):
        print(f"  ⚠️  Arquivo não encontrado: {caminho}")
        continue

    df = pd.read_csv(caminho)
    dataframes[nome] = df

    print(f"  Shape  : {df.shape}")
    print(f"  Colunas: {df.columns.tolist()}")

    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique().tolist())
        print(f"  Seasons: {seasons}")

    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if len(nulos) > 0:
        print(f"\n  Nulos por coluna:")
        for col, n in nulos.items():
            print(f"    {col}: {n}")
    else:
        print(f"\n  Sem nulos.")

# ---------------------------------------------------------------------------
# Corridas do Ergast sem pit stop
# ---------------------------------------------------------------------------
ergast  = dataframes.get("Ergast 2018–2024")
pitstop = dataframes.get("Ergast pitstops 2018–2024")

if ergast is not None and pitstop is not None:
    print(f"\n{'='*55}")
    print("  Corridas sem pit stop")
    print(f"{'='*55}")

    e_rounds = set(
        ergast[["season", "round"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
    )
    p_rounds = set(
        pitstop[["season", "round"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
    )

    ausentes = e_rounds - p_rounds
    print(f"  Total: {len(ausentes)}")
    for s, r in sorted(ausentes)[:20]:
        print(f"    Season {s}, Round {r}")
    if len(ausentes) > 20:
        print(f"    ... e mais {len(ausentes) - 20}")
