"""
Verifica completude dos dados extraidos.

Analisa os CSVs em data/raw/ e reporta:
  - Shape e colunas de cada arquivo
  - Seasons presentes
  - Nulos por coluna
  - Corridas do Ergast sem FastF1 qualifying correspondente
  - Corridas do Ergast sem FastF1 laps correspondente
  - Corridas do Ergast sem pit stop correspondente

Salva o relatorio em data/raw/dados_ausentes.txt.
"""

import os
from io import StringIO

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "../data/raw")
SAIDA = os.path.join(DATA_RAW, "dados_ausentes.txt")

ARQUIVOS = {
    "Ergast 2018-2024":            os.path.join(DATA_RAW, "ergast_2018_2024.csv"),
    "Ergast pitstops 2018-2024":   os.path.join(DATA_RAW, "ergast_pitstop_2018_2024.csv"),
    "Ergast 2025":                 os.path.join(DATA_RAW, "ergast_2025_results.csv"),
    "OpenF1 starting grid 2025":   os.path.join(DATA_RAW, "openf1_starting_grid_2025.csv"),
    "FastF1 qualifying 2018-2024": os.path.join(DATA_RAW, "fastf1_qualifying_2018_2024.csv"),
    "FastF1 laps 2018-2024":       os.path.join(DATA_RAW, "fastf1_laps_2018_2024.csv"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
output = StringIO()


def log(msg=""):
    print(msg)
    output.write(msg + "\n")


def cross_check(label, base_df, other_df, base_name, other_name):
    """Imprime corridas presentes em base_df mas ausentes em other_df."""
    log(f"\n{'='*55}")
    log(f"  {label}")
    log(f"{'='*55}")

    base_rounds = set(
        base_df[["season", "round"]].drop_duplicates().apply(tuple, axis=1)
    )
    other_rounds = set(
        other_df[["season", "round"]].drop_duplicates().apply(tuple, axis=1)
    )

    ausentes = sorted(base_rounds - other_rounds)
    log(f"  Corridas em {base_name} sem correspondencia em {other_name}: {len(ausentes)}")
    for s, r in ausentes[:30]:
        log(f"    Season {s}, Round {r}")
    if len(ausentes) > 30:
        log(f"    ... e mais {len(ausentes) - 30}")
    return ausentes


# ---------------------------------------------------------------------------
# Analise por arquivo
# ---------------------------------------------------------------------------
dataframes = {}

for nome, caminho in ARQUIVOS.items():
    log(f"\n{'='*55}")
    log(f"  {nome}")
    log(f"{'='*55}")

    if not os.path.exists(caminho):
        log(f"  AVISO: Arquivo nao encontrado: {caminho}")
        continue

    df = pd.read_csv(caminho)
    dataframes[nome] = df

    log(f"  Shape  : {df.shape}")
    log(f"  Colunas: {df.columns.tolist()}")

    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique().tolist())
        log(f"  Seasons: {seasons}")

    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if len(nulos) > 0:
        log(f"\n  Nulos por coluna:")
        for col, n in nulos.items():
            pct = n / len(df) * 100
            log(f"    {col}: {n} ({pct:.1f}%)")
    else:
        log(f"\n  Sem nulos.")

# ---------------------------------------------------------------------------
# Cross-checks: Ergast vs FastF1
# ---------------------------------------------------------------------------
ergast  = dataframes.get("Ergast 2018-2024")
quali   = dataframes.get("FastF1 qualifying 2018-2024")
laps    = dataframes.get("FastF1 laps 2018-2024")
pitstop = dataframes.get("Ergast pitstops 2018-2024")

if ergast is not None and quali is not None:
    cross_check(
        "Ergast vs FastF1 Qualifying",
        ergast, quali,
        "Ergast", "FastF1 Qualifying",
    )

if ergast is not None and laps is not None:
    cross_check(
        "Ergast vs FastF1 Laps",
        ergast, laps,
        "Ergast", "FastF1 Laps",
    )

if ergast is not None and pitstop is not None:
    cross_check(
        "Ergast vs Pitstops",
        ergast, pitstop,
        "Ergast", "Pitstops",
    )

# ---------------------------------------------------------------------------
# Salvar relatorio
# ---------------------------------------------------------------------------
with open(SAIDA, "w", encoding="utf-8") as f:
    f.write(output.getvalue())

log(f"\nRelatorio salvo em: {SAIDA}")
