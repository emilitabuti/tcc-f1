import pandas as pd

DRIVER_CODE_TO_ID = {
    "AIT": "aitken",
    "ALB": "albon",
    "ALO": "alonso",
    "ANT": "antonelli",
    "BEA": "bearman",
    "BOR": "bortoleto",
    "BOT": "bottas",
    "COL": "colapinto",
    "DEV": "de_vries",
    "DOO": "doohan",
    "ERI": "ericsson",
    "FIT": "pietro_fittipaldi",
    "GAS": "gasly",
    "GIO": "giovinazzi",
    "GRO": "grosjean",
    "HAD": "hadjar",
    "HAM": "hamilton",
    "HAR": "brendon_hartley",
    "HUL": "hulkenberg",
    "KUB": "kubica",
    "KVY": "kvyat",
    "LAT": "latifi",
    "LAW": "lawson",
    "LEC": "leclerc",
    "MAG": "kevin_magnussen",
    "MAZ": "mazepin",
    "MSC": "mick_schumacher",
    "NOR": "norris",
    "OCO": "ocon",
    "PER": "perez",
    "PIA": "piastri",
    "RAI": "raikkonen",
    "RIC": "ricciardo",
    "RUS": "russell",
    "SAI": "sainz",
    "SAR": "sargeant",
    "SIR": "sirotkin",
    "STR": "stroll",
    "TSU": "tsunoda",
    "VAN": "vandoorne",
    "VER": "max_verstappen",
    "VET": "vettel",
    "ZHO": "zhou",
}

fastf1 = pd.read_csv("data/raw/fastf1_laps_2018_2025.csv")
ergast_2018_2024 = pd.read_csv("data/raw/ergast_2018_2024.csv")
ergast_2025 = pd.read_csv("data/raw/ergast_2025_results.csv")

ergast = pd.concat([ergast_2018_2024, ergast_2025], ignore_index=True)

codigos_fastf1 = set(fastf1["Driver"].dropna().unique())
codigos_mapeados = set(DRIVER_CODE_TO_ID.keys())

ids_ergast = set(ergast["driver_id"].dropna().unique())
ids_mapeados = set(DRIVER_CODE_TO_ID.values())

codigos_sem_mapeamento = sorted(codigos_fastf1 - codigos_mapeados)
ids_mapeados_sem_ergast = sorted(ids_mapeados - ids_ergast)
ids_ergast_sem_mapeamento = sorted(ids_ergast - ids_mapeados)

print("Quantidade de entradas no DRIVER_CODE_TO_ID:", len(DRIVER_CODE_TO_ID))
print("Quantidade de códigos FastF1:", len(codigos_fastf1))
print("Códigos FastF1 sem mapeamento:", codigos_sem_mapeamento)
print("driver_id mapeado sem existir no Ergast:", ids_mapeados_sem_ergast)
print("driver_id Ergast sem mapeamento:", ids_ergast_sem_mapeamento)