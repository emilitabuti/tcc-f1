from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

BASE_INTEGRADA = PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2025.csv"
BASE_DNF_EXCLUDED = PROCESSED_DIR / "historico_dnf_excluded_2018_2025.csv"
DATASET_PRE_FEATURES = PROCESSED_DIR / "dataset_pre_features_2018_2025.csv"
DATASET_FEATURES_FINAL = PROCESSED_DIR / "dataset_features_final_2018_2025.csv"
DATASET_MODELAGEM = PROCESSED_DIR / "dataset_modelagem_2018_2025.csv"

fastf1_cols = [
    "fastf1_laps_count",
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_max_tyre_life",
    "fastf1_pit_in_count",
    "fastf1_pit_out_count",
]

base = pd.read_csv(BASE_INTEGRADA)

sem_fastf1 = base[base[fastf1_cols].isna().all(axis=1)].copy()

print("\n=== REGISTROS SEM CORRESPONDÊNCIA FASTF1 NA BASE INTEGRADA ===\n")
print("Quantidade:", len(sem_fastf1))

colunas_exibir = [
    "season",
    "round",
    "race_name",
    "driver_id",
    "constructor_id",
    "grid_position",
    "finish_position",
    "status",
    "RaceID",
]

print(sem_fastf1[colunas_exibir].to_string(index=False))

raceids_sem_fastf1 = set(sem_fastf1["RaceID"])

arquivos_verificacao = {
    "historico_dnf_excluded_2018_2025.csv": BASE_DNF_EXCLUDED,
    "dataset_pre_features_2018_2025.csv": DATASET_PRE_FEATURES,
    "dataset_features_final_2018_2025.csv": DATASET_FEATURES_FINAL,
    "dataset_modelagem_2018_2025.csv": DATASET_MODELAGEM,
}

print("\n=== VERIFICAÇÃO NAS ETAPAS POSTERIORES ===\n")

for nome, caminho in arquivos_verificacao.items():
    df = pd.read_csv(caminho)

    if "RaceID" not in df.columns:
        print(f"{nome}: não possui coluna RaceID")
        continue

    qtd = df["RaceID"].isin(raceids_sem_fastf1).sum()
    print(f"{nome}: {qtd} registros encontrados")

print("\nRESULTADO ESPERADO:")
print("6 registros na base integrada inicial e 0 registros nas bases posteriores.")