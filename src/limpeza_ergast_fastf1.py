from pathlib import Path
import pandas as pd
import numpy as np


# 01 - Limpeza do dataset histórico Ergast + FastF1
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Arquivos de entrada
ERGAST_FILE_2018_2024 = RAW_DIR / "ergast_2018_2024.csv"
ERGAST_FILE_2025 = RAW_DIR / "ergast_2025_results.csv"

FASTF1_LAPS_FILE = RAW_DIR / "fastf1_laps_2018_2025.csv"



# Arquivos de saída
OUTPUT_FILE_ATE_2024 = PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2024.csv"
OUTPUT_FILE_ATE_2025 = PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2025.csv"

REPORT_FILE = PROCESSED_DIR / "relatorio_01_limpeza_ergast_fastf1_2018_2025.txt"



# Mapeamento FastF1 Driver Code -> Ergast driver_id
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



# Funções auxiliares
def time_to_seconds(value):
    #Converte tempos do FastF1 para segundos.
    
    if pd.isna(value):
        return np.nan

    try:
        return pd.to_timedelta(value).total_seconds()
    except Exception:
        return np.nan


def first_valid(series):
    #Retorna o primeiro valor válido de uma coluna.
    #Usado para pegar o primeiro composto de pneu do piloto na corrida.
    
    values = series.dropna()

    if values.empty:
        return np.nan

    return values.iloc[0]


def most_frequent(series):
    #Retorna o valor mais frequente de uma coluna. Usado para pegar o composto de pneu predominante do piloto na corrida.
    
    values = series.dropna()

    if values.empty:
        return np.nan

    return values.mode().iloc[0]


def validate_columns(df, required_columns, dataframe_name):
    #Valida se as colunas obrigatórias existem no DataFrame.
    
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"As seguintes colunas obrigatórias estão ausentes em {dataframe_name}: "
            f"{missing_columns}"
        )


# 1. Carregar os arquivos
ergast_2018_2024 = pd.read_csv(ERGAST_FILE_2018_2024)
ergast_2025 = pd.read_csv(ERGAST_FILE_2025)
fastf1_laps = pd.read_csv(FASTF1_LAPS_FILE)

print("Arquivos carregados com sucesso.")
print(f"Ergast 2018-2024: {ergast_2018_2024.shape}")
print(f"Ergast 2025: {ergast_2025.shape}")
print(f"FastF1 laps: {fastf1_laps.shape}")



# 2. Padronizar nomes das colunas
ergast_2018_2024.columns = ergast_2018_2024.columns.str.strip().str.lower()
ergast_2025.columns = ergast_2025.columns.str.strip().str.lower()
fastf1_laps.columns = fastf1_laps.columns.str.strip()



# 3. Validar colunas obrigatórias
required_ergast_columns = [
    "season",
    "round",
    "driver_id",
    "grid_position",
    "finish_position",
]

required_fastf1_columns = [
    "season",
    "round",
    "Driver",
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Compound",
    "TyreLife",
    "Stint",
    "PitInTime",
    "PitOutTime",
]

validate_columns(
    ergast_2018_2024,
    required_ergast_columns,
    "ergast_2018_2024.csv"
)

validate_columns(
    ergast_2025,
    required_ergast_columns,
    "ergast_2025_results.csv"
)

validate_columns(
    fastf1_laps,
    required_fastf1_columns,
    "fastf1_laps_2018_2025.csv"
)


# 4. Concatenar Ergast 2018-2024 + Ergast 2025
ergast = pd.concat(
    [ergast_2018_2024, ergast_2025],
    ignore_index=True
)

print("\nErgast concatenado:")
print(ergast.shape)

print("\nTemporadas encontradas no Ergast concatenado:")
print(sorted(ergast["season"].dropna().unique()))


# 5. Converter campos numéricos importantes
ergast["season"] = pd.to_numeric(ergast["season"], errors="coerce")
ergast["round"] = pd.to_numeric(ergast["round"], errors="coerce")
ergast["grid_position"] = pd.to_numeric(ergast["grid_position"], errors="coerce")
ergast["finish_position"] = pd.to_numeric(ergast["finish_position"], errors="coerce")

fastf1_laps["season"] = pd.to_numeric(fastf1_laps["season"], errors="coerce")
fastf1_laps["round"] = pd.to_numeric(fastf1_laps["round"], errors="coerce")


# 6. Filtrar era híbrida disponível: 2018 em diante
linhas_ergast_inicial = len(ergast)
linhas_fastf1_inicial = len(fastf1_laps)

ergast = ergast[ergast["season"] >= 2018].copy()
fastf1_laps = fastf1_laps[fastf1_laps["season"] >= 2018].copy()

print("\nFiltro de temporada aplicado: season >= 2018")
print(f"Ergast após filtro: {ergast.shape}")
print(f"FastF1 após filtro: {fastf1_laps.shape}")


# 7. Remover registros com grid_position ou finish_position nulos
nulos_grid_finish = ergast[
    ergast["grid_position"].isna() | ergast["finish_position"].isna()
].shape[0]

ergast = ergast.dropna(
    subset=["grid_position", "finish_position"]
).copy()

print("\nRemoção de nulos:")
print(f"Registros removidos por grid_position/finish_position nulos: {nulos_grid_finish}")


# 8. Criar chave primária RaceID no Ergast
# RaceID = piloto + temporada + round
ergast["RaceID"] = (
    ergast["driver_id"].astype(str)
    + "_"
    + ergast["season"].astype(int).astype(str)
    + "_"
    + ergast["round"].astype(int).astype(str)
)

print("\nRaceID criada no Ergast.")


# 9. Remover duplicatas por RaceID
duplicatas_raceid = ergast.duplicated(subset=["RaceID"]).sum()

ergast = ergast.drop_duplicates(
    subset=["RaceID"],
    keep="first"
).copy()

print("\nRemoção de duplicatas:")
print(f"Duplicatas removidas por RaceID: {duplicatas_raceid}")



# 10. Preparar FastF1 para juntar com Ergast
fastf1_laps["driver_id"] = fastf1_laps["Driver"].map(DRIVER_CODE_TO_ID)

drivers_sem_mapeamento = (
    fastf1_laps[fastf1_laps["driver_id"].isna()]["Driver"]
    .dropna()
    .unique()
    .tolist()
)

if drivers_sem_mapeamento:
    print("\nAtenção: existem códigos de pilotos sem mapeamento:")
    print(drivers_sem_mapeamento)
else:
    print("\nTodos os códigos de pilotos do FastF1 foram mapeados.")

fastf1_laps = fastf1_laps.dropna(
    subset=["driver_id", "season", "round"]
).copy()

fastf1_laps["RaceID"] = (
    fastf1_laps["driver_id"].astype(str)
    + "_"
    + fastf1_laps["season"].astype(int).astype(str)
    + "_"
    + fastf1_laps["round"].astype(int).astype(str)
)

print("\nRaceID criada no FastF1.")



# 11. Converter tempos do FastF1 para segundos
time_columns = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
]

for col in time_columns:
    fastf1_laps[f"{col}_seconds"] = fastf1_laps[col].apply(time_to_seconds)

print("\nTempos do FastF1 convertidos para segundos.")



# 12. Agregar FastF1 por piloto/corrida
# 1 linha = 1 piloto em 1 corrida
fastf1_agg = (
    fastf1_laps
    .groupby("RaceID", as_index=False)
    .agg(
        fastf1_laps_count=("LapNumber", "count"),
        fastf1_avg_lap_time=("LapTime_seconds", "mean"),
        fastf1_best_lap_time=("LapTime_seconds", "min"),
        fastf1_avg_sector1=("Sector1Time_seconds", "mean"),
        fastf1_avg_sector2=("Sector2Time_seconds", "mean"),
        fastf1_avg_sector3=("Sector3Time_seconds", "mean"),
        fastf1_first_compound=("Compound", first_valid),
        fastf1_main_compound=("Compound", most_frequent),
        fastf1_max_tyre_life=("TyreLife", "max"),
        fastf1_stints_count=("Stint", "nunique"),
        fastf1_pit_in_count=("PitInTime", lambda x: x.notna().sum()),
        fastf1_pit_out_count=("PitOutTime", lambda x: x.notna().sum()),
    )
)

print("\nFastF1 agregado por RaceID:")
print(fastf1_agg.shape)



# 13. Juntar Ergast + FastF1
historico_limpo = ergast.merge(
    fastf1_agg,
    on="RaceID",
    how="left"
)

sem_fastf1 = historico_limpo["fastf1_laps_count"].isna().sum()

print("\nMerge Ergast + FastF1 concluído.")
print(f"Linhas finais após merge: {historico_limpo.shape[0]}")
print(f"Registros Ergast sem correspondência FastF1: {sem_fastf1}")



# 14. Criar versões finais: até 2024 e até 2025
historico_limpo_ate_2024 = historico_limpo[
    historico_limpo["season"] <= 2024
].copy()

historico_limpo_ate_2025 = historico_limpo[
    historico_limpo["season"] <= 2025
].copy()

print("\nConferência das temporadas nos arquivos finais:")

print("\nArquivo até 2024:")
print(historico_limpo_ate_2024["season"].value_counts().sort_index())

print("\nArquivo até 2025:")
print(historico_limpo_ate_2025["season"].value_counts().sort_index())



# 15. Salvar arquivos finais
historico_limpo_ate_2024.to_csv(
    OUTPUT_FILE_ATE_2024,
    index=False,
    encoding="utf-8-sig"
)

historico_limpo_ate_2025.to_csv(
    OUTPUT_FILE_ATE_2025,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo histórico até 2024 salvo em:")
print(OUTPUT_FILE_ATE_2024)

print("\nArquivo histórico até 2025 salvo em:")
print(OUTPUT_FILE_ATE_2025)



# 16. Salvar relatório
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 01 LIMPEZA ERGAST + FASTF1\n")
    f.write("=" * 60 + "\n\n")

    f.write("ARQUIVOS UTILIZADOS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Ergast 2018-2024: {ERGAST_FILE_2018_2024}\n")
    f.write(f"Ergast 2025: {ERGAST_FILE_2025}\n")
    f.write(f"FastF1 laps 2018-2025: {FASTF1_LAPS_FILE}\n\n")

    f.write("DIMENSÕES INICIAIS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Linhas iniciais Ergast concatenado: {linhas_ergast_inicial}\n")
    f.write(f"Linhas iniciais FastF1: {linhas_fastf1_inicial}\n\n")

    f.write("LIMPEZA ERGAST\n")
    f.write("-" * 60 + "\n")
    f.write(f"Linhas Ergast após filtro season >= 2018: {len(ergast)}\n")
    f.write(f"Registros removidos por grid_position/finish_position nulos: {nulos_grid_finish}\n")
    f.write(f"Duplicatas removidas por RaceID: {duplicatas_raceid}\n\n")

    f.write("AGREGAÇÃO FASTF1\n")
    f.write("-" * 60 + "\n")
    f.write(f"Linhas FastF1 agregadas por RaceID: {len(fastf1_agg)}\n")

    if drivers_sem_mapeamento:
        f.write("Códigos de pilotos sem mapeamento FastF1 -> Ergast:\n")
        for driver in drivers_sem_mapeamento:
            f.write(f"- {driver}\n")
    else:
        f.write("Todos os códigos de pilotos do FastF1 foram mapeados.\n")

    f.write("\nMERGE FINAL\n")
    f.write("-" * 60 + "\n")
    f.write(f"Linhas finais após merge geral: {len(historico_limpo)}\n")
    f.write(f"Linhas finais 2018-2024: {len(historico_limpo_ate_2024)}\n")
    f.write(f"Linhas finais 2018-2025: {len(historico_limpo_ate_2025)}\n")
    f.write(f"Registros Ergast sem correspondência FastF1: {sem_fastf1}\n\n")

    f.write("Temporadas no arquivo 2018-2024:\n")
    f.write(str(historico_limpo_ate_2024["season"].value_counts().sort_index()))
    f.write("\n\n")

    f.write("Temporadas no arquivo 2018-2025:\n")
    f.write(str(historico_limpo_ate_2025["season"].value_counts().sort_index()))
    f.write("\n")

print("\nRelatório salvo em:")
print(REPORT_FILE)

print("\nEtapa 01 finalizada com sucesso.")