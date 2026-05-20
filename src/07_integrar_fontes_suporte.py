from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


# 07 - Integração de fontes de suporte ao dataset principal
# Adiciona: circuit features, weather_impact_factor, avg_pit_stops_circuit,
#           safety_car_flag (2025), track_complexity e corrige grid_position=0.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models" / "preprocessing"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Arquivos de entrada ────────────────────────────────────────────────────────
INPUT_2018_2024 = PROCESSED_DIR / "historico_outliers_tratados_2018_2024.csv"
INPUT_2018_2025 = PROCESSED_DIR / "historico_outliers_tratados_2018_2025.csv"

CIRCUITOS_MANUAL = RAW_DIR / "circuitos_manual.csv"
WEATHER_FILE = RAW_DIR / "fastf1_weather_2018_2025.csv"
PITSTOP_FILE = RAW_DIR / "ergast_pitstop_2018_2025.csv"
RACE_CONTROL_FILE = RAW_DIR / "openf1_race_control_2025_2026.csv"
MEETINGS_FILE = RAW_DIR / "openf1_meetings_2025_2026.csv"
QUALIFYING_FILE = RAW_DIR / "fastf1_qualifying_2018_2025.csv"
FASTF1_LAPS_FILE = RAW_DIR / "fastf1_laps_2018_2025.csv"

# ── Arquivos de saída ──────────────────────────────────────────────────────────
OUTPUT_2018_2024 = PROCESSED_DIR / "dataset_pre_features_2018_2024.csv"
OUTPUT_2018_2025 = PROCESSED_DIR / "dataset_pre_features_2018_2025.csv"

SCALER_WEATHER = MODELS_DIR / "scaler_weather_impact.joblib"
SCALER_PITSTOPS = MODELS_DIR / "scaler_avg_pitstops.joblib"
SCALER_TRACK = MODELS_DIR / "scaler_track_complexity.joblib"
SCALER_GRID = MODELS_DIR / "scaler_grid_position_fixed.joblib"

REPORT_FILE = PROCESSED_DIR / "relatorio_07_integracao_fontes.txt"

# ── Mapeamento código FastF1 → driver_id Ergast ───────────────────────────────
# Mesma tabela de limpeza_ergast_fastf1.py — fonte única de verdade para os códigos.
DRIVER_CODE_TO_ID = {
    "AIT": "aitken",        "ALB": "albon",           "ALO": "alonso",
    "ANT": "antonelli",     "BEA": "bearman",         "BOR": "bortoleto",
    "BOT": "bottas",        "COL": "colapinto",       "DEV": "de_vries",
    "DOO": "doohan",        "ERI": "ericsson",        "FIT": "pietro_fittipaldi",
    "GAS": "gasly",         "GIO": "giovinazzi",      "GRO": "grosjean",
    "HAD": "hadjar",        "HAM": "hamilton",        "HAR": "brendon_hartley",
    "HUL": "hulkenberg",    "KUB": "kubica",          "KVY": "kvyat",
    "LAT": "latifi",        "LAW": "lawson",          "LEC": "leclerc",
    "MAG": "kevin_magnussen","MAZ": "mazepin",         "MSC": "mick_schumacher",
    "NOR": "norris",        "OCO": "ocon",            "PER": "perez",
    "PIA": "piastri",       "RAI": "raikkonen",       "RIC": "ricciardo",
    "RUS": "russell",       "SAI": "sainz",           "SAR": "sargeant",
    "SIR": "sirotkin",      "STR": "stroll",          "TSU": "tsunoda",
    "VAN": "vandoorne",     "VER": "max_verstappen",  "VET": "vettel",
    "ZHO": "zhou",
}

# ── Mapeamento race_name → circuit_id ─────────────────────────────────────────
RACE_NAME_TO_CIRCUIT_ID = {
    "Australian Grand Prix": "albert_park",
    "Bahrain Grand Prix": "bahrain",
    "Chinese Grand Prix": "shanghai",
    "Azerbaijan Grand Prix": "baku",
    "Spanish Grand Prix": "catalunya",
    "Monaco Grand Prix": "monaco",
    "Canadian Grand Prix": "villeneuve",
    "French Grand Prix": "ricard",
    "Austrian Grand Prix": "red_bull_ring",
    "British Grand Prix": "silverstone",
    "German Grand Prix": "hockenheim",
    "Hungarian Grand Prix": "hungaroring",
    "Belgian Grand Prix": "spa",
    "Italian Grand Prix": "monza",
    "Singapore Grand Prix": "marina_bay",
    "Russian Grand Prix": "sochi",
    "Japanese Grand Prix": "suzuka",
    "United States Grand Prix": "americas",
    "Mexican Grand Prix": "rodriguez",
    "Mexico City Grand Prix": "rodriguez",
    "Brazilian Grand Prix": "interlagos",
    "São Paulo Grand Prix": "interlagos",
    "Abu Dhabi Grand Prix": "yas_marina",
    "Styrian Grand Prix": "red_bull_ring",
    "70th Anniversary Grand Prix": "silverstone",
    "Tuscan Grand Prix": "mugello",
    "Eifel Grand Prix": "nurburgring",
    "Turkish Grand Prix": "istanbul",
    "Sakhir Grand Prix": "bahrain_outer",
    "Dutch Grand Prix": "zandvoort",
    "Emilia Romagna Grand Prix": "imola",
    "Portuguese Grand Prix": "portimao",
    "Saudi Arabian Grand Prix": "jeddah",
    "Miami Grand Prix": "miami",
    "Qatar Grand Prix": "losail",
    "Las Vegas Grand Prix": "las_vegas",
}


# ─────────────────────────────────────────────────────────────────────────────
# LACUNA 2 — qualifying_position não integrado ao dataset
#
# Adiciona duas colunas:
#   qualifying_position : posição no qualifying (pode diferir de grid_position
#                         quando há penalidades de grid aplicadas após o quali)
#   grid_penalty        : grid_position - qualifying_position
#                         > 0 → penalidade (caiu posições)
#                         < 0 → promoção por penalidade de outro piloto
#                         = 0 → sem penalidade (ou pit lane start)
#
# Cobertura esperada: ~93% (restante → qualifying_position = grid_position,
# grid_penalty = 0, pit lane starts ou corridas sem dados de FastF1)
# ─────────────────────────────────────────────────────────────────────────────
def preparar_qualifying(qual_df):
    df = qual_df.copy()
    df["driver_id"] = df["Driver"].map(DRIVER_CODE_TO_ID)
    df = df.rename(columns={"position": "qualifying_position"})
    df["qualifying_position"] = pd.to_numeric(df["qualifying_position"], errors="coerce")
    return df[["season", "round", "driver_id", "qualifying_position"]]


def integrar_qualifying(df, qual_df):
    df = df.copy()
    qual_preparado = preparar_qualifying(qual_df)

    df = df.merge(qual_preparado, on=["season", "round", "driver_id"], how="left")

    # Pit lane starts: qualifying_position não é comparável a grid_position
    # (grid_position já corrigido para 21). Trata separadamente.
    mask_pit = df["grid_position_zero_flag"] == 1
    df.loc[mask_pit, "qualifying_position"] = df.loc[mask_pit, "grid_position"]

    # Restante sem dados: usa grid_position como proxy (sem penalidade conhecida)
    sem_qualifying = df["qualifying_position"].isnull()
    df.loc[sem_qualifying, "qualifying_position"] = df.loc[sem_qualifying, "grid_position"]

    df["qualifying_position"] = df["qualifying_position"].astype(int)

    # grid_penalty: positivo = penalizado (caiu), negativo = promovido
    df["grid_penalty"] = (df["grid_position"] - df["qualifying_position"]).astype(int)

    n_sem = sem_qualifying.sum()
    n_penalizados = (df["grid_penalty"] > 0).sum()
    taxa_cobertura = 1 - (n_sem / len(df))
    print(f"  qualifying_position: cobertura {taxa_cobertura:.1%} ({n_sem} sem dados → proxy grid_pos)")
    print(f"  grid_penalty > 0 (penalidades): {n_penalizados} registros")
    print(f"  grid_penalty < 0 (promoções):   {(df['grid_penalty'] < 0).sum()} registros")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# LACUNA 11 — grid_position = 0 indica pit lane start (semântica errada).
# Substitui por 21, além da última posição real, para que o MinMaxScaler
# não use 0 como mínimo absoluto.
# ─────────────────────────────────────────────────────────────────────────────
def corrigir_grid_position(df):
    df = df.copy()
    mask = df["grid_position"] == 0
    n = mask.sum()
    if n > 0:
        df.loc[mask, "grid_position"] = 21
        print(f"  grid_position: {n} pit-lane starts corrigidos (0 → 21).")
    return df, n


# ─────────────────────────────────────────────────────────────────────────────
# LACUNA 5 — Integrar circuitos_manual.csv
# Adiciona: altitude_m, corners, length_km, circuit_type, track_complexity
# ─────────────────────────────────────────────────────────────────────────────
def construir_track_complexity(circuitos_df):
    df = circuitos_df.copy()

    # Normaliza cada componente para [0,1] usando os 32/33 circuitos como universo
    for col in ["corners", "length_km", "altitude_m"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)

    # circuit_type já é 0/1 (permanente/urbano)
    df["track_complexity"] = (
        0.40 * df["corners_norm"]
        + 0.30 * df["length_km_norm"]
        + 0.20 * df["altitude_m_norm"]
        + 0.10 * df["circuit_type"]
    )

    return df[["circuit_id", "altitude_m", "corners", "length_km",
               "circuit_type", "track_complexity"]]


def integrar_circuitos(df, circuitos_df):
    df = df.copy()
    circ_features = construir_track_complexity(circuitos_df)
    df = df.merge(circ_features, on="circuit_id", how="left")
    n_sem_circuito = df["track_complexity"].isnull().sum()
    if n_sem_circuito > 0:
        print(f"  AVISO: {n_sem_circuito} linhas sem circuit_id mapeado em circuitos_manual.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LACUNA 4 — Integrar weather e calcular weather_impact_factor
# Agrega por (season, round): mean AirTemp, mean Humidity, max Rainfall
# Formula: (norm_hum + 2*rain_binary + (1 - norm_air)) / 4  ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def agregar_weather(weather_df):
    agg = weather_df.groupby(["season", "round"], as_index=False).agg(
        AirTemp=("AirTemp", "mean"),
        Humidity=("Humidity", "mean"),
        Rainfall=("Rainfall", "max"),
    )
    agg["Rainfall"] = agg["Rainfall"].astype(int)
    return agg


def calcular_weather_impact(weather_agg, treino_2024):
    air_min = treino_2024["AirTemp"].min()
    air_max = treino_2024["AirTemp"].max()
    hum_min = treino_2024["Humidity"].min()
    hum_max = treino_2024["Humidity"].max()

    df = weather_agg.copy()
    df["air_norm"] = (df["AirTemp"] - air_min) / (air_max - air_min + 1e-9)
    df["hum_norm"] = (df["Humidity"] - hum_min) / (hum_max - hum_min + 1e-9)

    df["weather_impact_factor"] = (
        df["hum_norm"]
        + 2.0 * df["Rainfall"]
        + (1.0 - df["air_norm"])
    ) / 4.0

    # Garante [0, 1]
    df["weather_impact_factor"] = df["weather_impact_factor"].clip(0, 1)
    return df[["season", "round", "weather_impact_factor"]]


def integrar_weather(df, weather_df, df_2024_agg):
    weather_agg = agregar_weather(weather_df)
    weather_feat = calcular_weather_impact(weather_agg, df_2024_agg)
    df = df.merge(weather_feat, on=["season", "round"], how="left")

    n_sem_weather = df["weather_impact_factor"].isnull().sum()
    if n_sem_weather > 0:
        mediana = df.loc[df["season"] <= 2024, "weather_impact_factor"].median()
        df["weather_impact_factor"] = df["weather_impact_factor"].fillna(mediana)
        print(f"  AVISO: {n_sem_weather} linhas sem weather → preenchido com mediana ({mediana:.3f}).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LACUNA 6 — avg_pit_stops_circuit
# Para cada corrida: conta o número máximo de stops por piloto.
# Agrega pela média histórica por circuit_id.
# ─────────────────────────────────────────────────────────────────────────────
def calcular_avg_pitstops_por_circuito(pitstop_df):
    # stop é ordinal (1, 2, 3...) — max por (season, round, driver) = total stops
    stops_por_corrida = (
        pitstop_df
        .groupby(["season", "round", "race_name", "driver_id"])["stop"]
        .max()
        .reset_index(name="total_stops")
    )

    stops_por_corrida["circuit_id"] = (
        stops_por_corrida["race_name"].map(RACE_NAME_TO_CIRCUIT_ID)
    )

    avg_stops = (
        stops_por_corrida
        .groupby("circuit_id")["total_stops"]
        .mean()
        .reset_index(name="avg_pit_stops_circuit")
    )
    return avg_stops


def integrar_pitstops(df, pitstop_df):
    avg_stops = calcular_avg_pitstops_por_circuito(pitstop_df)
    df = df.merge(avg_stops, on="circuit_id", how="left")

    n_sem_pitstop = df["avg_pit_stops_circuit"].isnull().sum()
    if n_sem_pitstop > 0:
        mediana = avg_stops["avg_pit_stops_circuit"].median()
        df["avg_pit_stops_circuit"] = df["avg_pit_stops_circuit"].fillna(mediana)
        print(f"  AVISO: {n_sem_pitstop} linhas sem pitstop → preenchido com mediana ({mediana:.2f}).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LACUNA 3 — safety_car_flag histórica
# Fonte principal: FastF1 TrackStatus em todas as temporadas 2018-2025.
# Códigos relevantes:
#   4 = Safety Car
#   5 = Virtual Safety Car
# TrackStatus pode combinar múltiplos códigos no mesmo valor (ex.: 41, 124).
# ─────────────────────────────────────────────────────────────────────────────
def construir_safety_car_fastf1(laps_df):
    df = laps_df[["season", "round", "TrackStatus"]].copy()
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    df["safety_car_lap_flag"] = (
        df["TrackStatus"].str.contains("4", regex=False)
        | df["TrackStatus"].str.contains("5", regex=False)
    ).astype(int)

    return (
        df.groupby(["season", "round"], as_index=False)["safety_car_lap_flag"]
        .max()
        .rename(columns={"safety_car_lap_flag": "safety_car_flag_fastf1"})
    )


def construir_safety_car_openf1_2025(race_control_df, meetings_df):
    sc_meetings = race_control_df[
        (race_control_df["category"] == "SafetyCar")
        & (race_control_df["message"] == "SAFETY CAR DEPLOYED")
    ]["meeting_key"].unique()

    meetings_2025 = meetings_df[meetings_df["season"] == 2025].copy()
    meetings_2025["safety_car_real"] = meetings_2025["meeting_key"].isin(sc_meetings).astype(int)

    # meeting_name coincide com race_name no dataset histórico
    sc_map = dict(zip(meetings_2025["meeting_name"], meetings_2025["safety_car_real"]))
    return sc_map


def integrar_safety_car(df, laps_df, race_control_df, meetings_df):
    df = df.copy()
    sc_fastf1 = construir_safety_car_fastf1(laps_df)
    sc_openf1_2025 = construir_safety_car_openf1_2025(race_control_df, meetings_df)

    df = df.drop(columns=["safety_car_flag"], errors="ignore")
    df = df.merge(sc_fastf1, on=["season", "round"], how="left")
    df["safety_car_flag"] = (
        df["safety_car_flag_fastf1"]
        .fillna(0)
        .astype(int)
    )
    df = df.drop(columns=["safety_car_flag_fastf1"])

    # Para 2025, combina FastF1 com OpenF1 Race Control como corroboracao.
    # Se qualquer uma das fontes indicar SC/VSC, a corrida recebe flag 1.
    mask_2025 = df["season"] == 2025
    openf1_flag = (
        df.loc[mask_2025, "race_name"]
        .map(sc_openf1_2025)
        .fillna(0)
        .astype(int)
    )
    df.loc[mask_2025, "safety_car_flag"] = np.maximum(
        df.loc[mask_2025, "safety_car_flag"].to_numpy(),
        openf1_flag.to_numpy(),
    )

    corridas_sc = df[df["safety_car_flag"] == 1].groupby(["season", "round"]).ngroups
    registros_sc = int(df["safety_car_flag"].sum())
    print(f"  safety_car_flag: {registros_sc} registros em {corridas_sc} corridas com SC/VSC.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Re-normalização do grid_position após correção do 0 → 21
# Refaz o MinMaxScaler somente para grid_position usando a base 2018-2024
# ─────────────────────────────────────────────────────────────────────────────
def renormalizar_grid_position(df_2024, df_2025):
    scaler = MinMaxScaler()
    scaler.fit(df_2024[["grid_position"]])
    joblib.dump(scaler, SCALER_GRID)

    df_2024 = df_2024.copy()
    df_2025 = df_2025.copy()
    df_2024["grid_position_minmax"] = scaler.transform(df_2024[["grid_position"]])[:, 0]
    df_2025["grid_position_minmax"] = scaler.transform(df_2025[["grid_position"]])[:, 0]
    return df_2024, df_2025


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
def processar(df_2024, df_2025, circuitos_df, weather_df, pitstop_df,
               race_control_df, meetings_df, qual_df, laps_df):

    print("\n[1/8] Corrigindo grid_position = 0 (pit lane start)...")
    df_2024, n_fix_2024 = corrigir_grid_position(df_2024)
    df_2025, n_fix_2025 = corrigir_grid_position(df_2025)

    print("\n[2/8] Integrando qualifying_position e grid_penalty...")
    df_2024 = integrar_qualifying(df_2024, qual_df)
    df_2025 = integrar_qualifying(df_2025, qual_df)

    print("\n[3/8] Garantindo circuit_id via race_name...")
    for df, label in [(df_2024, "2024"), (df_2025, "2025")]:
        if "circuit_id" not in df.columns:
            df["circuit_id"] = df["race_name"].map(RACE_NAME_TO_CIRCUIT_ID)
            print(f"  circuit_id adicionado em {label}.")
        else:
            print(f"  circuit_id já presente em {label}.")

    print("\n[4/8] Integrando circuit features (altitude, corners, length, track_complexity)...")
    df_2024 = integrar_circuitos(df_2024, circuitos_df)
    df_2025 = integrar_circuitos(df_2025, circuitos_df)

    print("\n[5/8] Calculando weather_impact_factor...")
    weather_agg_2024 = agregar_weather(weather_df[weather_df["season"] <= 2024])
    df_2024 = integrar_weather(df_2024, weather_df, weather_agg_2024)
    df_2025 = integrar_weather(df_2025, weather_df, weather_agg_2024)

    print("\n[6/8] Calculando avg_pit_stops_circuit...")
    df_2024 = integrar_pitstops(df_2024, pitstop_df)
    df_2025 = integrar_pitstops(df_2025, pitstop_df)

    print("\n[7/8] Integrando safety_car_flag histórico via FastF1 TrackStatus...")
    df_2024 = integrar_safety_car(df_2024, laps_df, race_control_df, meetings_df)
    df_2025 = integrar_safety_car(df_2025, laps_df, race_control_df, meetings_df)

    print("\n[8/8] Re-normalizando grid_position após correção...")
    df_2024, df_2025 = renormalizar_grid_position(df_2024, df_2025)

    return df_2024, df_2025, n_fix_2024, n_fix_2025


# ─────────────────────────────────────────────────────────────────────────────
# Relatório
# ─────────────────────────────────────────────────────────────────────────────
def salvar_relatorio(df_2024, df_2025, n_fix_2024, n_fix_2025,
                      weather_df, pitstop_df, circuitos_df, qual_df, laps_df):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO - 07 INTEGRAÇÃO DE FONTES DE SUPORTE\n")
        f.write("=" * 60 + "\n\n")

        f.write("ARQUIVOS DE ENTRADA\n")
        f.write("-" * 60 + "\n")
        f.write(f"{INPUT_2018_2024}\n")
        f.write(f"{INPUT_2018_2025}\n")
        f.write(f"{CIRCUITOS_MANUAL}\n")
        f.write(f"{WEATHER_FILE}\n")
        f.write(f"{PITSTOP_FILE}\n")
        f.write(f"{RACE_CONTROL_FILE}\n")
        f.write(f"{MEETINGS_FILE}\n")
        f.write(f"{QUALIFYING_FILE}\n\n")
        f.write(f"{FASTF1_LAPS_FILE}\n\n")

        f.write("ARQUIVOS DE SAÍDA\n")
        f.write("-" * 60 + "\n")
        f.write(f"{OUTPUT_2018_2024}\n")
        f.write(f"{OUTPUT_2018_2025}\n\n")

        f.write("CORREÇÕES APLICADAS\n")
        f.write("-" * 60 + "\n")
        f.write(f"grid_position 0→21 (2018-2024): {n_fix_2024} registros\n")
        f.write(f"grid_position 0→21 (2018-2025): {n_fix_2025} registros\n\n")

        f.write("QUALIFYING POSITION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Registros no arquivo de qualifying: {qual_df.shape[0]}\n")
        f.write(f"Cobertura 2018-2024: {df_2024['qualifying_position'].notna().mean():.1%}\n")
        penalizados = (df_2024['grid_penalty'] > 0).sum()
        promovidos = (df_2024['grid_penalty'] < 0).sum()
        sem_penalidade = (df_2024['grid_penalty'] == 0).sum()
        f.write(f"Penalizados (grid_penalty > 0): {penalizados}\n")
        f.write(f"Promovidos  (grid_penalty < 0): {promovidos}\n")
        f.write(f"Sem penalidade (grid_penalty = 0): {sem_penalidade}\n\n")

        f.write("CIRCUIT FEATURES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Circuitos integrados: {circuitos_df.shape[0]}\n")
        f.write("Colunas: altitude_m, corners, length_km, circuit_type, track_complexity\n\n")

        f.write("WEATHER IMPACT FACTOR\n")
        f.write("-" * 60 + "\n")
        f.write(f"Corridas com dados de weather: {weather_df.groupby(['season','round']).ngroups}\n")
        f.write("Formula: (norm_humidity + 2*rain_binary + (1-norm_air_temp)) / 4\n")
        f.write(f"weather_impact_factor 2018-2024 - mean: {df_2024['weather_impact_factor'].mean():.4f}\n")
        f.write(f"weather_impact_factor 2018-2024 - std:  {df_2024['weather_impact_factor'].std():.4f}\n\n")

        f.write("AVG PIT STOPS POR CIRCUITO\n")
        f.write("-" * 60 + "\n")
        f.write(f"Registros no arquivo de pitstops: {pitstop_df.shape[0]}\n")
        f.write(f"avg_pit_stops_circuit 2018-2024 - mean: {df_2024['avg_pit_stops_circuit'].mean():.4f}\n")
        f.write(f"avg_pit_stops_circuit 2018-2024 - std:  {df_2024['avg_pit_stops_circuit'].std():.4f}\n\n")

        f.write("SAFETY CAR FLAG\n")
        f.write("-" * 60 + "\n")
        f.write("Fonte principal: FastF1 TrackStatus, codigos 4=SC e 5=VSC.\n")
        f.write("Para 2025, OpenF1 Race Control foi usado como corroboracao adicional.\n")
        f.write(f"Registros FastF1 laps avaliados: {laps_df.shape[0]}\n")
        f.write(f"2018-2024: {df_2024['safety_car_flag'].sum()} registros com SC/VSC\n")
        f.write(f"2018-2025: {df_2025['safety_car_flag'].sum()} registros com SC/VSC\n")
        corridas_por_temporada = (
            df_2025[df_2025["safety_car_flag"] == 1]
            .groupby("season")["round"]
            .nunique()
        )
        f.write("Corridas com SC/VSC por temporada:\n")
        f.write(corridas_por_temporada.to_string())
        f.write("\n\n")

        f.write("DIMENSÕES FINAIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"2018-2024: {df_2024.shape}\n")
        f.write(f"2018-2025: {df_2025.shape}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────────────────────────
print("Carregando dados...")
df_2024 = pd.read_csv(INPUT_2018_2024)
df_2025 = pd.read_csv(INPUT_2018_2025)
circuitos_df = pd.read_csv(CIRCUITOS_MANUAL)
weather_df = pd.read_csv(WEATHER_FILE)
pitstop_df = pd.read_csv(PITSTOP_FILE)
race_control_df = pd.read_csv(RACE_CONTROL_FILE)
meetings_df = pd.read_csv(MEETINGS_FILE)
qual_df = pd.read_csv(QUALIFYING_FILE)
laps_df = pd.read_csv(FASTF1_LAPS_FILE, usecols=["season", "round", "TrackStatus"])

print(f"2018-2024: {df_2024.shape}  |  2018-2025: {df_2025.shape}")
print(f"Circuitos: {circuitos_df.shape[0]}  |  Weather: {weather_df.shape[0]} laps")
print(f"Qualifying: {qual_df.shape[0]} registros")
print(f"FastF1 laps TrackStatus: {laps_df.shape[0]} registros")

df_2024_out, df_2025_out, n_fix_2024, n_fix_2025 = processar(
    df_2024, df_2025, circuitos_df, weather_df, pitstop_df,
    race_control_df, meetings_df, qual_df, laps_df
)

print("\nSalvando outputs...")
df_2024_out.to_csv(OUTPUT_2018_2024, index=False, encoding="utf-8-sig")
df_2025_out.to_csv(OUTPUT_2018_2025, index=False, encoding="utf-8-sig")

salvar_relatorio(df_2024_out, df_2025_out, n_fix_2024, n_fix_2025,
                  weather_df, pitstop_df, circuitos_df, qual_df, laps_df)

print(f"\nArquivos salvos:")
print(f"  {OUTPUT_2018_2024}")
print(f"  {OUTPUT_2018_2025}")
print(f"  {REPORT_FILE}")

print(f"\nDimensões finais:")
print(f"  2018-2024: {df_2024_out.shape}")
print(f"  2018-2025: {df_2025_out.shape}")

print("\nEtapa 07 finalizada com sucesso.")
