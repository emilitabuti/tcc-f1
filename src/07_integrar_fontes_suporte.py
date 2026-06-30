from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


# 07 - integra circuit features, weather, pitstops, safety car e corrige grid=0
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models" / "preprocessing"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Arquivos de entrada
INPUT_2018_2024 = PROCESSED_DIR / "historico_outliers_tratados_2018_2024.csv"
INPUT_2018_2025 = PROCESSED_DIR / "historico_outliers_tratados_2018_2025.csv"

CIRCUITOS_MANUAL = RAW_DIR / "circuitos_manual.csv"
WEATHER_FILE = RAW_DIR / "fastf1_weather_2018_2025.csv"
PITSTOP_FILE = RAW_DIR / "ergast_pitstop_2018_2025.csv"
QUALIFYING_FILE = RAW_DIR / "fastf1_qualifying_2018_2025.csv"
FASTF1_LAPS_FILE = RAW_DIR / "fastf1_laps_2018_2025.csv"

# Arquivos de saída
OUTPUT_2018_2024 = PROCESSED_DIR / "dataset_pre_features_2018_2024.csv"
OUTPUT_2018_2025 = PROCESSED_DIR / "dataset_pre_features_2018_2025.csv"

SCALER_WEATHER = MODELS_DIR / "scaler_weather_impact.joblib"
SCALER_PITSTOPS = MODELS_DIR / "scaler_avg_pitstops.joblib"
SCALER_TRACK = MODELS_DIR / "scaler_track_complexity.joblib"
SCALER_GRID = MODELS_DIR / "scaler_grid_position_fixed.joblib"

# Mapeamento código FastF1 -> driver_id Ergast
# mesma tabela usada na limpeza - assim os códigos ficam consistentes em todo o projeto
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

# Mapeamento race_name -> circuit_id
# aqui a gente converte o nome da corrida pro id do circuito que o Ergast usa
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


# adiciona qualifying_position e grid_penalty que faltavam no dataset
# grid_penalty = grid_position - qualifying_position (positivo = penalidade)
def preparar_qualifying(qual_df):
    df = qual_df.copy()
    # converte o código do piloto pro driver_id do Ergast
    df["driver_id"] = df["Driver"].map(DRIVER_CODE_TO_ID)
    df = df.rename(columns={"position": "qualifying_position"})
    df["qualifying_position"] = pd.to_numeric(df["qualifying_position"], errors="coerce")
    return df[["season", "round", "driver_id", "qualifying_position"]]


def integrar_qualifying(df, qual_df):
    df = df.copy()
    qual_preparado = preparar_qualifying(qual_df)

    df = df.merge(qual_preparado, on=["season", "round", "driver_id"], how="left")

    # pit lane start: grid_position já foi corrigido pra 21, então usa o mesmo valor
    mask_pit = df["grid_position_zero_flag"] == 1
    df.loc[mask_pit, "qualifying_position"] = df.loc[mask_pit, "grid_position"]

    # sem dados de qualifying: usa grid_position como proxy (sem penalidade conhecida)
    sem_qualifying = df["qualifying_position"].isnull()
    df.loc[sem_qualifying, "qualifying_position"] = df.loc[sem_qualifying, "grid_position"]

    df["qualifying_position"] = df["qualifying_position"].astype(int)

    # positivo = perdeu posicoes, negativo = ganhou por penalidade de outro
    df["grid_penalty"] = (df["grid_position"] - df["qualifying_position"]).astype(int)

    n_sem = sem_qualifying.sum()
    n_penalizados = (df["grid_penalty"] > 0).sum()
    taxa_cobertura = 1 - (n_sem / len(df))
    print(f"  qualifying_position: cobertura {taxa_cobertura:.1%} ({n_sem} sem dados -> proxy grid_pos)")
    print(f"  grid_penalty > 0 (penalidades): {n_penalizados} registros")
    print(f"  grid_penalty < 0 (promoções):   {(df['grid_penalty'] < 0).sum()} registros")

    return df


# grid = 0 é pit lane start - substitui por 21 pra nao baguncar o scaler
def corrigir_grid_position(df):
    df = df.copy()
    # grid = 0 significa que o piloto saiu do pit lane, não que largou na frente
    mask = df["grid_position"] == 0
    n = mask.sum()
    if n > 0:
        df.loc[mask, "grid_position"] = 21
        print(f"  grid_position: {n} pit-lane starts corrigidos (0 -> 21).")
    return df, n


# integra dados do circuito: altitude, curvas, comprimento e track_complexity
def construir_track_complexity(circuitos_df):
    df = circuitos_df.copy()

    # normaliza cada variavel pro intervalo [0,1] usando os circuitos como universo
    for col in ["corners", "length_km", "altitude_m"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)

    # indice ponderado: curvas tem mais peso porque afetam mais o ritmo
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
    # junta pelo circuit_id - se nao tiver mapeamento, a linha fica com NaN
    df = df.merge(circ_features, on="circuit_id", how="left")
    n_sem_circuito = df["track_complexity"].isnull().sum()
    if n_sem_circuito > 0:
        print(f"  AVISO: {n_sem_circuito} linhas sem circuit_id mapeado em circuitos_manual.")
    return df


# calcula weather_impact_factor usando so o historico anterior da corrida
def agregar_weather(weather_df):
    # resume o clima de cada corrida: temperatura média, umidade média e se choveu
    agg = weather_df.groupby(["season", "round"], as_index=False).agg(
        AirTemp=("AirTemp", "mean"),
        Humidity=("Humidity", "mean"),
        Rainfall=("Rainfall", "max"),
    )
    agg["Rainfall"] = agg["Rainfall"].astype(int)
    return agg


def calcular_weather_impact_observado(weather_agg):
    df = weather_agg.copy()

    # escalas físicas fixas pra não vazar info do futuro na normalização
    df["air_norm"] = (df["AirTemp"] / 45.0).clip(0, 1)
    df["hum_norm"] = (df["Humidity"] / 100.0).clip(0, 1)

    # chuva tem peso 2x porque é o fator que mais bagunça a corrida
    df["weather_impact_observed"] = (
        df["hum_norm"]
        + 2.0 * df["Rainfall"]
        + (1.0 - df["air_norm"])
    ) / 4.0

    df["weather_impact_observed"] = df["weather_impact_observed"].clip(0, 1)
    return df[["season", "round", "weather_impact_observed"]]


def calcular_weather_impact_causal(df, weather_df):
    weather_agg = agregar_weather(weather_df)
    weather_obs = calcular_weather_impact_observado(weather_agg)

    race_weather = (
        df[["season", "round", "circuit_id"]]
        .drop_duplicates()
        .merge(weather_obs, on=["season", "round"], how="left")
        .sort_values(["season", "round", "circuit_id"])
        .reset_index(drop=True)
    )

    # expanding().mean().shift(1): pega só o histórico anterior, sem olhar a corrida atual
    race_weather["weather_impact_factor"] = (
        race_weather.groupby("circuit_id")["weather_impact_observed"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    # quando não tem histórico do circuito, usa a média global anterior como fallback
    race_weather["weather_impact_global_prior"] = (
        race_weather["weather_impact_observed"].expanding().mean().shift(1)
    )
    race_weather["weather_impact_cold_start_flag"] = (
        race_weather["weather_impact_factor"].isna()
    ).astype(int)
    race_weather["weather_impact_factor"] = (
        race_weather["weather_impact_factor"]
        .fillna(race_weather["weather_impact_global_prior"])
        .fillna(0.0)  # primeira corrida de todas não tem histórico nenhum
        .clip(0, 1)
    )

    return race_weather[
        [
            "season",
            "round",
            "circuit_id",
            "weather_impact_factor",
            "weather_impact_observed",
            "weather_impact_cold_start_flag",
        ]
    ]


def integrar_weather(df, weather_df):
    df = df.copy()
    weather_feat = calcular_weather_impact_causal(df, weather_df)
    # remove colunas antigas de weather se ja existirem, pra nao duplicar
    df = df.drop(
        columns=[
            "weather_impact_factor",
            "weather_impact_observed",
            "weather_impact_cold_start_flag",
        ],
        errors="ignore",
    )
    df = df.merge(weather_feat, on=["season", "round", "circuit_id"], how="left")

    n_sem_weather = df["weather_impact_factor"].isnull().sum()
    if n_sem_weather > 0:
        df["weather_impact_factor"] = df["weather_impact_factor"].fillna(0.0)
        df["weather_impact_cold_start_flag"] = df["weather_impact_cold_start_flag"].fillna(1).astype(int)
        print(f"  AVISO: {n_sem_weather} linhas sem weather historico -> preenchido com 0.0.")
    return df


# calcula media historica de pitstops por circuito
def calcular_avg_pitstops_por_circuito(pitstop_df):
    # o campo "stop" e ordinal (1, 2, 3...), entao o max por piloto = total de paradas
    stops_por_corrida = (
        pitstop_df
        .groupby(["season", "round", "race_name", "driver_id"])["stop"]
        .max()
        .reset_index(name="total_stops")
    )

    # converte o nome da corrida pro circuit_id pra poder agrupar por pista
    stops_por_corrida["circuit_id"] = (
        stops_por_corrida["race_name"].map(RACE_NAME_TO_CIRCUIT_ID)
    )

    # média de paradas por circuito ao longo de toda a base histórica
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

    # circuitos sem histórico de pit stop ficam com a mediana da base toda
    n_sem_pitstop = df["avg_pit_stops_circuit"].isnull().sum()
    if n_sem_pitstop > 0:
        mediana = avg_stops["avg_pit_stops_circuit"].median()
        df["avg_pit_stops_circuit"] = df["avg_pit_stops_circuit"].fillna(mediana)
        print(f"  AVISO: {n_sem_pitstop} linhas sem pitstop -> preenchido com mediana ({mediana:.2f}).")
    return df


# safety_car_flag via TrackStatus do FastF1: codigos 4 (SC), 6 e 7 (VSC)
def construir_safety_car_fastf1(laps_df):
    df = laps_df[["season", "round", "TrackStatus"]].copy()
    df["TrackStatus"] = df["TrackStatus"].astype(str)

    # marca as voltas onde houve SC ou VSC (codigos 4, 6, 7)
    df["safety_car_lap_flag"] = (
        df["TrackStatus"].str.contains("4", regex=False)
        | df["TrackStatus"].str.contains("6", regex=False)
        | df["TrackStatus"].str.contains("7", regex=False)
    ).astype(int)

    # se qualquer volta da corrida teve SC/VSC, a corrida toda recebe flag = 1
    return (
        df.groupby(["season", "round"], as_index=False)["safety_car_lap_flag"]
        .max()
        .rename(columns={"safety_car_lap_flag": "safety_car_flag_fastf1"})
    )



def integrar_safety_car(df, laps_df):
    df = df.copy()
    sc_fastf1 = construir_safety_car_fastf1(laps_df)

    # joga fora a coluna antiga de safety car pra refazer com a fonte FastF1
    df = df.drop(columns=["safety_car_flag"], errors="ignore")
    df = df.merge(sc_fastf1, on=["season", "round"], how="left")
    df["safety_car_flag"] = (
        df["safety_car_flag_fastf1"]
        .fillna(0)
        .astype(int)
    )
    df = df.drop(columns=["safety_car_flag_fastf1"])

    corridas_sc = df[df["safety_car_flag"] == 1].groupby(["season", "round"]).ngroups
    registros_sc = int(df["safety_car_flag"].sum())
    print(f"  safety_car_flag: {registros_sc} registros em {corridas_sc} corridas com SC/VSC.")
    return df


# refaz o MinMax do grid_position depois de corrigir o 0 -> 21
def renormalizar_grid_position(df_2024, df_2025):
    # treina o scaler só na base até 2024 pra não vazar info de 2025
    scaler = MinMaxScaler()
    scaler.fit(df_2024[["grid_position"]])
    joblib.dump(scaler, SCALER_GRID)

    df_2024 = df_2024.copy()
    df_2025 = df_2025.copy()
    df_2024["grid_position_minmax"] = scaler.transform(df_2024[["grid_position"]])[:, 0]
    df_2025["grid_position_minmax"] = scaler.transform(df_2025[["grid_position"]])[:, 0]
    return df_2024, df_2025


# pipeline principal
def processar(df_2024, df_2025, circuitos_df, weather_df, pitstop_df,
               qual_df, laps_df):

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

    print("\n[5/8] Calculando weather_impact_factor historico causal...")
    df_2024 = integrar_weather(df_2024, weather_df)
    df_2025 = integrar_weather(df_2025, weather_df)

    print("\n[6/8] Calculando avg_pit_stops_circuit...")
    df_2024 = integrar_pitstops(df_2024, pitstop_df)
    df_2025 = integrar_pitstops(df_2025, pitstop_df)

    print("\n[7/8] Integrando safety_car_flag histórico via FastF1 TrackStatus...")
    df_2024 = integrar_safety_car(df_2024, laps_df)
    df_2025 = integrar_safety_car(df_2025, laps_df)

    print("\n[8/8] Re-normalizando grid_position após correção...")
    df_2024, df_2025 = renormalizar_grid_position(df_2024, df_2025)

    return df_2024, df_2025, n_fix_2024, n_fix_2025


# execucao
print("Carregando dados...")
df_2024 = pd.read_csv(INPUT_2018_2024)
df_2025 = pd.read_csv(INPUT_2018_2025)
circuitos_df = pd.read_csv(CIRCUITOS_MANUAL)
weather_df = pd.read_csv(WEATHER_FILE)
pitstop_df = pd.read_csv(PITSTOP_FILE)
qual_df = pd.read_csv(QUALIFYING_FILE)
# só carrega as colunas necessárias do arquivo de laps pra não travar a memória
laps_df = pd.read_csv(FASTF1_LAPS_FILE, usecols=["season", "round", "TrackStatus"])

print(f"2018-2024: {df_2024.shape}  |  2018-2025: {df_2025.shape}")
print(f"Circuitos: {circuitos_df.shape[0]}  |  Weather: {weather_df.shape[0]} laps")
print(f"Qualifying: {qual_df.shape[0]} registros")
print(f"FastF1 laps TrackStatus: {laps_df.shape[0]} registros")

df_2024_out, df_2025_out, n_fix_2024, n_fix_2025 = processar(
    df_2024, df_2025, circuitos_df, weather_df, pitstop_df,
    qual_df, laps_df
)

print("\nSalvando outputs...")
df_2024_out.to_csv(OUTPUT_2018_2024, index=False, encoding="utf-8-sig")
df_2025_out.to_csv(OUTPUT_2018_2025, index=False, encoding="utf-8-sig")

print(f"\nArquivos salvos:")
print(f"  {OUTPUT_2018_2024}")
print(f"  {OUTPUT_2018_2025}")

print(f"\nDimensões finais:")
print(f"  2018-2024: {df_2024_out.shape}")
print(f"  2018-2025: {df_2025_out.shape}")

print("\nEtapa 07 finalizada com sucesso.")
