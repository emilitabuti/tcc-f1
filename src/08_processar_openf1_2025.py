from pathlib import Path
import pandas as pd


# 08 - Extrai e valida o conjunto de 2025 para walk-forward validation
# Fonte: dataset_pre_features_2018_2025.csv (output da etapa 07)
# Output principal: validacao_2025_clean.csv — conjunto isolado do season 2025,
#                   pronto para ser usado como fold de validação no walk-forward.
#
# Observacao: openf1_2025_clean.csv e mantido como alias de compatibilidade,
# mas a fonte deste arquivo e o pipeline Ergast/FastF1/Jolpica enriquecido,
# nao uma reconstrucao OpenF1-first.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "dataset_pre_features_2018_2025.csv"
OUTPUT_FILE = PROCESSED_DIR / "validacao_2025_clean.csv"
LEGACY_OUTPUT_FILE = PROCESSED_DIR / "openf1_2025_clean.csv"
REPORT_FILE = PROCESSED_DIR / "relatorio_08_openf1_2025.txt"


def validar_2025(df):
    erros = []
    if df.empty:
        erros.append("Dataset 2025 está vazio.")
    if df["season"].nunique() != 1 or df["season"].iloc[0] != 2025:
        erros.append("Dataset contém seasons diferentes de 2025.")
    colunas_essenciais = [
        "season", "round", "driver_id", "constructor_id", "circuit_id",
        "race_name", "grid_position", "finish_position", "points", "laps",
        "RaceID", "weather_impact_factor", "avg_pit_stops_circuit",
        "track_complexity", "safety_car_flag",
    ]
    for col in colunas_essenciais:
        if col not in df.columns:
            erros.append(f"Coluna essencial ausente: {col}")
    return erros


def gerar_sumario(df):
    corridas = df.groupby(["round", "race_name"]).size().reset_index(name="drivers")
    corridas_com_sc = (
        df[df["safety_car_flag"] == 1]
        .groupby("race_name").size()
        .rename("sc_drivers")
        .reset_index()
    )
    sumario = corridas.merge(corridas_com_sc, on="race_name", how="left")
    sumario["safety_car"] = sumario["sc_drivers"].notna().map({True: "sim", False: "nao"})
    sumario = sumario[["round", "race_name", "drivers", "safety_car"]]
    return sumario


# ─────────────────────────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────────────────────────
print("Carregando dataset completo (2018-2025)...")
df_completo = pd.read_csv(INPUT_FILE)
print(f"  Shape completo: {df_completo.shape}")

df_2025 = df_completo[df_completo["season"] == 2025].copy()
df_2025 = df_2025.reset_index(drop=True)
print(f"  Shape 2025: {df_2025.shape}")

print("\nValidando dataset 2025...")
erros = validar_2025(df_2025)
if erros:
    for e in erros:
        print(f"  ERRO: {e}")
    raise RuntimeError("Validação do dataset 2025 falhou.")
else:
    print("  Validação OK.")

sumario = gerar_sumario(df_2025)
print("\nResumo de corridas 2025:")
print(sumario.to_string(index=False))

df_2025.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
df_2025.to_csv(LEGACY_OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nDataset 2025 salvo em: {OUTPUT_FILE}")
print(f"Alias legado salvo em: {LEGACY_OUTPUT_FILE}")

# Relatório
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 08 DATASET 2025 PARA WALK-FORWARD VALIDATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Arquivo de entrada: {INPUT_FILE}\n")
    f.write(f"Arquivo de saída principal: {OUTPUT_FILE}\n")
    f.write(f"Alias legado: {LEGACY_OUTPUT_FILE}\n")
    f.write(
        "Observação metodológica: este arquivo é um fold de validação 2025 "
        "derivado do pipeline processado, não uma base OpenF1-first.\n\n"
    )
    f.write(f"Shape: {df_2025.shape}\n")
    f.write(f"Corridas: {df_2025['round'].nunique()}\n")
    f.write(f"Pilotos únicos: {df_2025['driver_id'].nunique()}\n")
    f.write(f"Construtores únicos: {df_2025['constructor_id'].nunique()}\n\n")
    f.write("CORRIDAS 2025\n")
    f.write("-" * 60 + "\n")
    f.write(sumario.to_string(index=False))
    f.write("\n\nESTATÍSTICAS DAS FEATURES\n")
    f.write("-" * 60 + "\n")
    features = ["weather_impact_factor", "avg_pit_stops_circuit",
                "track_complexity", "safety_car_flag", "grid_position"]
    f.write(df_2025[features].describe().to_string())
    f.write("\n")

print(f"Relatório salvo em: {REPORT_FILE}")
print("\nEtapa 08 finalizada com sucesso.")
