from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

arquivos = {
    "historico_ergast_fastf1_limpo_2018_2025.csv": PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2025.csv",
    "dataset_pre_features_2018_2025.csv": PROCESSED_DIR / "dataset_pre_features_2018_2025.csv",
    "dataset_features_final_2018_2025.csv": PROCESSED_DIR / "dataset_features_final_2018_2025.csv",
    "dataset_modelagem_2018_2025.csv": PROCESSED_DIR / "dataset_modelagem_2018_2025.csv",
    "dataset_modelagem_X_2018_2025.csv": PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv",
}

print("\n=== VERIFICAÇÃO DE grid_position = 0 ===\n")

for nome, caminho in arquivos.items():
    if not caminho.exists():
        print(f"{nome}: arquivo não encontrado")
        continue

    df = pd.read_csv(caminho)

    print(f"\nArquivo: {nome}")
    print(f"Linhas: {len(df)}")

    if "grid_position" in df.columns:
        qtd_zero = (df["grid_position"] == 0).sum()
        minimo = df["grid_position"].min()
        maximo = df["grid_position"].max()

        print(f"Possui grid_position: SIM")
        print(f"Quantidade grid_position = 0: {qtd_zero}")
        print(f"Mínimo grid_position: {minimo}")
        print(f"Máximo grid_position: {maximo}")
    else:
        print("Possui grid_position: NÃO")

    if "grid_position_zero_flag" in df.columns:
        qtd_flag = (df["grid_position_zero_flag"] == 1).sum()
        print(f"Possui grid_position_zero_flag: SIM")
        print(f"Quantidade flag = 1: {qtd_flag}")
    else:
        print("Possui grid_position_zero_flag: NÃO")

    if "qualifying_position" in df.columns:
        qtd_q0 = (df["qualifying_position"] == 0).sum()
        print(f"Possui qualifying_position: SIM")
        print(f"Quantidade qualifying_position = 0: {qtd_q0}")
    else:
        print("Possui qualifying_position: NÃO")

    if "grid_penalty" in df.columns:
        print("Possui grid_penalty: SIM")
    else:
        print("Possui grid_penalty: NÃO")