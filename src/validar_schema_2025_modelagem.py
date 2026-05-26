from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"
MODELS_DIR = BASE_DIR / "models" / "feature_selection"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
INPUT_2025 = PROCESSED_DIR / "openf1_2025_clean.csv"
INPUT_FEATURES = MODELS_DIR / "features_modelagem_2018_2025.json"

OUTPUT_REPORT = REPORTS_DIR / "validacao_schema_2025_modelagem.txt"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)
    df_2025 = pd.read_csv(INPUT_2025)

    erros = []

    if len(x) != len(y):
        erros.append(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    if x.isna().sum().sum() > 0:
        erros.append("X contem valores nulos.")

    if y.isna().sum().sum() > 0:
        erros.append("y contem valores nulos.")

    if df_2025.isna().sum().sum() > 0:
        erros.append("openf1_2025_clean.csv contem valores nulos.")

    if "season" not in y.columns:
        erros.append("Coluna season ausente em y.")
    elif set(y["season"].unique()) != {2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025}:
        erros.append(f"Temporadas inesperadas em y: {sorted(y['season'].unique())}")

    if "season" not in df_2025.columns:
        erros.append("Coluna season ausente em openf1_2025_clean.csv.")
    elif set(df_2025["season"].unique()) != {2025}:
        erros.append(
            f"openf1_2025_clean.csv contem seasons inesperadas: {sorted(df_2025['season'].unique())}"
        )

    y_2025 = y[y["season"] == 2025].copy()
    if len(y_2025) != len(df_2025):
        erros.append(
            f"Quantidade de linhas 2025 divergente entre y e openf1_2025_clean: "
            f"{len(y_2025)} vs {len(df_2025)}"
        )

    if "RaceID" in y_2025.columns and "RaceID" in df_2025.columns:
        ids_y = set(y_2025["RaceID"])
        ids_2025 = set(df_2025["RaceID"])
        faltando_no_fold = sorted(ids_y.difference(ids_2025))
        faltando_no_y = sorted(ids_2025.difference(ids_y))

        if faltando_no_fold:
            erros.append(f"RaceIDs de y ausentes no fold 2025: {len(faltando_no_fold)}")
        if faltando_no_y:
            erros.append(f"RaceIDs do fold 2025 ausentes em y: {len(faltando_no_y)}")

    features_numericas = [
        col for col in x.columns
        if pd.api.types.is_numeric_dtype(x[col])
    ]
    nao_numericas = sorted(set(x.columns).difference(features_numericas))
    if nao_numericas:
        erros.append(f"Features nao numericas em X: {nao_numericas}")

    linhas = [
        "Validacao de Schema - Fold 2025 para Modelagem",
        "=" * 52,
        "",
        "Arquivos verificados:",
        f"- {INPUT_X}",
        f"- {INPUT_Y}",
        f"- {INPUT_2025}",
        f"- {INPUT_FEATURES}",
        "",
        "Resumo:",
        f"- X shape: {x.shape}",
        f"- y shape: {y.shape}",
        f"- openf1_2025_clean shape: {df_2025.shape}",
        f"- Linhas 2025 em y: {len(y_2025)}",
        f"- Corridas 2025 no fold: {df_2025['round'].nunique() if 'round' in df_2025.columns else 'N/A'}",
        f"- Features em X: {len(x.columns)}",
        f"- Features numericas em X: {len(features_numericas)}",
        "",
        "Resultado:",
    ]

    if erros:
        linhas.extend(f"- ERRO: {erro}" for erro in erros)
    else:
        linhas.append("- OK: schema compativel para executar o walk-forward 2025.")

    OUTPUT_REPORT.write_text("\n".join(linhas), encoding="utf-8")

    if erros:
        raise RuntimeError("Validacao de schema falhou. Ver relatorio gerado.")

    print("Validacao de schema 2025 concluida.")
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
