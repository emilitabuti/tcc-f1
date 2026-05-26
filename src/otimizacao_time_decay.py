from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from metricas import calcular_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"

OUTPUT_RESULTADOS = REPORTS_DIR / "otimizacao_time_decay_xgboost.csv"
OUTPUT_RESUMO = REPORTS_DIR / "otimizacao_time_decay_xgboost_resumo.csv"
OUTPUT_ESCOLHIDO = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

DECAYS = [0.50, 0.65, 0.75, 0.85, 0.95]
FOLDS_OTIMIZACAO = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
]


def carregar_dados():
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    if x.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
        raise RuntimeError("X ou y ainda contem valores nulos.")

    return x, y


def criar_modelo_xgboost() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=4,
    )


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float):
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()


def avaliar_decay(x, y, decay: float, train_until: int, valid_season: int):
    train_mask = y["season"] <= train_until
    valid_mask = y["season"] == valid_season

    if train_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha de treino ate {train_until}.")
    if valid_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha de validacao para {valid_season}.")

    x_train = x.loc[train_mask]
    y_train = y.loc[train_mask]
    x_valid = x.loc[valid_mask]
    y_valid = y.loc[valid_mask].copy()

    sample_weight = calcular_sample_weight(
        y_train=y_train,
        valid_season=valid_season,
        decay=decay,
    )

    modelo = criar_modelo_xgboost()
    modelo.fit(
        x_train,
        y_train["finish_position"],
        sample_weight=sample_weight,
    )

    y_valid["pred_finish_position"] = modelo.predict(x_valid)
    metricas = calcular_metricas(y_valid)

    return {
        "decay": decay,
        "train_until": train_until,
        "valid_season": valid_season,
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        **metricas,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    resultados = []

    for decay in DECAYS:
        for fold in FOLDS_OTIMIZACAO:
            resultados.append(
                avaliar_decay(
                    x=x,
                    y=y,
                    decay=decay,
                    train_until=fold["train_until"],
                    valid_season=fold["valid_season"],
                )
            )

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(OUTPUT_RESULTADOS, index=False, encoding="utf-8-sig")

    resumo = (
        df_resultados
        .groupby("decay", as_index=False)
        .agg(
            mae_medio=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_medio=("rmse", "mean"),
            r2_medio=("r2", "mean"),
            kendall_tau_medio=("kendall_tau", "mean"),
            top3_accuracy_medio=("top3_accuracy", "mean"),
        )
        .sort_values(["mae_medio", "decay"])
        .reset_index(drop=True)
    )
    resumo.to_csv(OUTPUT_RESUMO, index=False, encoding="utf-8-sig")

    melhor = resumo.iloc[0]
    texto = (
        f"Time-decay escolhido: {melhor['decay']}\n"
        f"MAE medio 2023-2024: {melhor['mae_medio']:.6f}\n"
        f"Desvio padrao do MAE: {melhor['mae_std']:.6f}\n"
        f"RMSE medio 2023-2024: {melhor['rmse_medio']:.6f}\n"
        f"Kendall tau medio 2023-2024: {melhor['kendall_tau_medio']:.6f}\n"
        f"Acuracia top-3 media 2023-2024: {melhor['top3_accuracy_medio']:.6f}\n"
    )

    OUTPUT_ESCOLHIDO.write_text(texto, encoding="utf-8")

    print("Otimizacao de time-decay concluida.")
    print(resumo.to_string(index=False))
    print()
    print(texto)


if __name__ == "__main__":
    main()
