from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from metricas import calcular_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
INPUT_DECAY = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_random_forest.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_walk_forward_random_forest.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_terca_semana2_random_forest.txt"

FOLDS = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]
DECAY_PADRAO = 0.75


def carregar_dados():
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    if x.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
        raise RuntimeError("X ou y ainda contem valores nulos.")

    return x, y


def carregar_decay_escolhido() -> float:
    if not INPUT_DECAY.exists():
        return DECAY_PADRAO

    for linha in INPUT_DECAY.read_text(encoding="utf-8").splitlines():
        if linha.lower().startswith("time-decay escolhido:"):
            return float(linha.split(":", 1)[1].strip())

    return DECAY_PADRAO


def criar_modelo_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=350,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=4,
    )


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float):
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()


def rodar_fold(x, y, train_until: int, valid_season: int, decay: float):
    train_mask = y["season"] <= train_until
    valid_mask = y["season"] == valid_season

    if train_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha de treino ate {train_until}.")
    if valid_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha encontrada para validacao {valid_season}.")

    sample_weight = calcular_sample_weight(
        y_train=y.loc[train_mask],
        valid_season=valid_season,
        decay=decay,
    )

    modelo = criar_modelo_random_forest()
    modelo.fit(
        x.loc[train_mask],
        y.loc[train_mask, "finish_position"],
        sample_weight=sample_weight,
    )

    pred = modelo.predict(x.loc[valid_mask])

    df_pred = y.loc[valid_mask].copy()
    df_pred["pred_finish_position"] = pred
    df_pred["train_until"] = train_until
    df_pred["valid_season"] = valid_season
    df_pred["decay"] = decay

    metricas = calcular_metricas(df_pred)
    metricas["train_until"] = train_until
    metricas["valid_season"] = valid_season
    metricas["decay"] = decay
    metricas["n_train"] = int(train_mask.sum())
    metricas["n_valid"] = int(valid_mask.sum())

    return df_pred, metricas


def gerar_relatorio(df_metricas: pd.DataFrame, decay: float):
    linhas = [
        "Relatorio - Terca Semana 2 - Modelagem Random Forest",
        "=" * 58,
        "",
        "Bases utilizadas:",
        f"- {INPUT_X}",
        f"- {INPUT_Y}",
        "",
        "Recorte temporal:",
        (
            "O cronograma menciona treino a partir de 2014, mas a base final "
            "de modelagem esta filtrada para 2018-2025. Portanto, os folds "
            "foram adaptados para o recorte real disponivel."
        ),
        "",
        "Folds executados:",
    ]

    for fold in FOLDS:
        linhas.append(
            f"- treino 2018-{fold['train_until']} -> validacao {fold['valid_season']}"
        )

    linhas.extend(
        [
            "",
            "Modelo:",
            "- Random Forest sem tuning Optuna, baseline preliminar da terca-feira.",
            "- Um novo RandomForestRegressor e criado em cada fold.",
            "- Parametros: n_estimators=350, max_features=sqrt, random_state=42.",
            "",
            "Metricas calculadas:",
            "- MAE",
            "- RMSE",
            "- R2",
            "- Kendall tau medio por corrida",
            "- Acuracia top-3",
            "",
            "Time-decay:",
            f"- Fator usado no walk-forward Random Forest: {decay}",
            "- Pesos passados via RandomForestRegressor.fit(..., sample_weight=sample_weight).",
            "",
            "Metricas por fold:",
            df_metricas.to_string(index=False),
            "",
            "Artefatos gerados:",
            f"- {OUTPUT_PREDICOES}",
            f"- {OUTPUT_METRICAS}",
            f"- {OUTPUT_RELATORIO}",
        ]
    )

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    predicoes = []
    metricas = []

    for fold in FOLDS:
        df_pred, fold_metricas = rodar_fold(
            x=x,
            y=y,
            train_until=fold["train_until"],
            valid_season=fold["valid_season"],
            decay=decay,
        )
        predicoes.append(df_pred)
        metricas.append(fold_metricas)

    df_predicoes = pd.concat(predicoes, ignore_index=True)
    df_metricas = pd.DataFrame(metricas)

    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")
    gerar_relatorio(df_metricas, decay)

    print("Walk-forward Random Forest concluido.")
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
