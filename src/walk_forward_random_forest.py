from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from metricas import calcular_metricas


# caminhos base
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
# reutiliza o decay que foi escolhido no XGBoost
INPUT_DECAY = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_random_forest.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_walk_forward_random_forest.csv"

FOLDS = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]
DECAY_PADRAO = 0.75


def carregar_dados():
    # carrega features e target
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    # nulo aqui quebraria o treino, melhor parar logo
    if x.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
        raise RuntimeError("X ou y ainda contem valores nulos.")

    return x, y


def carregar_decay_escolhido() -> float:
    if not INPUT_DECAY.exists():
        return DECAY_PADRAO

    # pega o valor que foi salvo na otimização do XGBoost
    for linha in INPUT_DECAY.read_text(encoding="utf-8").splitlines():
        if linha.lower().startswith("time-decay escolhido:"):
            return float(linha.split(":", 1)[1].strip())

    return DECAY_PADRAO


def criar_modelo_random_forest() -> RandomForestRegressor:
    # baseline com Random Forest, sem tuning por enquanto
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
    # quanto mais longe do ano de validacao, menos peso
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()


def rodar_fold(x, y, train_until: int, valid_season: int, decay: float):
    # separa treino (até train_until) e validação (só valid_season)
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

    # treina com os pesos de time-decay
    modelo = criar_modelo_random_forest()
    modelo.fit(
        x.loc[train_mask],
        y.loc[train_mask, "finish_position"],
        sample_weight=sample_weight,
    )

    pred = modelo.predict(x.loc[valid_mask])

    # monta o resultado desse fold
    df_pred = y.loc[valid_mask].copy()
    df_pred["pred_finish_position"] = pred
    df_pred["train_until"] = train_until
    df_pred["valid_season"] = valid_season
    df_pred["decay"] = decay

    # calcula métricas e anota o contexto do fold
    metricas = calcular_metricas(df_pred)
    metricas["train_until"] = train_until
    metricas["valid_season"] = valid_season
    metricas["decay"] = decay
    metricas["n_train"] = int(train_mask.sum())
    metricas["n_valid"] = int(valid_mask.sum())

    return df_pred, metricas


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    predicoes = []
    metricas = []

    # roda os três folds e vai guardando os resultados
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

    # une tudo e salva
    df_predicoes = pd.concat(predicoes, ignore_index=True)
    df_metricas = pd.DataFrame(metricas)

    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")

    print("Walk-forward Random Forest concluido.")
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
