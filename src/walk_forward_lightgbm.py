from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from metricas import calcular_metricas


# caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
# aproveita o decay que foi otimizado junto com o XGBoost
INPUT_DECAY = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_lightgbm.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_walk_forward_lightgbm.csv"

# três folds: valida 2023, 2024 e 2025
FOLDS = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]
DECAY_PADRAO = 0.75


def carregar_dados():
    # lê os csvs de features e target
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    # verifica se os tamanhos batem
    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    if x.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
        raise RuntimeError("X ou y ainda contem valores nulos.")

    return x, y


def carregar_decay_escolhido() -> float:
    # se o arquivo não existe ainda, usa o padrão
    if not INPUT_DECAY.exists():
        return DECAY_PADRAO

    # lê o valor salvo na etapa de otimização
    for linha in INPUT_DECAY.read_text(encoding="utf-8").splitlines():
        if linha.lower().startswith("time-decay escolhido:"):
            return float(linha.split(":", 1)[1].strip())

    return DECAY_PADRAO


def criar_modelo_lightgbm() -> LGBMRegressor:
    # verbosity=-1 pra nao poluir o terminal com logs do LightGBM
    return LGBMRegressor(
        objective="regression",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float):
    # anos mais distantes da validação recebem peso menor
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()


def rodar_fold(x, y, train_until: int, valid_season: int, decay: float):
    # mascaras pra separar treino e validacao
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

    # modelo novo em cada fold, sem reaproveitar nada
    modelo = criar_modelo_lightgbm()
    modelo.fit(
        x.loc[train_mask],
        y.loc[train_mask, "finish_position"],
        sample_weight=sample_weight,
    )

    pred = modelo.predict(x.loc[valid_mask])

    # anota as predicoes junto com as infos de rastreabilidade
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


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    predicoes = []
    metricas = []

    # passa por cada fold e acumula os resultados
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

    # concatena tudo e salva os csvs
    df_predicoes = pd.concat(predicoes, ignore_index=True)
    df_metricas = pd.DataFrame(metricas)

    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")

    print("Walk-forward LightGBM concluido.")
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
