from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from metricas import calcular_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
INPUT_DECAY = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

# folds usados só durante o tuning (sem 2025 pra não vazar dados do futuro)
FOLDS_TUNING = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
]
# folds de avaliação final - inclui 2025
FOLDS_AVALIACAO = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]
DECAY_PADRAO = 0.75
METRIC_WEIGHTS = {
    "mae_score": 0.35,
    "rmse_score": 0.20,
    "r2_score": 0.20,
    "kendall_tau_score": 0.25,
}


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    if x.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
        raise RuntimeError("X ou y ainda contem valores nulos.")

    return x, y


def carregar_decay_escolhido() -> float:
    # se não encontrar o arquivo, usa o decay padrão
    if not INPUT_DECAY.exists():
        return DECAY_PADRAO

    for linha in INPUT_DECAY.read_text(encoding="utf-8").splitlines():
        if linha.lower().startswith("time-decay escolhido:"):
            return float(linha.split(":", 1)[1].strip())

    return DECAY_PADRAO


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float) -> np.ndarray:
    # corridas mais antigas recebem peso menor - decay elevado à distância em anos
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()


def rodar_fold(
    x: pd.DataFrame,
    y: pd.DataFrame,
    criar_modelo: Callable[[], object],
    train_until: int,
    valid_season: int,
    decay: float,
) -> tuple[pd.DataFrame, dict]:
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

    # treina e gera previsoes para o fold de validacao
    modelo = criar_modelo()
    modelo.fit(
        x.loc[train_mask],
        y.loc[train_mask, "finish_position"],
        sample_weight=sample_weight,
    )

    df_pred = y.loc[valid_mask].copy()
    df_pred["pred_finish_position"] = modelo.predict(x.loc[valid_mask])
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


def avaliar_modelo(
    x: pd.DataFrame,
    y: pd.DataFrame,
    criar_modelo: Callable[[], object],
    folds: list[dict],
    decay: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicoes = []
    metricas = []

    # roda fold a fold e junta tudo no final
    for fold in folds:
        df_pred, fold_metricas = rodar_fold(
            x=x,
            y=y,
            criar_modelo=criar_modelo,
            train_until=fold["train_until"],
            valid_season=fold["valid_season"],
            decay=decay,
        )
        predicoes.append(df_pred)
        metricas.append(fold_metricas)

    return pd.concat(predicoes, ignore_index=True), pd.DataFrame(metricas)


def score_composto_metricas(df_metricas: pd.DataFrame) -> float:
    # converte cada metrica em um valor entre 0 e 1 e pondera pelos pesos definidos
    medias = df_metricas[["mae", "rmse", "r2", "kendall_tau"]].mean()
    componentes = {
        "mae_score": 1.0 / (1.0 + float(medias["mae"])),
        "rmse_score": 1.0 / (1.0 + float(medias["rmse"])),
        "r2_score": max(0.0, min(1.0, (float(medias["r2"]) + 1.0) / 2.0)),
        "kendall_tau_score": max(0.0, min(1.0, (float(medias["kendall_tau"]) + 1.0) / 2.0)),
    }
    return float(
        sum(METRIC_WEIGHTS[coluna] * componentes[coluna] for coluna in METRIC_WEIGHTS)
    )


def salvar_json(caminho: Path, dados: dict) -> None:
    caminho.write_text(
        json.dumps(dados, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
