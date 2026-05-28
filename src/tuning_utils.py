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

FOLDS_TUNING = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
]
FOLDS_AVALIACAO = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]
DECAY_PADRAO = 0.75


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float) -> np.ndarray:
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


def salvar_json(caminho: Path, dados: dict) -> None:
    caminho.write_text(
        json.dumps(dados, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def gerar_relatorio_tuning(
    modelo: str,
    data_cronograma: str,
    caminho_relatorio: Path,
    caminho_trials: Path,
    caminho_best_params: Path,
    caminho_predicoes: Path,
    caminho_metricas: Path,
    df_metricas: pd.DataFrame,
    best_value: float,
    best_params: dict,
    decay: float,
    n_trials: int,
    tempo_tuning_segundos: float | None = None,
) -> None:
    linhas = [
        f"Relatorio - {data_cronograma} - Tuning Optuna {modelo}",
        "=" * 64,
        "",
        "Objetivo:",
        "- Minimizar MAE medio nos folds de tuning 2023 e 2024.",
        "- Reavaliar o melhor conjunto em 2023, 2024 e 2025 via walk-forward.",
        "",
        "Bases utilizadas:",
        f"- {INPUT_X}",
        f"- {INPUT_Y}",
        "",
        "Time-decay:",
        f"- Fator usado via sample_weight: {decay}",
        "",
        "Optuna:",
        f"- Trials executados: {n_trials}",
        (
            f"- Tempo total de tuning: {tempo_tuning_segundos:.2f} segundos"
            if tempo_tuning_segundos is not None
            else "- Tempo total de tuning: nao registrado"
        ),
        f"- Melhor MAE medio nos folds de tuning: {best_value:.6f}",
        f"- Melhores parametros: {best_params}",
        "",
        "Metricas finais por fold:",
        df_metricas.to_string(index=False),
        "",
        "Artefatos gerados:",
        f"- {caminho_trials}",
        f"- {caminho_best_params}",
        f"- {caminho_predicoes}",
        f"- {caminho_metricas}",
        f"- {caminho_relatorio}",
    ]

    caminho_relatorio.write_text("\n".join(linhas), encoding="utf-8")
