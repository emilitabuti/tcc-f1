from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_mae(y_true, y_pred) -> float:
    """Calcula erro absoluto medio em posicoes finais."""
    return float(mean_absolute_error(y_true, y_pred))


def calcular_rmse(y_true, y_pred) -> float:
    """Calcula RMSE para penalizar erros grandes de posicao."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calcular_r2(y_true, y_pred) -> float:
    """Calcula R2 do fold temporal avaliado."""
    return float(r2_score(y_true, y_pred))


def kendall_tau_por_corrida(df_pred: pd.DataFrame) -> float:
    """Calcula Kendall tau corrida a corrida e retorna a media valida.

    A correlacao e calculada dentro de cada GP para medir qualidade de ranking
    sem misturar pilotos de corridas diferentes. Corridas sem variacao no alvo
    ou com tau indefinido sao ignoradas.
    """
    valores = []

    for _, grupo in df_pred.groupby(["season", "round"]):
        if grupo["finish_position"].nunique() < 2:
            continue

        tau, _ = kendalltau(
            grupo["finish_position"],
            grupo["pred_finish_position"],
        )

        if not np.isnan(tau):
            valores.append(tau)

    if not valores:
        return float("nan")

    return float(np.mean(valores))


def calcular_metricas(df_pred: pd.DataFrame) -> dict:
    """Calcula o conjunto oficial de metricas do TCC.

    O pipeline oficial avalia regressao causal de `finish_position` por MAE,
    RMSE, R2 e Kendall tau. Metricas de top-3 nao entram nesta funcao porque
    pertencem a uma formulacao de classificacao de podio, nao ao criterio
    oficial de regressao/ranking.
    """
    colunas_obrigatorias = {
        "season",
        "round",
        "driver_id",
        "finish_position",
        "pred_finish_position",
    }
    faltantes = colunas_obrigatorias.difference(df_pred.columns)

    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(faltantes)}")

    y_true = df_pred["finish_position"]
    y_pred = df_pred["pred_finish_position"]

    return {
        "mae": calcular_mae(y_true, y_pred),
        "rmse": calcular_rmse(y_true, y_pred),
        "r2": calcular_r2(y_true, y_pred),
        "kendall_tau": kendall_tau_por_corrida(df_pred),
    }
