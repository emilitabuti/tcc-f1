from __future__ import annotations

"""
OBSOLETO / HISTORICO.

Este script explorava target_delta = finish_position - qualifying_position.
Ele foi desativado na versao final porque o target oficial do TCC nao pode
mudar: todos os modelos comparaveis devem prever diretamente finish_position.
"""

import time
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from estudos_ablacao_modelos import avaliar_regressao, preparar_x, resumir_metricas
from tuning_utils import (
    BASE_DIR,
    FOLDS_AVALIACAO,
    FOLDS_TUNING,
    calcular_sample_weight,
    score_composto_metricas,
)


PROCESSED_DIR = BASE_DIR / "data" / "processed"
ABLATION_DIR = BASE_DIR / "reports" / "ablacao"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"

DECAY = 0.99


def carregar_dados():
    return pd.read_csv(INPUT_X), pd.read_csv(INPUT_Y)


def criar_lgb(params: dict) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        **params,
    )


def criar_xgb(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
        **params,
    )


def sugerir_lgb(trial: optuna.Trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 3, 10)
    max_leaves = min(127, 2**max_depth - 1)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 7, max_leaves),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
    }


def sugerir_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
    }


def avaliar_tuning_delta(x: pd.DataFrame, y: pd.DataFrame, criar_modelo):
    metricas = []
    for fold in FOLDS_TUNING:
        train_mask = y["season"] <= fold["train_until"]
        valid_mask = y["season"] == fold["valid_season"]
        sample_weight = calcular_sample_weight(
            y_train=y.loc[train_mask],
            valid_season=fold["valid_season"],
            decay=DECAY,
        )
        target_delta = (
            y.loc[train_mask, "finish_position"]
            - x.loc[train_mask, "qualifying_position"]
        )
        model = criar_modelo()
        model.fit(x.loc[train_mask], target_delta, sample_weight=sample_weight)
        pred_delta = model.predict(x.loc[valid_mask])

        df_pred = y.loc[valid_mask].copy()
        df_pred["pred_finish_position"] = (
            x.loc[valid_mask, "qualifying_position"].to_numpy() + pred_delta
        )
        from metricas import calcular_metricas

        metricas.append(calcular_metricas(df_pred))
    return pd.DataFrame(metricas)


def tunar_modelo(nome: str, x: pd.DataFrame, y: pd.DataFrame, sugerir, criar, trials: int):
    def objective(trial: optuna.Trial) -> float:
        params = sugerir(trial)
        met = avaliar_tuning_delta(x, y, lambda: criar(params))
        return score_composto_metricas(met)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    inicio = time.perf_counter()
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    tempo = time.perf_counter() - inicio

    best_params = dict(study.best_params)
    _, metricas_final = avaliar_regressao(
        x=x,
        y=y,
        criar_modelo=lambda: criar(best_params),
        decay=DECAY,
        target_mode="delta_grid",
    )
    resumo = resumir_metricas(
        nome=f"{nome}_target_delta_retuned",
        grupo="target_retuned",
        modelo=nome,
        metricas=metricas_final,
    )
    resumo["score_tuning"] = float(study.best_value)
    resumo["tempo_tuning_segundos"] = tempo

    pd.DataFrame([resumo]).to_csv(
        ABLATION_DIR / f"resultado_{nome.lower()}_target_delta_retuned.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metricas_final.to_csv(
        ABLATION_DIR / f"metricas_{nome.lower()}_target_delta_retuned.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (ABLATION_DIR / f"params_{nome.lower()}_target_delta_retuned.json").write_text(
        pd.Series(best_params).to_json(indent=2),
        encoding="utf-8",
    )
    return resumo


def main() -> None:
    raise SystemExit(
        "Script obsoleto: target_delta foi rejeitado na versao final. "
        "Use src/ablacao_pareada_lgbm_xgboost.py ou os scripts oficiais com target finish_position."
    )


if __name__ == "__main__":
    main()
