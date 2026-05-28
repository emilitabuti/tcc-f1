from __future__ import annotations

import argparse

import lightgbm as lgb
import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from tuning_utils import (
    FOLDS_AVALIACAO,
    FOLDS_TUNING,
    REPORTS_DIR,
    avaliar_modelo,
    carregar_dados,
    carregar_decay_escolhido,
    gerar_relatorio_tuning,
    salvar_json,
)


OUTPUT_TRIALS = REPORTS_DIR / "optuna_lightgbm_trials.csv"
OUTPUT_BEST_PARAMS = REPORTS_DIR / "optuna_lightgbm_best_params.json"
OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_lightgbm_tuned.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_walk_forward_lightgbm_tuned.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_quinta_semana2_lightgbm_tuning.txt"


def criar_modelo(params: dict) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        **params,
    )


def sugerir_params(trial: optuna.Trial) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    def objective(trial: optuna.Trial) -> float:
        params = sugerir_params(trial)
        _, df_metricas = avaliar_modelo(
            x=x,
            y=y,
            criar_modelo=lambda: criar_modelo(params),
            folds=FOLDS_TUNING,
            decay=decay,
        )
        return float(df_metricas["mae"].mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params["lightgbm_version"] = lgb.__version__
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_trials.to_csv(OUTPUT_TRIALS, index=False, encoding="utf-8-sig")
    salvar_json(OUTPUT_BEST_PARAMS, best_params)

    model_params = {k: v for k, v in best_params.items() if k != "lightgbm_version"}
    df_predicoes, df_metricas = avaliar_modelo(
        x=x,
        y=y,
        criar_modelo=lambda: criar_modelo(model_params),
        folds=FOLDS_AVALIACAO,
        decay=decay,
    )
    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")

    gerar_relatorio_tuning(
        modelo="LightGBM",
        data_cronograma="27/05 - acrescimo",
        caminho_relatorio=OUTPUT_RELATORIO,
        caminho_trials=OUTPUT_TRIALS,
        caminho_best_params=OUTPUT_BEST_PARAMS,
        caminho_predicoes=OUTPUT_PREDICOES,
        caminho_metricas=OUTPUT_METRICAS,
        df_metricas=df_metricas,
        best_value=float(study.best_value),
        best_params=best_params,
        decay=decay,
        n_trials=len(study.trials),
    )

    print("Tuning LightGBM concluido.")
    print(pd.Series(best_params).to_string())
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
