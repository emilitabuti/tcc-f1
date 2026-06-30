from __future__ import annotations

import argparse
import time

import optuna
import pandas as pd
from xgboost import XGBRegressor

from tuning_utils import (
    FOLDS_AVALIACAO,
    FOLDS_TUNING,
    REPORTS_DIR,
    avaliar_modelo,
    carregar_dados,
    carregar_decay_escolhido,
    salvar_json,
    score_composto_metricas,
)


# caminhos de saida
OUTPUT_TRIALS = REPORTS_DIR / "optuna_xgboost_trials.csv"
OUTPUT_BEST_PARAMS = REPORTS_DIR / "optuna_xgboost_best_params.json"
OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_xgboost_tuned.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_walk_forward_xgboost_tuned.csv"


def criar_modelo(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
        **params,
    )


def sugerir_params(trial: optuna.Trial) -> dict:
    # espaço de busca - learning_rate em escala log porque varia muito
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    # função objetivo: testa os params nos folds de tuning e retorna o score
    def objective(trial: optuna.Trial) -> float:
        params = sugerir_params(trial)
        _, df_metricas = avaliar_modelo(
            x=x,
            y=y,
            criar_modelo=lambda: criar_modelo(params),
            folds=FOLDS_TUNING,
            decay=decay,
        )
        return score_composto_metricas(df_metricas)

    # roda o tuning e mede quanto tempo levou
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    inicio_tuning = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    tempo_tuning_segundos = time.perf_counter() - inicio_tuning

    best_params = dict(study.best_params)
    best_params["score_composto_tuning"] = float(study.best_value)
    best_params["tempo_tuning_segundos"] = float(tempo_tuning_segundos)
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_trials.to_csv(OUTPUT_TRIALS, index=False, encoding="utf-8-sig")
    salvar_json(OUTPUT_BEST_PARAMS, best_params)

    # só os hiperparâmetros de verdade vão pro modelo final
    model_params = {
        k: v
        for k, v in best_params.items()
        if k not in {"tempo_tuning_segundos", "score_composto_tuning"}
    }
    # avalia com todos os folds, incluindo 2025
    df_predicoes, df_metricas = avaliar_modelo(
        x=x,
        y=y,
        criar_modelo=lambda: criar_modelo(model_params),
        folds=FOLDS_AVALIACAO,
        decay=decay,
    )
    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")

    print("Tuning XGBoost concluido.")
    print(pd.Series(best_params).to_string())
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
