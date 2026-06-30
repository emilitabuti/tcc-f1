from __future__ import annotations

import argparse
import time

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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


# caminhos de saida dos artefatos do tuning
OUTPUT_TRIALS = REPORTS_DIR / "optuna_randomforest_trials.csv"
OUTPUT_BEST_PARAMS = REPORTS_DIR / "optuna_randomforest_best_params.json"
OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_randomforest_tuned.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_walk_forward_randomforest_tuned.csv"


def criar_modelo(params: dict) -> RandomForestRegressor:
    return RandomForestRegressor(
        random_state=42,
        n_jobs=4,
        bootstrap=True,
        **params,
    )


def sugerir_params(trial: optuna.Trial) -> dict:
    # espaço de busca do optuna - max_features aceita string ou float
    max_depth_opcao = trial.suggest_int("max_depth", 3, 15)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": max_depth_opcao,
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    # cada trial testa uma combinacao de params e devolve o score composto
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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    inicio_tuning = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    tempo_tuning_segundos = time.perf_counter() - inicio_tuning

    # pega os melhores params e adiciona metadados antes de salvar
    best_params = dict(study.best_params)
    best_params["score_composto_tuning"] = float(study.best_value)
    best_params["tempo_tuning_segundos"] = float(tempo_tuning_segundos)
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_trials.to_csv(OUTPUT_TRIALS, index=False, encoding="utf-8-sig")
    salvar_json(OUTPUT_BEST_PARAMS, best_params)

    # filtra fora as chaves que não são hiperparâmetros do modelo
    model_params = {
        k: v
        for k, v in best_params.items()
        if k not in {"tempo_tuning_segundos", "score_composto_tuning"}
    }
    df_predicoes, df_metricas = avaliar_modelo(
        x=x,
        y=y,
        criar_modelo=lambda: criar_modelo(model_params),
        folds=FOLDS_AVALIACAO,
        decay=decay,
    )
    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")

    print("Tuning Random Forest concluido.")
    print(pd.Series(best_params).to_string())
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
