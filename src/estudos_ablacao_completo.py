from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from metricas import calcular_metricas
from estudos_ablacao_modelos import (
    carregar_dados,
    carregar_params,
    preparar_x,
    rodar_feature_ablations,
)
from tuning_utils import BASE_DIR, FOLDS_AVALIACAO, FOLDS_TUNING, REPORTS_DIR, calcular_sample_weight


PROCESSED_DIR = BASE_DIR / "data" / "processed"
ABLATION_DIR = BASE_DIR / "reports" / "ablacao"
FEATURE_SELECTION_DIR = BASE_DIR / "models" / "feature_selection"

LGB_PARAMS = REPORTS_DIR / "optuna_lightgbm_best_params.json"
XGB_PARAMS = REPORTS_DIR / "optuna_xgboost_best_params.json"

DEFAULT_DECAY = 0.99
DEFAULT_TRIALS = 30
RANK_NORM_GRID_SIZE = 20

# perfis de ponderacao do score composto pra testar se mudar os pesos altera o ranking
SCORE_PROFILES = {
    "atual": {
        "mae_score": 0.35,
        "rmse_score": 0.20,
        "r2_score": 0.20,
        "kendall_tau_score": 0.25,
    },
    "rmse_r2": {
        "mae_score": 0.25,
        "rmse_score": 0.30,
        "r2_score": 0.30,
        "kendall_tau_score": 0.15,
    },
    "ranking": {
        "mae_score": 0.25,
        "rmse_score": 0.15,
        "r2_score": 0.20,
        "kendall_tau_score": 0.40,
    },
    "erro_continuo": {
        "mae_score": 0.40,
        "rmse_score": 0.30,
        "r2_score": 0.20,
        "kendall_tau_score": 0.10,
    },
}


def score_metricas(df_metricas: pd.DataFrame, weights: dict[str, float] | None = None) -> float:
    weights = weights or SCORE_PROFILES["atual"]
    medias = df_metricas[["mae", "rmse", "r2", "kendall_tau"]].mean()
    # normaliza cada métrica pra escala 0-1 antes de ponderar
    componentes = {
        "mae_score": 1.0 / (1.0 + float(medias["mae"])),
        "rmse_score": 1.0 / (1.0 + float(medias["rmse"])),
        "r2_score": max(0.0, min(1.0, (float(medias["r2"]) + 1.0) / 2.0)),
        "kendall_tau_score": max(0.0, min(1.0, (float(medias["kendall_tau"]) + 1.0) / 2.0)),
    }
    return float(sum(weights[k] * componentes[k] for k in weights))


def resumo(
    grupo: str,
    experimento: str,
    modelo: str,
    metricas: pd.DataFrame,
    **extras,
) -> dict:
    # monta o dicionário com as médias das métricas e os campos extras que vierem
    row = {
        "grupo": grupo,
        "experimento": experimento,
        "modelo": modelo,
        "mae_medio": metricas["mae"].mean(),
        "rmse_medio": metricas["rmse"].mean(),
        "r2_medio": metricas["r2"].mean(),
        "kendall_tau_medio": metricas["kendall_tau"].mean(),
        "score_composto": score_metricas(metricas),
    }
    row.update(extras)
    return row


def cast_params(params: dict, model: str) -> dict:
    # alguns params precisam ser int, nao float - aqui converte os que precisam
    int_keys = {"n_estimators", "max_depth"}
    if model == "lgb":
        int_keys |= {"num_leaves", "min_child_samples"}
    return {k: int(v) if k in int_keys else float(v) for k, v in params.items()}


def sugerir_lgb(trial: optuna.Trial) -> dict:
    # espaco de busca dos hiperparametros do LightGBM
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
    # espaco de busca dos hiperparametros do XGBoost
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }


def criar_regressor(modelo: str, params: dict, objective: str):
    if modelo == "LightGBM":
        return LGBMRegressor(
            objective=objective,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
            **cast_params(params, "lgb"),
        )
    return XGBRegressor(
        objective=objective,
        random_state=42,
        n_jobs=4,
        **cast_params(params, "xgb"),
    )


def target_train(x: pd.DataFrame, y: pd.DataFrame, mask: pd.Series, mode: str) -> pd.Series:
    finish = y.loc[mask, "finish_position"]
    if mode == "finish":
        return finish
    if mode == "delta_grid":
        return finish - x.loc[mask, "qualifying_position"]
    if mode == "log1p_finish":
        return np.log1p(finish)
    # normaliza pela quantidade de carros na corrida
    if mode == "rank_norm":
        sizes = y.loc[mask].groupby(["season", "round"])["finish_position"].transform("max")
        return (finish - 1.0) / (sizes - 1.0).replace(0, np.nan)
    if mode == "rank_norm_grid20":
        return (finish - 1.0) / (RANK_NORM_GRID_SIZE - 1.0)
    raise ValueError(f"Target desconhecido: {mode}")


def target_pred(x_valid: pd.DataFrame, y_valid: pd.DataFrame, pred: np.ndarray, mode: str) -> np.ndarray:
    # desfaz a transformacao do target pra voltar a escala de finish_position
    if mode == "finish":
        return pred
    if mode == "delta_grid":
        return x_valid["qualifying_position"].to_numpy() + pred
    if mode == "log1p_finish":
        return np.expm1(pred)
    if mode == "rank_norm":
        sizes = y_valid.groupby(["season", "round"])["finish_position"].transform("max")
        return 1.0 + pred * (sizes.to_numpy() - 1.0)
    if mode == "rank_norm_grid20":
        return 1.0 + pred * (RANK_NORM_GRID_SIZE - 1.0)
    raise ValueError(f"Target desconhecido: {mode}")


def avaliar_regressor(
    x: pd.DataFrame,
    y: pd.DataFrame,
    criar_modelo: Callable[[], object],
    folds: list[dict],
    decay: float,
    target_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicoes = []
    metricas = []
    for fold in folds:
        train_mask = y["season"] <= fold["train_until"]
        valid_mask = y["season"] == fold["valid_season"]
        sample_weight = calcular_sample_weight(y.loc[train_mask], fold["valid_season"], decay)
        model = criar_modelo()
        model.fit(x.loc[train_mask], target_train(x, y, train_mask, target_mode), sample_weight=sample_weight)
        pred = target_pred(
            x.loc[valid_mask],
            y.loc[valid_mask],
            model.predict(x.loc[valid_mask]),
            target_mode,
        )
        df_pred = y.loc[valid_mask].copy()
        df_pred["pred_finish_position"] = pred
        df_pred["train_until"] = fold["train_until"]
        df_pred["valid_season"] = fold["valid_season"]
        df_pred["decay"] = decay
        fold_metrics = calcular_metricas(df_pred)
        fold_metrics.update(
            {
                "train_until": fold["train_until"],
                "valid_season": fold["valid_season"],
                "decay": decay,
            }
        )
        predicoes.append(df_pred)
        metricas.append(fold_metrics)
    return pd.concat(predicoes, ignore_index=True), pd.DataFrame(metricas)


def tunar_regressor(
    rows: list[dict],
    x: pd.DataFrame,
    y: pd.DataFrame,
    grupo: str,
    experimento: str,
    modelo: str,
    objective: str,
    target_mode: str = "finish",
    decay: float = DEFAULT_DECAY,
    score_profile: str = "atual",
    trials: int = DEFAULT_TRIALS,
) -> pd.DataFrame | None:
    sugerir = sugerir_lgb if modelo == "LightGBM" else sugerir_xgb
    weights = SCORE_PROFILES[score_profile]

    def opt_objective(trial: optuna.Trial) -> float:
        params = sugerir(trial)
        _, met = avaliar_regressor(
            x,
            y,
            lambda: criar_regressor(modelo, params, objective),
            FOLDS_TUNING,
            decay,
            target_mode,
        )
        return score_metricas(met, weights)

    # cria o estudo e otimiza com TPE
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    start = time.perf_counter()
    try:
        study.optimize(opt_objective, n_trials=trials, show_progress_bar=False)
        elapsed = time.perf_counter() - start

        # pega os melhores params e avalia nos folds de avaliação final
        best_params = dict(study.best_params)
        pred, met = avaliar_regressor(
            x,
            y,
            lambda: criar_regressor(modelo, best_params, objective),
            FOLDS_AVALIACAO,
            decay,
            target_mode,
        )
    except Exception as exc:
        rows.append(
            {
                "grupo": grupo,
                "experimento": experimento,
                "modelo": modelo,
                "erro": repr(exc),
            }
        )
        return None

    # salva predições, métricas e params do melhor trial
    safe_name = f"{experimento}_{modelo}".lower().replace(":", "_").replace("/", "_")
    pred.to_csv(ABLATION_DIR / f"predicoes_{safe_name}.csv", index=False, encoding="utf-8-sig")
    met.to_csv(ABLATION_DIR / f"metricas_{safe_name}.csv", index=False, encoding="utf-8-sig")
    (ABLATION_DIR / f"params_{safe_name}.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rows.append(
        resumo(
            grupo,
            experimento,
            modelo,
            met,
            decay=decay,
            objective=objective,
            target_mode=target_mode,
            score_profile=score_profile,
            score_tuning=float(study.best_value),
            score_perfil=score_metricas(met, weights),
            tempo_tuning_segundos=elapsed,
        )
    )
    return pred


def rodar_feature_screen_completo(rows: list[dict], full, y, features, lgb_params, xgb_params) -> None:
    # primeiro roda a ablação básica de features
    rodar_feature_ablations(rows, full, y, features, lgb_params, xgb_params)
    # aqui testa adicionar features candidatas que podem ou não estar no dataset
    for feature in ["recent_form_3", "weather_impact_observed", "safety_car_flag"]:
        if feature not in full.columns:
            continue
        from estudos_ablacao_modelos import avaliar_experimento_regressao

        avaliar_experimento_regressao(
            rows,
            nome=f"add_{feature}",
            grupo="features_screen",
            x=preparar_x(full, features + [feature]),
            y=y,
            lgb_params=lgb_params,
            xgb_params=xgb_params,
        )


def rodar_retunings(rows: list[dict], x: pd.DataFrame, y: pd.DataFrame) -> None:
    # retuna decay com Optuna - diferente da ablação sem retuning
    for decay in [0.95, 0.96, 0.97, 0.98, 0.99, 1.00]:
        for modelo, objective in [("LightGBM", "regression"), ("XGBoost", "reg:squarederror")]:
            tunar_regressor(
                rows,
                x,
                y,
                grupo="decay_retuned",
                experimento=f"decay_{decay:.2f}_retuned",
                modelo=modelo,
                objective=objective,
                decay=decay,
            )

    # retuna funções de loss alternativas
    for objective in ["regression_l1", "huber", "fair"]:
        tunar_regressor(rows, x, y, "loss_retuned", f"lgb_objective_{objective}_retuned", "LightGBM", objective)
    for objective in ["reg:absoluteerror", "reg:pseudohubererror"]:
        tunar_regressor(rows, x, y, "loss_retuned", f"xgb_objective_{objective}_retuned", "XGBoost", objective)

    # experimentos de target transformado foram desativados
    rows.append(
        {
            "grupo": "target_retuned_completo",
            "experimento": "target_transformations_desativadas",
            "modelo": "nao_executado",
            "erro": (
                "Transformacoes de target desativadas na versao final; "
                "target oficial fixo em finish_position."
            ),
        }
    )

    # testa se mudar os pesos do score muda qual modelo ganha
    for profile in SCORE_PROFILES:
        for modelo, objective in [("LightGBM", "regression"), ("XGBoost", "reg:squarederror")]:
            tunar_regressor(
                rows,
                x,
                y,
                grupo="score_weights_retuned",
                experimento=f"score_{profile}_retuned",
                modelo=modelo,
                objective=objective,
                score_profile=profile,
            )


def metricas_por_fold(df_pred: pd.DataFrame) -> pd.DataFrame:
    # quebra as predições por temporada de validação e calcula métricas em cada uma
    rows = []
    for valid_season, group in df_pred.groupby("valid_season"):
        met = calcular_metricas(group)
        met["valid_season"] = valid_season
        rows.append(met)
    return pd.DataFrame(rows)


def optimized_ensembles(rows: list[dict]) -> None:
    # carrega predicoes dos modelos base - pula se algum nao existir
    arquivos = {
        "ridge": REPORTS_DIR / "predicoes_walk_forward_ridge_baseline.csv",
        "lgb": REPORTS_DIR / "predicoes_walk_forward_lightgbm_tuned.csv",
        "xgb": REPORTS_DIR / "predicoes_walk_forward_xgboost_tuned.csv",
        "rf": REPORTS_DIR / "predicoes_walk_forward_randomforest_tuned.csv",
    }
    available = {name: pd.read_csv(path) for name, path in arquivos.items() if path.exists()}
    if len(available) < 2:
        return
    base = next(iter(available.values())).copy()
    names = list(available)
    pred_matrix = {name: df["pred_finish_position"].to_numpy() for name, df in available.items()}

    candidates = []
    unidades = range(11)

    # Grade discreta restrita: pares completos e trios estrategicos. A grade
    # exaustiva com todos os modelos cresce rapido e nao agrega comparacao util.
    combinacoes = []
    for par in itertools.combinations(range(len(names)), 2):
        for a in unidades:
            b = 10 - a
            if a == 0 or b == 0:
                continue
            weights = np.zeros(len(names), dtype=float)
            weights[par[0]] = a / 10.0
            weights[par[1]] = b / 10.0
            combinacoes.append(weights)

    # trios que fazem sentido misturar dado o que ja sabemos
    trios_preferidos = [
        ("ridge", "lgb", "xgb"),
        ("lgb", "xgb", "rf"),
    ]
    index = {name: i for i, name in enumerate(names)}
    for trio in trios_preferidos:
        if not all(name in index for name in trio):
            continue
        idxs = [index[name] for name in trio]
        for a in unidades:
            for b in unidades:
                c = 10 - a - b
                if c <= 0 or a <= 0 or b <= 0:
                    continue
                weights = np.zeros(len(names), dtype=float)
                weights[idxs[0]] = a / 10.0
                weights[idxs[1]] = b / 10.0
                weights[idxs[2]] = c / 10.0
                combinacoes.append(weights)

    # avalia cada combinação de pesos e guarda o score pra comparar
    for weights in combinacoes:
        values = sum(weights[i] * pred_matrix[names[i]] for i in range(len(names)))
        df_pred = base.copy()
        df_pred["pred_finish_position"] = values
        # só usa 2023 e 2024 pra tunar - 2025 fica pra avaliação final
        met_tuning = metricas_por_fold(df_pred[df_pred["valid_season"].isin([2023, 2024])])
        candidates.append((score_metricas(met_tuning), weights, df_pred))

    candidates.sort(key=lambda item: item[0], reverse=True)
    # pega os top 10 e avalia nos folds completos
    for rank, (score_tuning, weights, df_pred) in enumerate(candidates[:10], start=1):
        met = metricas_por_fold(df_pred)
        weight_desc = ";".join(f"{name}={weights[i]:.1f}" for i, name in enumerate(names) if weights[i] > 0)
        rows.append(
            resumo(
                "ensemble_otimizado",
                f"ensemble_grid_rank_{rank}",
                "ensemble",
                met,
                score_tuning=score_tuning,
                pesos=weight_desc,
            )
        )
    # salva as predições do melhor ensemble encontrado
    best_pred = candidates[0][2]
    best_pred.to_csv(ABLATION_DIR / "predicoes_ensemble_otimizado_melhor.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    x, y, full, features = carregar_dados()
    lgb_params = carregar_params(
        LGB_PARAMS,
        {"score_composto_tuning", "tempo_tuning_segundos", "lightgbm_version"},
    )
    xgb_params = carregar_params(
        XGB_PARAMS,
        {"score_composto_tuning", "tempo_tuning_segundos"},
    )

    rows: list[dict] = []
    rodar_feature_screen_completo(rows, full, y, features, lgb_params, xgb_params)
    rodar_retunings(rows, x, y)
    optimized_ensembles(rows)

    df = pd.DataFrame(rows)
    df = df.sort_values(["score_composto", "rmse_medio"], ascending=[False, True], na_position="last")
    df.to_csv(ABLATION_DIR / "resultados_estudos_ablacao_completo.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
