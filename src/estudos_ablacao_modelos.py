from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from metricas import calcular_metricas
from tuning_utils import (
    BASE_DIR,
    FOLDS_AVALIACAO,
    REPORTS_DIR,
    calcular_sample_weight,
    score_composto_metricas,
)


PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURE_SELECTION_DIR = BASE_DIR / "models" / "feature_selection"
ABLATION_DIR = BASE_DIR / "reports" / "ablacao"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
INPUT_FULL = PROCESSED_DIR / "dataset_features_final_2018_2025_sem_nan.csv"
INPUT_FEATURES = FEATURE_SELECTION_DIR / "features_modelagem_2018_2025.json"

LGB_PARAMS = REPORTS_DIR / "optuna_lightgbm_best_params.json"
XGB_PARAMS = REPORTS_DIR / "optuna_xgboost_best_params.json"

DECAY_OFICIAL = 0.99


def carregar_params(caminho: Path, ignorar: set[str]) -> dict:
    params = json.loads(caminho.read_text(encoding="utf-8"))
    return {k: v for k, v in params.items() if k not in ignorar}


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)
    full = pd.read_csv(INPUT_FULL)
    payload = json.loads(INPUT_FEATURES.read_text(encoding="utf-8"))
    features = payload["features"]

    if len(x) != len(y) or len(full) != len(y):
        raise RuntimeError("Datasets com tamanhos divergentes.")

    return x, y, full, features


def preparar_x(full: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    faltantes = [feature for feature in features if feature not in full.columns]
    if faltantes:
        raise ValueError(f"Features ausentes no dataset completo: {faltantes}")
    return full[features].copy()


def criar_lgb(params: dict, objective: str = "regression") -> LGBMRegressor:
    return LGBMRegressor(
        objective=objective,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        **params,
    )


def criar_xgb(params: dict, objective: str = "reg:squarederror") -> XGBRegressor:
    return XGBRegressor(
        objective=objective,
        random_state=42,
        n_jobs=4,
        **params,
    )


def avaliar_regressao(
    x: pd.DataFrame,
    y: pd.DataFrame,
    criar_modelo: Callable[[], object],
    decay: float = DECAY_OFICIAL,
    target_mode: str = "finish",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metricas = []
    predicoes = []

    for fold in FOLDS_AVALIACAO:
        train_mask = y["season"] <= fold["train_until"]
        valid_mask = y["season"] == fold["valid_season"]
        sample_weight = calcular_sample_weight(
            y_train=y.loc[train_mask],
            valid_season=fold["valid_season"],
            decay=decay,
        )

        if target_mode == "finish":
            y_train_target = y.loc[train_mask, "finish_position"]
        elif target_mode == "delta_grid":
            if "qualifying_position" not in x.columns:
                raise ValueError("target_mode=delta_grid exige qualifying_position em X.")
            y_train_target = (
                y.loc[train_mask, "finish_position"]
                - x.loc[train_mask, "qualifying_position"]
            )
        else:
            raise ValueError(f"target_mode desconhecido: {target_mode}")

        model = criar_modelo()
        model.fit(x.loc[train_mask], y_train_target, sample_weight=sample_weight)
        pred = model.predict(x.loc[valid_mask])

        if target_mode == "delta_grid":
            pred = x.loc[valid_mask, "qualifying_position"].to_numpy() + pred

        df_pred = y.loc[valid_mask].copy()
        df_pred["pred_finish_position"] = pred
        df_pred["train_until"] = fold["train_until"]
        df_pred["valid_season"] = fold["valid_season"]
        df_pred["decay"] = decay

        fold_metricas = calcular_metricas(df_pred)
        fold_metricas.update(
            {
                "train_until": fold["train_until"],
                "valid_season": fold["valid_season"],
                "decay": decay,
            }
        )
        metricas.append(fold_metricas)
        predicoes.append(df_pred)

    return pd.concat(predicoes, ignore_index=True), pd.DataFrame(metricas)


def resumir_metricas(nome: str, grupo: str, modelo: str, metricas: pd.DataFrame) -> dict:
    return {
        "grupo": grupo,
        "experimento": nome,
        "modelo": modelo,
        "mae_medio": metricas["mae"].mean(),
        "rmse_medio": metricas["rmse"].mean(),
        "r2_medio": metricas["r2"].mean(),
        "kendall_tau_medio": metricas["kendall_tau"].mean(),
        "top3_accuracy_medio": metricas["top3_accuracy"].mean(),
        "score_composto": score_composto_metricas(metricas),
    }


def avaliar_experimento_regressao(
    rows: list[dict],
    nome: str,
    grupo: str,
    x: pd.DataFrame,
    y: pd.DataFrame,
    lgb_params: dict,
    xgb_params: dict,
    decay: float = DECAY_OFICIAL,
    lgb_objective: str = "regression",
    xgb_objective: str = "reg:squarederror",
    target_mode: str = "finish",
) -> None:
    _, met_lgb = avaliar_regressao(
        x=x,
        y=y,
        criar_modelo=lambda: criar_lgb(lgb_params, objective=lgb_objective),
        decay=decay,
        target_mode=target_mode,
    )
    rows.append(resumir_metricas(nome, grupo, "LightGBM", met_lgb))

    _, met_xgb = avaliar_regressao(
        x=x,
        y=y,
        criar_modelo=lambda: criar_xgb(xgb_params, objective=xgb_objective),
        decay=decay,
        target_mode=target_mode,
    )
    rows.append(resumir_metricas(nome, grupo, "XGBoost", met_xgb))


def rodar_feature_ablations(
    rows: list[dict],
    full: pd.DataFrame,
    y: pd.DataFrame,
    features: list[str],
    lgb_params: dict,
    xgb_params: dict,
) -> None:
    low_rank = [
        "season_factor",
        "tire_compound_start",
        "grid_penalty",
        "altitude_m",
        "avg_pit_stops_circuit",
    ]
    experiments: dict[str, list[str]] = {"baseline_13": features}

    for feature in low_rank:
        experiments[f"remove_{feature}"] = [f for f in features if f != feature]

    experiments["top10_importance_media"] = [
        "qualifying_position",
        "constructor_coef_rapm",
        "recent_form_5",
        "driver_constructor_synergy",
        "constructor_wins_total",
        "driver_coef_rapm",
        "track_complexity",
        "constructor_dnf_rate",
        "avg_pit_stops_circuit",
        "season_factor",
    ]
    experiments["top8_importance_media"] = experiments["top10_importance_media"][:8]
    experiments["add_incident_rate_hist_norm"] = features + ["incident_rate_hist_norm"]
    experiments["add_driver_dnf_rate"] = features + ["driver_dnf_rate"]

    for nome, exp_features in experiments.items():
        x_exp = preparar_x(full, exp_features)
        avaliar_experimento_regressao(
            rows,
            nome=nome,
            grupo="features",
            x=x_exp,
            y=y,
            lgb_params=lgb_params,
            xgb_params=xgb_params,
        )


def rodar_decay_ablations(
    rows: list[dict],
    x: pd.DataFrame,
    y: pd.DataFrame,
    lgb_params: dict,
    xgb_params: dict,
) -> None:
    for decay in [0.95, 0.97, 0.98, 0.99, 1.00]:
        avaliar_experimento_regressao(
            rows,
            nome=f"decay_{decay:.2f}",
            grupo="decay_sem_retuning",
            x=x,
            y=y,
            lgb_params=lgb_params,
            xgb_params=xgb_params,
            decay=decay,
        )


def rodar_loss_ablations(
    rows: list[dict],
    x: pd.DataFrame,
    y: pd.DataFrame,
    lgb_params: dict,
    xgb_params: dict,
) -> None:
    for objective in ["regression_l1", "huber", "fair"]:
        _, met = avaliar_regressao(
            x=x,
            y=y,
            criar_modelo=lambda objective=objective: criar_lgb(lgb_params, objective=objective),
            decay=DECAY_OFICIAL,
        )
        rows.append(resumir_metricas(f"lgb_objective_{objective}", "loss", "LightGBM", met))

    for objective in ["reg:absoluteerror", "reg:pseudohubererror"]:
        _, met = avaliar_regressao(
            x=x,
            y=y,
            criar_modelo=lambda objective=objective: criar_xgb(xgb_params, objective=objective),
            decay=DECAY_OFICIAL,
        )
        rows.append(resumir_metricas(f"xgb_objective_{objective}", "loss", "XGBoost", met))


def rodar_target_delta(
    rows: list[dict],
    x: pd.DataFrame,
    y: pd.DataFrame,
    lgb_params: dict,
    xgb_params: dict,
) -> None:
    avaliar_experimento_regressao(
        rows,
        nome="target_delta_grid_to_finish",
        grupo="target",
        x=x,
        y=y,
        lgb_params=lgb_params,
        xgb_params=xgb_params,
        target_mode="delta_grid",
    )


def rodar_ensembles(rows: list[dict], y: pd.DataFrame) -> None:
    arquivos = {
        "ridge": REPORTS_DIR / "predicoes_walk_forward_ridge_baseline.csv",
        "lgb": REPORTS_DIR / "predicoes_walk_forward_lightgbm_tuned.csv",
        "xgb": REPORTS_DIR / "predicoes_walk_forward_xgboost_tuned.csv",
        "rf": REPORTS_DIR / "predicoes_walk_forward_randomforest_tuned.csv",
    }
    pred = {}
    for nome, caminho in arquivos.items():
        df = pd.read_csv(caminho)
        pred[nome] = df["pred_finish_position"].to_numpy()

    base = pd.read_csv(arquivos["lgb"])
    ensembles = {
        "ridge_70_lgb_30": 0.70 * pred["ridge"] + 0.30 * pred["lgb"],
        "ridge_50_lgb_50": 0.50 * pred["ridge"] + 0.50 * pred["lgb"],
        "ridge_70_xgb_30": 0.70 * pred["ridge"] + 0.30 * pred["xgb"],
        "lgb_70_xgb_30": 0.70 * pred["lgb"] + 0.30 * pred["xgb"],
        "media_lgb_xgb": 0.50 * pred["lgb"] + 0.50 * pred["xgb"],
        "media_arvores": (pred["lgb"] + pred["xgb"] + pred["rf"]) / 3,
    }

    for nome, values in ensembles.items():
        df_pred = base.copy()
        df_pred["pred_finish_position"] = values
        metricas_fold = []
        for valid_season, grupo in df_pred.groupby("valid_season"):
            fold_metricas = calcular_metricas(grupo)
            fold_metricas["valid_season"] = valid_season
            metricas_fold.append(fold_metricas)
        met = pd.DataFrame(metricas_fold)
        rows.append(resumir_metricas(nome, "ensemble", "ensemble", met))


def rodar_podio_classificador(
    rows: list[dict],
    x: pd.DataFrame,
    y: pd.DataFrame,
    lgb_params: dict,
    xgb_params: dict,
) -> None:
    y_class = y.copy()
    y_class["is_podium"] = 0
    for _, grupo in y_class.groupby(["season", "round"]):
        idx = grupo.sort_values("finish_position").head(3).index
        y_class.loc[idx, "is_podium"] = 1

    model_builders = {
        "LightGBMClassifier": lambda: LGBMClassifier(
            random_state=42,
            n_jobs=4,
            verbosity=-1,
            n_estimators=int(lgb_params["n_estimators"]),
            max_depth=int(lgb_params["max_depth"]),
            num_leaves=int(lgb_params["num_leaves"]),
            learning_rate=float(lgb_params["learning_rate"]),
            subsample=float(lgb_params["subsample"]),
            colsample_bytree=float(lgb_params["colsample_bytree"]),
            reg_alpha=float(lgb_params["reg_alpha"]),
            reg_lambda=float(lgb_params["reg_lambda"]),
            min_child_samples=int(lgb_params["min_child_samples"]),
            class_weight="balanced",
        ),
        "XGBClassifier": lambda: XGBClassifier(
            random_state=42,
            n_jobs=4,
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
            reg_alpha=float(xgb_params["reg_alpha"]),
            reg_lambda=float(xgb_params["reg_lambda"]),
        ),
    }

    for model_name, builder in model_builders.items():
        fold_scores = []
        for fold in FOLDS_AVALIACAO:
            train_mask = y["season"] <= fold["train_until"]
            valid_mask = y["season"] == fold["valid_season"]
            sample_weight = calcular_sample_weight(
                y_train=y.loc[train_mask],
                valid_season=fold["valid_season"],
                decay=DECAY_OFICIAL,
            )
            model = builder()
            model.fit(
                x.loc[train_mask],
                y_class.loc[train_mask, "is_podium"],
                sample_weight=sample_weight,
            )
            proba = model.predict_proba(x.loc[valid_mask])[:, 1]
            df_pred = y.loc[valid_mask].copy()
            df_pred["podium_score"] = proba

            acertos = []
            for _, grupo in df_pred.groupby(["season", "round"]):
                real = set(grupo.sort_values("finish_position").head(3)["driver_id"])
                pred = set(grupo.sort_values("podium_score", ascending=False).head(3)["driver_id"])
                acertos.append(int(real == pred))

            fold_scores.append(float(np.mean(acertos)))

        rows.append(
            {
                "grupo": "podio_classificador",
                "experimento": model_name,
                "modelo": model_name,
                "mae_medio": np.nan,
                "rmse_medio": np.nan,
                "r2_medio": np.nan,
                "kendall_tau_medio": np.nan,
                "top3_accuracy_medio": float(np.mean(fold_scores)),
                "score_composto": np.nan,
            }
        )


def gerar_relatorio(df: pd.DataFrame) -> None:
    best_score = df.dropna(subset=["score_composto"]).sort_values("score_composto", ascending=False).head(12)
    best_rmse = df.dropna(subset=["rmse_medio"]).sort_values("rmse_medio").head(12)
    best_top3 = df.sort_values("top3_accuracy_medio", ascending=False).head(12)

    linhas = [
        "# Relatorio de Estudos de Ablacao",
        "",
        "Baseline oficial atual: LightGBM e XGBoost com 13 features e decay=0.99.",
        "",
        "## Melhores por score composto",
        "",
        best_score.to_markdown(index=False),
        "",
        "## Melhores por RMSE",
        "",
        best_rmse.to_markdown(index=False),
        "",
        "## Melhores por top-3",
        "",
        best_top3.to_markdown(index=False),
        "",
        "## Observacoes",
        "",
        "- Experimentos de features, decay, loss e target usam os hiperparametros atuais, sem retuning completo.",
        "- Experimentos promissores devem ser retunados antes de virar decisao oficial.",
        "- O classificador de podium avalia apenas top-3, pois nao produz predicao numerica de posicao.",
    ]
    (ABLATION_DIR / "relatorio_estudos_ablacao.md").write_text("\n".join(linhas), encoding="utf-8")


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
    rodar_feature_ablations(rows, full, y, features, lgb_params, xgb_params)
    rodar_decay_ablations(rows, x, y, lgb_params, xgb_params)
    rodar_loss_ablations(rows, x, y, lgb_params, xgb_params)
    rodar_target_delta(rows, x, y, lgb_params, xgb_params)
    rodar_ensembles(rows, y)
    rodar_podio_classificador(rows, x, y, lgb_params, xgb_params)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["score_composto", "rmse_medio", "top3_accuracy_medio"],
        ascending=[False, True, False],
        na_position="last",
    )
    df.to_csv(ABLATION_DIR / "resultados_estudos_ablacao.csv", index=False, encoding="utf-8-sig")
    gerar_relatorio(df)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
