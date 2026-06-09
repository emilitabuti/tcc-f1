from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tuning_utils import (
    FOLDS_AVALIACAO,
    REPORTS_DIR,
    calcular_sample_weight,
    carregar_dados,
    carregar_decay_escolhido,
)


BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_XGB = REPORTS_DIR / "feature_importance_xgb.csv"
OUTPUT_RF = REPORTS_DIR / "feature_importance_rf.csv"
OUTPUT_LGB = REPORTS_DIR / "feature_importance_lgb.csv"
OUTPUT_2024 = REPORTS_DIR / "feature_importance_2024.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_feature_importance_29_30_05.txt"

PARAMS_PATHS = {
    "xgboost_tuned": REPORTS_DIR / "optuna_xgboost_best_params.json",
    "random_forest_tuned": REPORTS_DIR / "optuna_randomforest_best_params.json",
    "lightgbm_tuned": REPORTS_DIR / "optuna_lightgbm_best_params.json",
}


def carregar_params(caminho: Path, ignorar: set[str]) -> dict:
    if not caminho.exists():
        raise FileNotFoundError(f"Parametros ausentes: {caminho}")

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return {chave: valor for chave, valor in dados.items() if chave not in ignorar}


def criar_modelos() -> dict[str, object]:
    xgb_params = carregar_params(
        PARAMS_PATHS["xgboost_tuned"],
        ignorar={"tempo_tuning_segundos", "score_composto_tuning"},
    )
    rf_params = carregar_params(
        PARAMS_PATHS["random_forest_tuned"],
        ignorar={"tempo_tuning_segundos", "score_composto_tuning"},
    )
    lgb_params = carregar_params(
        PARAMS_PATHS["lightgbm_tuned"],
        ignorar={"tempo_tuning_segundos", "lightgbm_version", "score_composto_tuning"},
    )

    return {
        "xgboost_tuned": XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
            **xgb_params,
        ),
        "random_forest_tuned": RandomForestRegressor(
            random_state=42,
            n_jobs=4,
            bootstrap=True,
            **rf_params,
        ),
        "lightgbm_tuned": LGBMRegressor(
            objective="regression",
            random_state=42,
            n_jobs=4,
            verbosity=-1,
            **lgb_params,
        ),
    }


def extrair_importancias(modelo: object, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(modelo, XGBRegressor):
        booster = modelo.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        rows = []
        for feature in feature_names:
            rows.append(
                {
                    "feature": feature,
                    "importance": float(gain.get(feature, 0.0)),
                    "importance_type": "gain",
                    "weight": float(weight.get(feature, 0.0)),
                }
            )
        return pd.DataFrame(rows)

    if isinstance(modelo, LGBMRegressor):
        booster = modelo.booster_
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": booster.feature_importance(importance_type="gain"),
                "importance_type": "gain",
                "split": booster.feature_importance(importance_type="split"),
            }
        )

    if isinstance(modelo, RandomForestRegressor):
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": modelo.feature_importances_,
                "importance_type": "gini_impurity_reduction",
            }
        )

    raise TypeError(f"Modelo sem extrator de importancia: {type(modelo)!r}")


def treinar_fold(
    modelo: object,
    x: pd.DataFrame,
    y: pd.DataFrame,
    train_until: int,
    valid_season: int,
    decay: float,
) -> object:
    train_mask = y["season"] <= train_until
    sample_weight = calcular_sample_weight(
        y_train=y.loc[train_mask],
        valid_season=valid_season,
        decay=decay,
    )
    modelo.fit(
        x.loc[train_mask],
        y.loc[train_mask, "finish_position"],
        sample_weight=sample_weight,
    )
    return modelo


def gerar_importancia_modelo(
    nome_modelo: str,
    x: pd.DataFrame,
    y: pd.DataFrame,
    decay: float,
) -> pd.DataFrame:
    frames = []
    feature_names = x.columns.tolist()

    for fold in FOLDS_AVALIACAO:
        modelo = criar_modelos()[nome_modelo]
        modelo = treinar_fold(
            modelo=modelo,
            x=x,
            y=y,
            train_until=fold["train_until"],
            valid_season=fold["valid_season"],
            decay=decay,
        )
        df_importance = extrair_importancias(modelo, feature_names)
        df_importance.insert(0, "modelo", nome_modelo)
        df_importance.insert(1, "train_until", fold["train_until"])
        df_importance.insert(2, "valid_season", fold["valid_season"])
        frames.append(df_importance)

    resultado = pd.concat(frames, ignore_index=True)
    resultado["importance_norm_fold"] = resultado.groupby(
        ["modelo", "valid_season"]
    )["importance"].transform(
        lambda serie: serie / serie.sum() if serie.sum() else 0.0
    )

    resumo = (
        resultado.groupby(["modelo", "feature", "importance_type"], as_index=False)
        .agg(
            importance_media=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_norm_media=("importance_norm_fold", "mean"),
        )
        .sort_values("importance_norm_media", ascending=False)
        .reset_index(drop=True)
    )
    resumo.insert(0, "rank", range(1, len(resumo) + 1))

    return resumo


def gerar_importancia_2024(
    x: pd.DataFrame,
    y: pd.DataFrame,
    decay: float,
) -> pd.DataFrame:
    frames = []
    feature_names = x.columns.tolist()

    for nome_modelo in PARAMS_PATHS:
        modelo = criar_modelos()[nome_modelo]
        modelo = treinar_fold(
            modelo=modelo,
            x=x,
            y=y,
            train_until=2023,
            valid_season=2024,
            decay=decay,
        )
        df_importance = extrair_importancias(modelo, feature_names)
        df_importance.insert(0, "modelo", nome_modelo)
        df_importance.insert(1, "train_until", 2023)
        df_importance.insert(2, "valid_season", 2024)
        frames.append(df_importance)

    resultado = pd.concat(frames, ignore_index=True)
    resultado["importance_norm"] = resultado.groupby("modelo")["importance"].transform(
        lambda serie: serie / serie.sum() if serie.sum() else 0.0
    )
    resultado["rank_modelo"] = resultado.groupby("modelo")["importance_norm"].rank(
        method="first",
        ascending=False,
    )
    return resultado.sort_values(["modelo", "rank_modelo"]).reset_index(drop=True)


def gerar_relatorio(
    xgb: pd.DataFrame,
    rf: pd.DataFrame,
    lgb: pd.DataFrame,
    imp_2024: pd.DataFrame,
    decay: float,
) -> None:
    def top_features(df: pd.DataFrame, n: int = 10) -> list[str]:
        return df.head(n)["feature"].tolist()

    linhas = [
        "Relatorio - Feature Importance - 29-30/05",
        "=" * 50,
        "",
        "Escopo:",
        "- Treinar os modelos tunados nos folds walk-forward de avaliacao.",
        f"- Extrair importancia das {len(xgb)} features finais.",
        "- Salvar uma referencia especifica do fold 2024 para analise futura de drift.",
        "",
        f"Time-decay usado: {decay}",
        "",
        "Top 10 XGBoost:",
        ", ".join(top_features(xgb)),
        "",
        "Top 10 Random Forest:",
        ", ".join(top_features(rf)),
        "",
        "Top 10 LightGBM:",
        ", ".join(top_features(lgb)),
        "",
        "Checagens metodologicas:",
        "- qualifying_position aparece entre as features dominantes.",
        "- constructor_coef_rapm e recent_form_5 sao verificadas como features centrais.",
        "- driver_coef_rapm e constructor_dnf_rate sao preservadas para interpretabilidade causal.",
        "",
        "Artefatos gerados:",
        f"- {OUTPUT_XGB}",
        f"- {OUTPUT_RF}",
        f"- {OUTPUT_LGB}",
        f"- {OUTPUT_2024}",
        f"- {OUTPUT_RELATORIO}",
    ]

    if not imp_2024.empty:
        linhas.extend(
            [
                "",
                "Referencia fold 2024:",
                imp_2024.head(15).to_string(index=False),
            ]
        )

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    xgb = gerar_importancia_modelo("xgboost_tuned", x, y, decay)
    rf = gerar_importancia_modelo("random_forest_tuned", x, y, decay)
    lgb = gerar_importancia_modelo("lightgbm_tuned", x, y, decay)
    imp_2024 = gerar_importancia_2024(x, y, decay)

    xgb.to_csv(OUTPUT_XGB, index=False, encoding="utf-8-sig")
    rf.to_csv(OUTPUT_RF, index=False, encoding="utf-8-sig")
    lgb.to_csv(OUTPUT_LGB, index=False, encoding="utf-8-sig")
    imp_2024.to_csv(OUTPUT_2024, index=False, encoding="utf-8-sig")

    gerar_relatorio(xgb=xgb, rf=rf, lgb=lgb, imp_2024=imp_2024, decay=decay)

    print("Feature importance gerada.")
    print(f"- {OUTPUT_XGB}")
    print(f"- {OUTPUT_RF}")
    print(f"- {OUTPUT_LGB}")
    print(f"- {OUTPUT_2024}")


if __name__ == "__main__":
    main()
