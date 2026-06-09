from pathlib import Path
import json

import pandas as pd
from xgboost import XGBRegressor

from metricas import calcular_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "feature_selection"

INPUT_DATASET = PROCESSED_DIR / "dataset_features_final_2018_2025_sem_nan.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"

OUTPUT_RANKING = MODELS_DIR / "rfe_xgboost_ranking.csv"
OUTPUT_SUBSETS = MODELS_DIR / "rfe_xgboost_subsets.csv"
OUTPUT_PARETO = MODELS_DIR / "rfe_xgboost_pareto.csv"
OUTPUT_REPORT = MODELS_DIR / "relatorio_rfe_xgboost.txt"
OUTPUT_MANIFEST = MODELS_DIR / "manifest_rfe_xgboost.json"

METRIC_WEIGHTS = {
    "mae_score": 0.30,
    "rmse_score": 0.15,
    "r2_score": 0.20,
    "kendall_tau_score": 0.20,
    "top3_accuracy_score": 0.15,
}

FOLDS_RFE = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]

FEATURES_CANDIDATAS = [
    "qualifying_position",
    "grid_penalty",
    "recent_form_5",
    "driver_coef_rapm",
    "driver_dnf_rate",
    "constructor_coef_rapm",
    "constructor_dnf_rate",
    "constructor_wins_total",
    "driver_constructor_synergy",
    "track_complexity",
    "altitude_m",
    "tire_compound_start",
    "avg_pit_stops_circuit",
    "season_factor",
    "incident_rate_hist_norm",
]


def load_data():
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_DATASET}")
    if not INPUT_Y.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_Y}")

    df = pd.read_csv(INPUT_DATASET)
    y = pd.read_csv(INPUT_Y)
    faltantes = [col for col in FEATURES_CANDIDATAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"Features candidatas ausentes: {faltantes}")

    x = df[FEATURES_CANDIDATAS].copy()

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    return x, y


def build_model(seed=42):
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=4,
    )


def rank_features(model, features):
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")

    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "gain": float(gain.get(feature, 0.0)),
                "weight": float(weight.get(feature, 0.0)),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["gain", "weight", "feature"],
        ascending=[False, False, True],
    )
    ranking["rank"] = range(1, len(ranking) + 1)

    return ranking[["rank", "feature", "gain", "weight"]]


def rank_features_temporal(x, y):
    rankings = []

    for fold in FOLDS_RFE:
        train_until = fold["train_until"]
        valid_season = fold["valid_season"]
        train_mask = y["season"] <= train_until

        if train_mask.sum() == 0:
            raise RuntimeError(f"Fold {valid_season} sem dados de treino.")

        model = build_model(seed=42 + valid_season)
        model.fit(
            x.loc[train_mask],
            y.loc[train_mask, "finish_position"],
        )

        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")

        rows = []
        for feature in x.columns:
            rows.append(
                {
                    "feature": feature,
                    f"gain_{valid_season}": float(gain.get(feature, 0.0)),
                    f"weight_{valid_season}": float(weight.get(feature, 0.0)),
                }
            )

        fold_ranking = pd.DataFrame(rows)
        fold_ranking[f"gain_norm_{valid_season}"] = normalizar_coluna(
            fold_ranking,
            f"gain_{valid_season}",
            maior_melhor=True,
        )
        rankings.append(fold_ranking)

    ranking = rankings[0]
    for fold_ranking in rankings[1:]:
        ranking = ranking.merge(fold_ranking, on="feature", how="inner")

    gain_norm_cols = [f"gain_norm_{fold['valid_season']}" for fold in FOLDS_RFE]
    gain_cols = [f"gain_{fold['valid_season']}" for fold in FOLDS_RFE]
    weight_cols = [f"weight_{fold['valid_season']}" for fold in FOLDS_RFE]

    ranking["gain_medio_norm"] = ranking[gain_norm_cols].mean(axis=1)
    ranking["gain_medio"] = ranking[gain_cols].mean(axis=1)
    ranking["weight_medio"] = ranking[weight_cols].mean(axis=1)
    ranking = ranking.sort_values(
        ["gain_medio_norm", "gain_medio", "weight_medio", "feature"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)

    ordered_cols = [
        "rank",
        "feature",
        "gain_medio_norm",
        "gain_medio",
        "weight_medio",
    ]
    detail_cols = []
    for fold in FOLDS_RFE:
        season = fold["valid_season"]
        detail_cols.extend([
            f"gain_{season}",
            f"gain_norm_{season}",
            f"weight_{season}",
        ])

    return ranking[ordered_cols + detail_cols]


def evaluate_subsets(x, y, ranking):
    ordered_features = ranking["feature"].tolist()
    rows = []

    for k in range(12, len(ordered_features) + 1):
        features = ordered_features[:k]
        row = {"n_features": k, "features": ",".join(features)}

        fold_metricas = []
        for fold in FOLDS_RFE:
            train_until = fold["train_until"]
            valid_season = fold["valid_season"]
            train_mask = y["season"] <= train_until
            valid_mask = y["season"] == valid_season

            if valid_mask.sum() == 0:
                raise RuntimeError(f"Nao ha temporada {valid_season} para validacao temporal.")

            model = build_model(seed=100 + k + valid_season)
            model.fit(
                x.loc[train_mask, features],
                y.loc[train_mask, "finish_position"],
            )
            pred = model.predict(x.loc[valid_mask, features])
            df_pred = y.loc[valid_mask].copy()
            df_pred["pred_finish_position"] = pred
            metricas = calcular_metricas(df_pred)

            for metrica in ["mae", "rmse", "r2", "kendall_tau", "top3_accuracy"]:
                row[f"{metrica}_{valid_season}"] = float(metricas[metrica])

            fold_metricas.append(metricas)

        for metrica in ["mae", "rmse", "r2", "kendall_tau", "top3_accuracy"]:
            row[f"{metrica}_medio"] = float(
                sum(float(item[metrica]) for item in fold_metricas) / len(fold_metricas)
            )

        rows.append(row)

    subsets = pd.DataFrame(rows)
    subsets = adicionar_score_composto(subsets)
    subsets = subsets.sort_values(
        ["score_composto", "mae_medio", "n_features"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return subsets


def normalizar_coluna(df, coluna, maior_melhor):
    serie = df[coluna]
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return pd.Series(1.0, index=df.index)

    if maior_melhor:
        return (serie - minimo) / (maximo - minimo)

    return (maximo - serie) / (maximo - minimo)


def adicionar_score_composto(subsets):
    subsets = subsets.copy()
    subsets["mae_score"] = normalizar_coluna(subsets, "mae_medio", maior_melhor=False)
    subsets["rmse_score"] = normalizar_coluna(subsets, "rmse_medio", maior_melhor=False)
    subsets["r2_score"] = normalizar_coluna(subsets, "r2_medio", maior_melhor=True)
    subsets["kendall_tau_score"] = normalizar_coluna(
        subsets,
        "kendall_tau_medio",
        maior_melhor=True,
    )
    subsets["top3_accuracy_score"] = normalizar_coluna(
        subsets,
        "top3_accuracy_medio",
        maior_melhor=True,
    )

    subsets["score_composto"] = 0.0
    for coluna, peso in METRIC_WEIGHTS.items():
        subsets["score_composto"] += peso * subsets[coluna]

    return subsets


def identificar_pareto(subsets):
    metricas = [
        ("mae_medio", False),
        ("rmse_medio", False),
        ("r2_medio", True),
        ("kendall_tau_medio", True),
        ("top3_accuracy_medio", True),
    ]
    pareto = []

    for idx, row in subsets.iterrows():
        dominado = False
        for other_idx, other in subsets.iterrows():
            if idx == other_idx:
                continue

            melhor_ou_igual = []
            estritamente_melhor = []
            for coluna, maior_melhor in metricas:
                if maior_melhor:
                    melhor_ou_igual.append(other[coluna] >= row[coluna])
                    estritamente_melhor.append(other[coluna] > row[coluna])
                else:
                    melhor_ou_igual.append(other[coluna] <= row[coluna])
                    estritamente_melhor.append(other[coluna] < row[coluna])

            if all(melhor_ou_igual) and any(estritamente_melhor):
                dominado = True
                break

        if not dominado:
            pareto.append(row)

    if not pareto:
        return subsets.head(0).copy()

    return pd.DataFrame(pareto).sort_values(
        ["score_composto", "mae_medio"],
        ascending=[False, True],
    ).reset_index(drop=True)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_data()
    ranking = rank_features_temporal(x, y)
    subsets = evaluate_subsets(x, y, ranking)
    pareto = identificar_pareto(subsets)

    best = subsets.iloc[0].to_dict()
    selected = best["features"].split(",")

    ranking.to_csv(OUTPUT_RANKING, index=False)
    subsets.to_csv(OUTPUT_SUBSETS, index=False)
    pareto.to_csv(OUTPUT_PARETO, index=False)

    lines = [
        "Relatorio RFE Multi-Metrica - XGBoost",
        "=" * 45,
        "",
        f"Entrada dataset tratado: {INPUT_DATASET}",
        f"Entrada y: {INPUT_Y}",
        "Folds temporais:",
        *[
            (
                f"- treino <= {fold['train_until']} "
                f"({int((y['season'] <= fold['train_until']).sum())} linhas), "
                f"validacao = {fold['valid_season']} "
                f"({int((y['season'] == fold['valid_season']).sum())} linhas)"
            )
            for fold in FOLDS_RFE
        ],
        f"Features candidatas: {len(FEATURES_CANDIDATAS)}",
        f"Pesos do score composto: {METRIC_WEIGHTS}",
        "",
        "Melhor subconjunto por score composto medio multi-fold:",
        f"- n_features: {int(best['n_features'])}",
        f"- score_composto: {best['score_composto']:.6f}",
        f"- mae_medio: {best['mae_medio']:.6f}",
        f"- rmse_medio: {best['rmse_medio']:.6f}",
        f"- r2_medio: {best['r2_medio']:.6f}",
        f"- kendall_tau_medio: {best['kendall_tau_medio']:.6f}",
        f"- top3_accuracy_medio: {best['top3_accuracy_medio']:.6f}",
        f"- features: {', '.join(selected)}",
        "",
        "Subconjuntos Pareto-otimos:",
        pareto[[
            "n_features",
            "score_composto",
            "mae_medio",
            "rmse_medio",
            "r2_medio",
            "kendall_tau_medio",
            "top3_accuracy_medio",
        ]].to_string(index=False),
        "",
        "Ranking por gain medio normalizado:",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"{int(row['rank']):02d}. {row['feature']} "
            f"(gain_medio_norm={row['gain_medio_norm']:.6f}, "
            f"gain_medio={row['gain_medio']:.6f}, "
            f"weight_medio={row['weight_medio']:.1f})"
        )

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    OUTPUT_MANIFEST.write_text(
        json.dumps(
            {
                "method": "XGBRegressor gain ranking medio temporal + subset ablation multi-fold multi-metrica",
                "folds": FOLDS_RFE,
                "metric_weights": METRIC_WEIGHTS,
                "candidate_features": x.columns.tolist(),
                "best_n_features": int(best["n_features"]),
                "best_score_composto": float(best["score_composto"]),
                "best_mae_medio": float(best["mae_medio"]),
                "best_rmse_medio": float(best["rmse_medio"]),
                "best_r2_medio": float(best["r2_medio"]),
                "best_kendall_tau_medio": float(best["kendall_tau_medio"]),
                "best_top3_accuracy_medio": float(best["top3_accuracy_medio"]),
                "selected_features": selected,
                "ranking_csv": OUTPUT_RANKING.relative_to(BASE_DIR).as_posix(),
                "subsets_csv": OUTPUT_SUBSETS.relative_to(BASE_DIR).as_posix(),
                "pareto_csv": OUTPUT_PARETO.relative_to(BASE_DIR).as_posix(),
                "report": OUTPUT_REPORT.relative_to(BASE_DIR).as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("RFE XGBoost concluido com sucesso.")
    print(f"Melhor n_features: {int(best['n_features'])}")
    print(f"Score composto: {best['score_composto']:.6f}")
    print(f"MAE medio: {best['mae_medio']:.6f}")
    print(f"Relatorio: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
