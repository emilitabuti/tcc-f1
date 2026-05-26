from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "feature_selection"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"

OUTPUT_RANKING = MODELS_DIR / "rfe_xgboost_ranking.csv"
OUTPUT_SUBSETS = MODELS_DIR / "rfe_xgboost_subsets.csv"
OUTPUT_REPORT = MODELS_DIR / "relatorio_rfe_xgboost.txt"
OUTPUT_MANIFEST = MODELS_DIR / "manifest_rfe_xgboost.json"


def load_data():
    if not INPUT_X.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_X}")
    if not INPUT_Y.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_Y}")

    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

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


def evaluate_subsets(x, y, ranking):
    train_mask = y["season"] <= 2024
    valid_mask = y["season"] == 2025

    if valid_mask.sum() == 0:
        raise RuntimeError("Nao ha temporada 2025 para validacao temporal.")

    ordered_features = ranking["feature"].tolist()
    rows = []

    for k in range(12, len(ordered_features) + 1):
        features = ordered_features[:k]
        model = build_model(seed=100 + k)
        model.fit(
            x.loc[train_mask, features],
            y.loc[train_mask, "finish_position"],
        )
        pred = model.predict(x.loc[valid_mask, features])
        mae = mean_absolute_error(y.loc[valid_mask, "finish_position"], pred)
        rows.append(
            {
                "n_features": k,
                "mae_2025": float(mae),
                "features": ",".join(features),
            }
        )

    subsets = pd.DataFrame(rows).sort_values(["mae_2025", "n_features"]).reset_index(drop=True)
    return subsets


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_data()
    train_mask = y["season"] <= 2024

    model = build_model(seed=42)
    model.fit(x.loc[train_mask], y.loc[train_mask, "finish_position"])

    ranking = rank_features(model, x.columns.tolist())
    subsets = evaluate_subsets(x, y, ranking)

    best = subsets.iloc[0].to_dict()
    selected = best["features"].split(",")

    ranking.to_csv(OUTPUT_RANKING, index=False)
    subsets.to_csv(OUTPUT_SUBSETS, index=False)

    lines = [
        "Relatorio RFE - XGBoost",
        "=" * 45,
        "",
        f"Entrada X: {INPUT_X}",
        f"Entrada y: {INPUT_Y}",
        f"Treino: temporadas <= 2024 ({int(train_mask.sum())} linhas)",
        f"Validacao: temporada 2025 ({int((y['season'] == 2025).sum())} linhas)",
        "",
        "Melhor subconjunto por MAE em 2025:",
        f"- n_features: {int(best['n_features'])}",
        f"- mae_2025: {best['mae_2025']:.6f}",
        f"- features: {', '.join(selected)}",
        "",
        "Ranking por gain:",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"{int(row['rank']):02d}. {row['feature']} "
            f"(gain={row['gain']:.6f}, weight={row['weight']:.0f})"
        )

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    OUTPUT_MANIFEST.write_text(
        json.dumps(
            {
                "method": "XGBRegressor gain ranking + temporal subset ablation",
                "train": "season <= 2024",
                "validation": "season == 2025",
                "candidate_features": x.columns.tolist(),
                "best_n_features": int(best["n_features"]),
                "best_mae_2025": float(best["mae_2025"]),
                "selected_features": selected,
                "ranking_csv": OUTPUT_RANKING.relative_to(BASE_DIR).as_posix(),
                "subsets_csv": OUTPUT_SUBSETS.relative_to(BASE_DIR).as_posix(),
                "report": OUTPUT_REPORT.relative_to(BASE_DIR).as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("RFE XGBoost concluido com sucesso.")
    print(f"Melhor n_features: {int(best['n_features'])}")
    print(f"MAE 2025: {best['mae_2025']:.6f}")
    print(f"Relatorio: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
