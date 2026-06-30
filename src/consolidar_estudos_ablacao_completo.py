from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from estudos_ablacao_completo import ABLATION_DIR, REPORTS_DIR, metricas_por_fold, resumo, score_metricas


def inferir_grupo_experimento_modelo(path: Path) -> tuple[str, str, str]:
    # tira o prefixo "metricas_" do nome do arquivo pra extrair o experimento
    stem = path.stem.removeprefix("metricas_")
    if stem.endswith("_lightgbm"):
        modelo = "LightGBM"
        experimento = stem.removesuffix("_lightgbm")
    elif stem.endswith("_xgboost"):
        modelo = "XGBoost"
        experimento = stem.removesuffix("_xgboost")
    elif stem == "lightgbm_target_delta_retuned":
        return "retuned", stem, "LightGBM"
    elif stem == "xgboost_target_delta_retuned":
        return "retuned", stem, "XGBoost"
    else:
        modelo = "desconhecido"
        experimento = stem

    # infere o grupo pelo prefixo do nome do experimento
    if experimento.startswith("decay_"):
        grupo = "decay_retuned"
    elif experimento.startswith("lgb_objective_") or experimento.startswith("xgb_objective_"):
        grupo = "loss_retuned"
    elif "rank_norm_grid20" in experimento:
        grupo = "validacao_causal_target"
    elif experimento.startswith("target_"):
        grupo = "target_retuned_completo"
    elif experimento.startswith("score_"):
        grupo = "score_weights_retuned"
    else:
        grupo = "retuned"
    return grupo, experimento, modelo


def rows_de_metricas() -> list[dict]:
    rows: list[dict] = []
    # lê todos os csvs de métricas que foram salvos pelos scripts de ablação
    for path in sorted(ABLATION_DIR.glob("metricas_*.csv")):
        df = pd.read_csv(path)
        grupo, experimento, modelo = inferir_grupo_experimento_modelo(path)
        if "mae" in df.columns:
            rows.append(resumo(grupo, experimento, modelo, df))
    return rows


def rows_ensembles_otimizados() -> list[dict]:
    # pega predições dos modelos base e também variantes com delta_grid se existirem
    arquivos = {
        "ridge": REPORTS_DIR / "predicoes_walk_forward_ridge_baseline.csv",
        "lgb": REPORTS_DIR / "predicoes_walk_forward_lightgbm_tuned.csv",
        "xgb": REPORTS_DIR / "predicoes_walk_forward_xgboost_tuned.csv",
        "rf": REPORTS_DIR / "predicoes_walk_forward_randomforest_tuned.csv",
        "lgb_delta": ABLATION_DIR / "predicoes_target_delta_grid_retuned_lightgbm.csv",
        "xgb_delta": ABLATION_DIR / "predicoes_target_delta_grid_retuned_xgboost.csv",
    }
    available = {name: pd.read_csv(path) for name, path in arquivos.items() if path.exists()}
    if len(available) < 2:
        return []

    names = list(available)
    base = next(iter(available.values())).copy()
    pred_matrix = {name: df["pred_finish_position"].to_numpy() for name, df in available.items()}
    candidates = []

    # gera vetores de pesos pra todos os pares com passo de 0.1
    weight_vectors: list[np.ndarray] = []
    for i, j in itertools.combinations(range(len(names)), 2):
        for w in range(11):
            weights = np.zeros(len(names), dtype=float)
            weights[i] = w / 10.0
            weights[j] = 1.0 - weights[i]
            weight_vectors.append(weights)

    # trios que têm sentido combinar - inclui as variantes delta se disponíveis
    trios_preferenciais = [
        ("ridge", "lgb", "xgb"),
        ("ridge", "lgb", "lgb_delta"),
        ("ridge", "xgb", "xgb_delta"),
        ("lgb", "xgb", "xgb_delta"),
        ("lgb", "xgb", "lgb_delta"),
        ("lgb", "xgb", "rf"),
    ]
    for trio in trios_preferenciais:
        if not all(name in names for name in trio):
            continue
        idx = [names.index(name) for name in trio]
        for w1 in range(0, 11, 2):
            for w2 in range(0, 11 - w1, 2):
                w3 = 10 - w1 - w2
                weights = np.zeros(len(names), dtype=float)
                weights[idx[0]] = w1 / 10.0
                weights[idx[1]] = w2 / 10.0
                weights[idx[2]] = w3 / 10.0
                weight_vectors.append(weights)

    # remove duplicatas antes de avaliar
    unique_vectors = {tuple(weights.round(4)): weights for weights in weight_vectors}

    for weights in unique_vectors.values():
        values = sum(weights[i] * pred_matrix[names[i]] for i in range(len(names)))
        df_pred = base.copy()
        df_pred["pred_finish_position"] = values
        # so usa 2023-2024 pra escolher o melhor - 2025 fica de fora do tuning
        met_tuning = metricas_por_fold(df_pred[df_pred["valid_season"].isin([2023, 2024])])
        candidates.append((score_metricas(met_tuning), weights, df_pred))

    candidates.sort(key=lambda item: item[0], reverse=True)
    rows = []
    # reporta os 20 melhores com as metricas nos folds completos
    for rank, (score_tuning, weights, df_pred) in enumerate(candidates[:20], start=1):
        met = metricas_por_fold(df_pred)
        pesos = ";".join(f"{name}={weights[i]:.1f}" for i, name in enumerate(names) if weights[i] > 0)
        rows.append(
            resumo(
                "ensemble_otimizado",
                f"ensemble_grid_rank_{rank}",
                "ensemble",
                met,
                score_tuning=score_tuning,
                pesos=pesos,
            )
        )

    # salva as predicoes do ensemble campeao
    candidates[0][2].to_csv(
        ABLATION_DIR / "predicoes_ensemble_otimizado_melhor.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return rows


def main() -> None:
    # junta os resultados de métricas individuais e dos ensembles
    rows = rows_de_metricas()
    rows.extend(rows_ensembles_otimizados())

    df = pd.DataFrame(rows).sort_values(
        ["score_composto", "rmse_medio"],
        ascending=[False, True],
        na_position="last",
    )
    df.to_csv(ABLATION_DIR / "resultados_estudos_ablacao_completo.csv", index=False, encoding="utf-8-sig")
    print(df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
