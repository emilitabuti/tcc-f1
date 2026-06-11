from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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


OUTPUT_ALPHAS = REPORTS_DIR / "ridge_alpha_grid.csv"
OUTPUT_BEST_PARAMS = REPORTS_DIR / "ridge_best_params.json"
OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_ridge_baseline.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_ridge_baseline.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_quinta_semana2_ridge_baseline.txt"

ALPHAS = np.logspace(-2, 2, 25)


class RidgeBaselineRegressor:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)

    def fit(self, x, y, sample_weight=None):
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))


def criar_modelo(alpha: float) -> RidgeBaselineRegressor:
    return RidgeBaselineRegressor(alpha=alpha)


def gerar_relatorio_ridge(
    df_metricas: pd.DataFrame,
    best_value: float,
    best_params: dict,
    decay: float,
) -> None:
    linhas = [
        "Relatorio - 28/05 - Ridge Regression Baseline",
        "=" * 56,
        "",
        "Objetivo:",
        "- Otimizar alpha por score composto multi-metrica nos folds de tuning 2023 e 2024.",
        "- Reavaliar o melhor alpha em 2023, 2024 e 2025 via walk-forward.",
        "- Usar StandardScaler porque Ridge e sensivel a escala das features.",
        "",
        "Bases utilizadas:",
        "- data/processed/dataset_modelagem_X_2018_2025.csv",
        "- data/processed/dataset_modelagem_y_2018_2025.csv",
        "",
        "Time-decay:",
        f"- Fator usado via sample_weight: {decay}",
        "",
        "Grid-search:",
        f"- Alphas avaliados: {len(ALPHAS)}",
        f"- Melhor score composto nos folds de tuning: {best_value:.6f}",
        f"- Melhores parametros: {best_params}",
        "",
        "Metricas finais por fold:",
        df_metricas.to_string(index=False),
        "",
        "Artefatos gerados:",
        f"- {OUTPUT_ALPHAS}",
        f"- {OUTPUT_BEST_PARAMS}",
        f"- {OUTPUT_PREDICOES}",
        f"- {OUTPUT_METRICAS}",
        f"- {OUTPUT_RELATORIO}",
    ]

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


def main() -> None:
    inicio = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    decay = carregar_decay_escolhido()

    resultados = []
    for alpha in ALPHAS:
        _, df_metricas = avaliar_modelo(
            x=x,
            y=y,
            criar_modelo=lambda alpha=alpha: criar_modelo(alpha),
            folds=FOLDS_TUNING,
            decay=decay,
        )
        score = score_composto_metricas(df_metricas)
        resultados.append(
            {
                "alpha": float(alpha),
                "score_composto": float(score),
                "mae_medio": float(df_metricas["mae"].mean()),
                "mae_std": float(df_metricas["mae"].std()),
                "rmse_medio": float(df_metricas["rmse"].mean()),
                "r2_medio": float(df_metricas["r2"].mean()),
                "kendall_tau_medio": float(df_metricas["kendall_tau"].mean()),
            }
        )

    df_alphas = pd.DataFrame(resultados).sort_values(
        ["score_composto", "mae_medio"],
        ascending=[False, True],
    )
    melhor = df_alphas.iloc[0].to_dict()
    best_params = {
        "alpha": float(melhor["alpha"]),
        "normalizacao": "StandardScaler",
        "modelo": "Ridge",
        "tempo_tuning_segundos": float(time.perf_counter() - inicio),
        "score_composto_tuning": float(melhor["score_composto"]),
    }

    df_predicoes, df_metricas = avaliar_modelo(
        x=x,
        y=y,
        criar_modelo=lambda: criar_modelo(best_params["alpha"]),
        folds=FOLDS_AVALIACAO,
        decay=decay,
    )

    df_alphas.to_csv(OUTPUT_ALPHAS, index=False, encoding="utf-8-sig")
    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")
    salvar_json(OUTPUT_BEST_PARAMS, best_params)

    gerar_relatorio_ridge(
        df_metricas=df_metricas,
            best_value=float(melhor["score_composto"]),
        best_params=best_params,
        decay=decay,
    )

    print("Otimizacao Ridge baseline concluida.")
    print(pd.Series(best_params).to_string())
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
