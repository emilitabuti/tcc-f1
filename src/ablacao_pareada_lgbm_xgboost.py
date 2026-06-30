from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna
import pandas as pd

from estudos_ablacao_completo import (
    ABLATION_DIR,
    FOLDS_AVALIACAO,
    FOLDS_TUNING,
    SCORE_PROFILES,
    avaliar_regressor,
    criar_regressor,
    score_metricas,
    sugerir_lgb,
    sugerir_xgb,
)
from tuning_utils import carregar_dados


OUTPUT_DIR = ABLATION_DIR / "pareada_lgbm_xgboost"

MODELOS = {
    "LightGBM": {
        "objective": "regression",
        "sugerir": sugerir_lgb,
    },
    "XGBoost": {
        "objective": "reg:squarederror",
        "sugerir": sugerir_xgb,
    },
}

# escopo do experimento: só finish_position como target
TARGET_MODES = ["finish"]
DECAYS = [0.95, 0.99]
SCORE_PROFILE_NAMES = ["atual", "rmse_r2", "erro_continuo"]


def safe_name(*partes: object) -> str:
    # monta um nome de arquivo sem caracteres especiais
    return "_".join(str(parte).replace(".", "p").replace("=", "-") for parte in partes)


def resumo_metricas(metricas: pd.DataFrame) -> dict:
    # agrega as métricas dos folds em médias
    return {
        "mae_medio": float(metricas["mae"].mean()),
        "rmse_medio": float(metricas["rmse"].mean()),
        "r2_medio": float(metricas["r2"].mean()),
        "kendall_tau_medio": float(metricas["kendall_tau"].mean()),
        "score_composto": score_metricas(metricas, SCORE_PROFILES["atual"]),
    }


def tunar_configuracao(
    x: pd.DataFrame,
    y: pd.DataFrame,
    modelo: str,
    target_mode: str,
    decay: float,
    score_profile: str,
    trials: int,
) -> dict:
    spec = MODELOS[modelo]
    weights = SCORE_PROFILES[score_profile]
    sugerir = spec["sugerir"]
    objective = spec["objective"]

    def objective_fn(trial: optuna.Trial) -> float:
        params = sugerir(trial)
        _, metricas_tuning = avaliar_regressor(
            x=x,
            y=y,
            criar_modelo=lambda: criar_regressor(modelo, params, objective),
            folds=FOLDS_TUNING,
            decay=decay,
            target_mode=target_mode,
        )
        return score_metricas(metricas_tuning, weights)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    inicio = time.perf_counter()
    study.optimize(objective_fn, n_trials=trials, show_progress_bar=False)
    tempo = time.perf_counter() - inicio

    # avalia os melhores params nos folds de validação final
    best_params = dict(study.best_params)
    predicoes, metricas = avaliar_regressor(
        x=x,
        y=y,
        criar_modelo=lambda: criar_regressor(modelo, best_params, objective),
        folds=FOLDS_AVALIACAO,
        decay=decay,
        target_mode=target_mode,
    )

    # salva os artefatos de cada configuração individualmente
    nome = safe_name(target_mode, f"decay{decay:.2f}", score_profile, modelo.lower())
    caminho_metricas = OUTPUT_DIR / f"metricas_{nome}.csv"
    caminho_predicoes = OUTPUT_DIR / f"predicoes_{nome}.csv"
    caminho_params = OUTPUT_DIR / f"params_{nome}.json"
    metricas.to_csv(caminho_metricas, index=False)
    predicoes.to_csv(caminho_predicoes, index=False)
    caminho_params.write_text(
        json.dumps(
            {
                "modelo": modelo,
                "objective": objective,
                "target_mode": target_mode,
                "decay": decay,
                "score_profile": score_profile,
                "score_weights": weights,
                "trials": trials,
                "best_value_tuning": float(study.best_value),
                "best_params": best_params,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # monta a linha de resultado com as métricas e flags de metas
    row = {
        "config_id": safe_name(target_mode, f"decay{decay:.2f}", score_profile),
        "modelo": modelo,
        "objective": objective,
        "target_mode": target_mode,
        "decay": decay,
        "score_profile": score_profile,
        "trials": trials,
        "best_value_tuning": float(study.best_value),
        "tempo_tuning_segundos": tempo,
        "caminho_metricas": str(caminho_metricas),
        "caminho_predicoes": str(caminho_predicoes),
        "caminho_params": str(caminho_params),
    }
    row.update(resumo_metricas(metricas))
    row["score_perfil_avaliacao"] = score_metricas(metricas, weights)
    # verifica se a configuracao bate as metas numericas do TCC
    row["bate_meta_rmse_lt_3"] = row["rmse_medio"] < 3.0
    row["bate_meta_r2_ge_066"] = row["r2_medio"] >= 0.66
    row["bate_meta_kendall_ge_060"] = row["kendall_tau_medio"] >= 0.60
    row["bate_metas_principais"] = (
        row["bate_meta_rmse_lt_3"]
        and row["bate_meta_r2_ge_066"]
        and row["bate_meta_kendall_ge_060"]
    )
    return row


def tabela_markdown(df: pd.DataFrame, colunas: list[str], n: int | None = None) -> str:
    dados = df[colunas].copy()
    if n is not None:
        dados = dados.head(n)
    return dados.to_markdown(index=False, floatfmt=".4f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa ablação pareada entre LightGBM e XGBoost."
    )
    parser.add_argument("--trials", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = carregar_dados()

    rows = []
    total = len(TARGET_MODES) * len(DECAYS) * len(SCORE_PROFILE_NAMES) * len(MODELOS)
    atual = 0
    # loop principal: varre todas as combinações de config e modelo
    for target_mode in TARGET_MODES:
        for decay in DECAYS:
            for score_profile in SCORE_PROFILE_NAMES:
                for modelo in MODELOS:
                    atual += 1
                    print(
                        f"[{atual}/{total}] {modelo} | target={target_mode} | "
                        f"decay={decay:.2f} | score={score_profile} | trials={args.trials}",
                        flush=True,
                    )
                    rows.append(
                        tunar_configuracao(
                            x=x,
                            y=y,
                            modelo=modelo,
                            target_mode=target_mode,
                            decay=decay,
                            score_profile=score_profile,
                            trials=args.trials,
                        )
                    )

    resultados = pd.DataFrame(rows)
    resultados = resultados.sort_values(
        ["score_composto", "rmse_medio", "r2_medio"],
        ascending=[False, True, False],
    )
    resultados.to_csv(OUTPUT_DIR / "resultados_pareados.csv", index=False)

    print("\nMelhores resultados:")
    print(
        resultados[
            [
                "modelo",
                "target_mode",
                "decay",
                "score_profile",
                "mae_medio",
                "rmse_medio",
                "r2_medio",
                "kendall_tau_medio",
                "score_composto",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
