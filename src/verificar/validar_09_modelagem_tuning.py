from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "modelagem"
SRC = ROOT / "src"

MODELOS_OPTUNA = {
    "xgboost": {
        "trials": REPORTS / "optuna_xgboost_trials.csv",
        "best": REPORTS / "optuna_xgboost_best_params.json",
    },
    "lightgbm": {
        "trials": REPORTS / "optuna_lightgbm_trials.csv",
        "best": REPORTS / "optuna_lightgbm_best_params.json",
    },
    "randomforest": {
        "trials": REPORTS / "optuna_randomforest_trials.csv",
        "best": REPORTS / "optuna_randomforest_best_params.json",
    },
}

PARAM_PREFIX = "params_"


def carregar_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalizar_valor(valor):
    if isinstance(valor, float) and valor.is_integer():
        return int(valor)
    return valor


def validar_optuna() -> list[dict]:
    linhas = []
    for modelo, paths in MODELOS_OPTUNA.items():
        trials = pd.read_csv(paths["trials"], encoding="utf-8-sig")
        best_params = carregar_json(paths["best"])

        best_trial = trials.sort_values("value", ascending=True).iloc[0]
        divergencias = []
        for chave, valor_best in best_params.items():
            if chave in {"tempo_tuning_segundos", "lightgbm_version"}:
                continue

            col = f"{PARAM_PREFIX}{chave}"
            if col not in trials.columns:
                divergencias.append(f"{chave}: coluna ausente")
                continue

            valor_trial = normalizar_valor(best_trial[col])
            if isinstance(valor_trial, float) or isinstance(valor_best, float):
                iguais = math.isclose(float(valor_trial), float(valor_best), rel_tol=1e-12, abs_tol=1e-12)
            else:
                iguais = str(valor_trial) == str(valor_best)

            if not iguais:
                divergencias.append(f"{chave}: json={valor_best}, trial={valor_trial}")

        linhas.append(
            {
                "modelo": modelo,
                "n_trials": int(len(trials)),
                "melhor_trial": int(best_trial["number"]),
                "mae_tuning_best_trial": float(best_trial["value"]),
                "best_params_conferem": not divergencias,
                "divergencias": "; ".join(divergencias) if divergencias else "",
            }
        )
    return linhas


def validar_ridge() -> dict:
    grid = pd.read_csv(REPORTS / "ridge_alpha_grid.csv", encoding="utf-8-sig")
    best = carregar_json(REPORTS / "ridge_best_params.json")
    grid_ordenado = grid.sort_values(["mae_medio", "kendall_tau_medio"], ascending=[True, False])
    melhor = grid_ordenado.iloc[0]
    script = (SRC / "otimizacao_ridge_lambda.py").read_text(encoding="utf-8")

    alphas_interesse = [0.01, 0.1, 1.0, 10.0, 100.0]
    sensibilidade = []
    mae_base = float(melhor["mae_medio"])
    for alpha in alphas_interesse:
        linha = grid.iloc[(grid["alpha"] - alpha).abs().argsort()].iloc[0]
        sensibilidade.append(
            {
                "alpha": float(linha["alpha"]),
                "mae_tuning": float(linha["mae_medio"]),
                "delta_vs_001": float(linha["mae_medio"] - mae_base),
            }
        )

    return {
        "alpha_json": float(best["alpha"]),
        "alpha_grid": float(melhor["alpha"]),
        "mae_tuning_melhor": mae_base,
        "n_alphas": int(len(grid)),
        "usa_ridgecv": "RidgeCV" in script,
        "usa_folds_tuning": "FOLDS_TUNING" in script,
        "sensibilidade": sensibilidade,
    }


def validar_folds() -> dict:
    script = (SRC / "tuning_utils.py").read_text(encoding="utf-8")
    return {
        "fold_2023_presente": '{"train_until": 2022, "valid_season": 2023}' in script,
        "fold_2024_presente": '{"train_until": 2023, "valid_season": 2024}' in script,
        "fold_2025_em_avaliacao": '{"train_until": 2024, "valid_season": 2025}' in script,
    }


def main() -> None:
    optuna = validar_optuna()
    ridge = validar_ridge()
    folds = validar_folds()

    print("Validacao dos hiperparametros - docs/tecnico/09_modelagem_tuning.md")
    print("=" * 72)
    print("\nOptuna:")
    for linha in optuna:
        print(
            f"- {linha['modelo']}: trials={linha['n_trials']}, "
            f"melhor_trial={linha['melhor_trial']}, "
            f"MAE_tuning={linha['mae_tuning_best_trial']:.6f}, "
            f"best_params_conferem={linha['best_params_conferem']}"
        )
        if linha["divergencias"]:
            print(f"  divergencias: {linha['divergencias']}")

    print("\nRidge:")
    print(f"- alpha no JSON: {ridge['alpha_json']}")
    print(f"- alpha vencedor no grid temporal: {ridge['alpha_grid']}")
    print(f"- MAE medio tuning vencedor: {ridge['mae_tuning_melhor']:.6f}")
    print(f"- alphas avaliados: {ridge['n_alphas']}")
    print(f"- usa RidgeCV no script: {ridge['usa_ridgecv']}")
    print(f"- usa FOLDS_TUNING no script: {ridge['usa_folds_tuning']}")
    print("- sensibilidade:")
    for item in ridge["sensibilidade"]:
        print(
            f"  alpha={item['alpha']}: MAE={item['mae_tuning']:.6f}, "
            f"delta={item['delta_vs_001']:.6f}"
        )

    print("\nFolds:")
    for chave, valor in folds.items():
        print(f"- {chave}: {valor}")


if __name__ == "__main__":
    main()
