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

TARGET_MODES = ["finish"]
DECAYS = [0.95, 0.99]
SCORE_PROFILE_NAMES = ["atual", "rmse_r2", "erro_continuo"]


def safe_name(*partes: object) -> str:
    return "_".join(str(parte).replace(".", "p").replace("=", "-") for parte in partes)


def resumo_metricas(metricas: pd.DataFrame) -> dict:
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

    best_params = dict(study.best_params)
    predicoes, metricas = avaliar_regressor(
        x=x,
        y=y,
        criar_modelo=lambda: criar_regressor(modelo, best_params, objective),
        folds=FOLDS_AVALIACAO,
        decay=decay,
        target_mode=target_mode,
    )

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


def gerar_relatorio(resultados: pd.DataFrame, trials: int) -> None:
    ordenado = resultados.sort_values(
        ["score_composto", "rmse_medio", "r2_medio"],
        ascending=[False, True, False],
    )
    melhor = ordenado.iloc[0]
    metas = ordenado[ordenado["bate_metas_principais"]].copy()
    melhores_modelo = (
        ordenado.sort_values("score_composto", ascending=False)
        .groupby("modelo", as_index=False)
        .head(1)
        .sort_values("score_composto", ascending=False)
    )

    comparacao = resultados.pivot_table(
        index=["target_mode", "decay", "score_profile"],
        columns="modelo",
        values="score_composto",
        aggfunc="first",
    ).reset_index()
    comparacao["melhor_modelo_config"] = comparacao[["LightGBM", "XGBoost"]].idxmax(axis=1)
    comparacao["delta_lgbm_menos_xgb"] = comparacao["LightGBM"] - comparacao["XGBoost"]
    comparacao = comparacao.sort_values(
        ["target_mode", "decay", "score_profile"],
        ascending=[True, True, True],
    )

    colunas_resultado = [
        "modelo",
        "target_mode",
        "decay",
        "score_profile",
        "mae_medio",
        "rmse_medio",
        "r2_medio",
        "kendall_tau_medio",
        "score_composto",
        "score_perfil_avaliacao",
    ]
    colunas_comparacao = [
        "target_mode",
        "decay",
        "score_profile",
        "LightGBM",
        "XGBoost",
        "delta_lgbm_menos_xgb",
        "melhor_modelo_config",
    ]

    linhas = [
        "# Ablação pareada LightGBM vs XGBoost",
        "",
        "## Conclusão Executiva",
        "",
        (
            f"- Melhor configuração por score oficial: {melhor['modelo']} com "
            f"target oficial fixo `finish_position`, `decay={melhor['decay']:.2f}` "
            f"e `score_profile={melhor['score_profile']}`."
        ),
        (
            f"- Métricas médias: MAE {melhor['mae_medio']:.4f}, "
            f"RMSE {melhor['rmse_medio']:.4f}, R2 {melhor['r2_medio']:.4f}, "
            f"Kendall tau {melhor['kendall_tau_medio']:.4f}, "
            f"score {melhor['score_composto']:.4f}."
        ),
        "- O alvo nao foi transformado: todas as configuracoes mantem `target_mode=finish`.",
        "- Como este estudo compara apenas LightGBM e XGBoost, ele orienta o próximo retuning; não substitui sozinho a tabela final com todos os modelos.",
        "",
        "## Protocolo",
        "",
        "- Todos os modelos foram avaliados com as mesmas configurações experimentais: mesmos folds, mesmos targets, mesmos fatores de time-decay, mesmos perfis de score e mesmo número de trials.",
        "- A única diferença é o espaço de hiperparâmetros próprio de cada algoritmo.",
        "- Folds de tuning: 2023 e 2024.",
        "- Folds de avaliação final: 2023, 2024 e 2025.",
        f"- Trials por combinação e modelo: {trials}.",
        f"- Target oficial mantido: {', '.join(TARGET_MODES)}.",
        f"- Decays testados: {', '.join(str(v) for v in DECAYS)}.",
        f"- Perfis de score testados: {', '.join(SCORE_PROFILE_NAMES)}.",
        "- Transformacoes do alvo, como `delta_grid` ou `rank_norm_grid20`, nao fazem parte do escopo oficial.",
        "",
        "## Melhores resultados por score oficial",
        "",
        tabela_markdown(ordenado, colunas_resultado, n=12),
        "",
        "## Melhores por modelo",
        "",
        tabela_markdown(melhores_modelo, colunas_resultado),
        "",
        "## Configurações que batem as metas principais",
        "",
    ]

    if metas.empty:
        linhas.append("Nenhuma configuração bateu simultaneamente RMSE < 3.0, R2 >= 0.66 e Kendall >= 0.60.")
    else:
        linhas.append(tabela_markdown(metas, colunas_resultado, n=20))

    linhas.extend(
        [
            "",
            "## Comparação pareada por configuração",
            "",
            tabela_markdown(comparacao, colunas_comparacao),
            "",
            "## Artefatos",
            "",
            f"- Resultados consolidados: `{OUTPUT_DIR / 'resultados_pareados.csv'}`",
            f"- Relatório: `{OUTPUT_DIR / 'relatorio_pareado.md'}`",
            "- Métricas, predições e parâmetros por configuração foram salvos no mesmo diretório.",
        ]
    )

    (OUTPUT_DIR / "relatorio_pareado.md").write_text("\n".join(linhas), encoding="utf-8")


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
    gerar_relatorio(resultados, args.trials)

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
    print(f"\nRelatorio salvo em: {OUTPUT_DIR / 'relatorio_pareado.md'}")


if __name__ == "__main__":
    main()
