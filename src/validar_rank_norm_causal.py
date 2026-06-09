from __future__ import annotations

import pandas as pd

from estudos_ablacao_completo import (
    ABLATION_DIR,
    DEFAULT_DECAY,
    DEFAULT_TRIALS,
    FOLDS_AVALIACAO,
    criar_regressor,
    resumo,
    score_metricas,
    tunar_regressor,
)
from estudos_ablacao_modelos import carregar_dados


def main() -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    x, y, _, _ = carregar_dados()
    rows: list[dict] = []

    resultados = []
    for modelo, objective in [
        ("LightGBM", "regression"),
        ("XGBoost", "reg:squarederror"),
    ]:
        pred = tunar_regressor(
            rows,
            x,
            y,
            grupo="validacao_causal_target",
            experimento="target_rank_norm_grid20_retuned",
            modelo=modelo,
            objective=objective,
            target_mode="rank_norm_grid20",
            decay=DEFAULT_DECAY,
            score_profile="atual",
            trials=DEFAULT_TRIALS,
        )
        if pred is not None:
            resultados.append(pred)

    df = pd.DataFrame(rows).sort_values("score_composto", ascending=False)
    saida = ABLATION_DIR / "resultados_rank_norm_causal.csv"
    df.to_csv(saida, index=False, encoding="utf-8-sig")

    completo = ABLATION_DIR / "resultados_estudos_ablacao_completo.csv"
    if completo.exists():
        atual = pd.read_csv(completo)
        chave = ["grupo", "experimento", "modelo"]
        atual = atual[
            ~atual.set_index(chave).index.isin(df.set_index(chave).index)
        ]
        pd.concat([atual, df], ignore_index=True).sort_values(
            ["score_composto", "rmse_medio"],
            ascending=[False, True],
            na_position="last",
        ).to_csv(completo, index=False, encoding="utf-8-sig")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
