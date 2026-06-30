from __future__ import annotations

# script obsoleto - esse target foi testado mas a gente decidiu manter finish_position
# deixei aqui como histórico, não roda mais

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
    raise SystemExit(
        "Script obsoleto: rank_norm_grid20 foi rejeitado na versao final. "
        "Use scripts oficiais com target finish_position."
    )


if __name__ == "__main__":
    main()
