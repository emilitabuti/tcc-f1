from __future__ import annotations

"""
OBSOLETO / HISTORICO.

Este script validava um target rank_norm_grid20. Ele foi desativado na versao
final porque o target oficial do TCC deve permanecer finish_position.
"""

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
