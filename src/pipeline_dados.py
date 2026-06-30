"""
Pipeline de dados TCC-F1 - Executa todas as etapas de pré-processamento em sequência.

Etapas:
  01  limpeza_ergast_fastf1.py         - Merge Ergast + FastF1, filtro 2018+
  02  tratamento_dnf.py                - Classificação de DNFs, base DNF-Excluded
  03  encoding.py                      - One-Hot (circuito, construtor) + ordinal composto
  04  normalizacao.py                  - Z-score + MinMaxScaler
  05  tratamento_valores_ausentes.py   - Imputação de nulos
  06  tratamento_outliers.py           - Identificação e tratamento de outliers
  07  07_integrar_fontes_suporte.py    - Weather, circuit features, pitstops, safety car
  08  09_preparar_base_feature_engineering.py - Base FE-ready + manifest anti-leakage

Uso:
  python src/pipeline_dados.py                  # todas as etapas
  python src/pipeline_dados.py --from 3         # a partir da etapa 3
  python src/pipeline_dados.py --only 7 8       # apenas etapas específicas
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

# ordem das etapas com o script correspondente
ETAPAS = [
    {"numero": 1, "script": "limpeza_ergast_fastf1.py",       "descricao": "Limpeza Ergast + FastF1"},
    {"numero": 2, "script": "tratamento_dnf.py",              "descricao": "Tratamento de DNFs"},
    {"numero": 3, "script": "encoding.py",                    "descricao": "Encoding categórico"},
    {"numero": 4, "script": "normalizacao.py",                "descricao": "Normalização numérica"},
    {"numero": 5, "script": "tratamento_valores_ausentes.py", "descricao": "Imputação de ausentes"},
    {"numero": 6, "script": "tratamento_outliers.py",         "descricao": "Tratamento de outliers"},
    {"numero": 7, "script": "07_integrar_fontes_suporte.py",  "descricao": "Integração de fontes de suporte"},
    {"numero": 8, "script": "09_preparar_base_feature_engineering.py", "descricao": "Preparação para Feature Engineering"},
]


def executar_etapa(etapa):
    script = SRC_DIR / etapa["script"]
    num = etapa["numero"]
    desc = etapa["descricao"]

    print(f"\n{'='*60}")
    print(f"ETAPA {num:02d}: {desc}")
    print(f"Script: {script.name}")
    print("=" * 60)

    inicio = time.time()
    # roda o script como subprocesso e deixa o output aparecer no terminal
    resultado = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    duracao = time.time() - inicio

    if resultado.returncode != 0:
        print(f"\n[FALHA] Etapa {num} encerrou com código {resultado.returncode}.")
        print(f"Duração: {duracao:.1f}s")
        return False

    print(f"\n[OK] Etapa {num} concluída em {duracao:.1f}s.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de dados TCC-F1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from",
        type=int,
        dest="from_step",
        metavar="N",
        help="Executa a partir da etapa N (inclusive)",
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        metavar="N",
        help="Executa apenas as etapas listadas",
    )
    args = parser.parse_args()

    etapas_disponiveis = {e["numero"]: e for e in ETAPAS}

    # decide quais etapas rodar com base nos argumentos
    if args.only:
        numeros = sorted(set(args.only))
    elif args.from_step:
        numeros = [e["numero"] for e in ETAPAS if e["numero"] >= args.from_step]
    else:
        numeros = [e["numero"] for e in ETAPAS]

    invalidos = [n for n in numeros if n not in etapas_disponiveis]
    if invalidos:
        print(f"Etapas inválidas: {invalidos}")
        print(f"Etapas disponíveis: {list(etapas_disponiveis.keys())}")
        sys.exit(1)

    print(f"\nPipeline TCC-F1 - executando etapas: {numeros}")
    inicio_total = time.time()
    falhas = []

    # roda cada etapa em sequencia, para no primeiro erro
    for num in numeros:
        sucesso = executar_etapa(etapas_disponiveis[num])
        if not sucesso:
            falhas.append(num)
            print(f"\nAbortando pipeline na etapa {num}.")
            break

    duracao_total = time.time() - inicio_total
    print(f"\n{'='*60}")
    if falhas:
        print(f"Pipeline encerrado com FALHA na etapa {falhas[0]}.")
    else:
        print(f"Pipeline concluído com sucesso. ({duracao_total:.1f}s total)")
    print("=" * 60)

    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
