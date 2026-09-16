"""ETAPA 6 - MATRIZ DE MODELAGEM COM ONE-HOT ENCODING.

Prepara uma base diretamente utilizavel por modelos do scikit-learn:

- usa somente variaveis disponiveis antes da corrida;
- separa `finish_position` como alvo;
- aplica One-Hot Encoding apenas nas categoricas escolhidas;
- remove colunas textuais auxiliares e colunas pos-corrida;
- valida que a matriz final nao possui NaN, object dtype ou vazamento.

Observacao metodologica:
para avaliacao temporal, o encoder deve ser ajustado apenas no conjunto de
treino e aplicado no conjunto de teste. Este script gera uma matriz historica
unica para a proxima etapa do projeto.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from colunas_modelagem import COLUNA_ALVO, validar_sem_vazamento

PASTA_DADOS = "dados"
PASTA_PROCESSADOS = os.path.join(PASTA_DADOS, "processados")
ARQUIVO_ENTRADA = os.path.join(PASTA_PROCESSADOS, "base_tratada_2018_2025.csv")
ARQUIVO_SAIDA = os.path.join(PASTA_PROCESSADOS, "base_modelagem_2018_2025.csv")
ARQUIVO_ENCODER = os.path.join(PASTA_PROCESSADOS, "one_hot_encoder_2018_2025.joblib")
ARQUIVO_METADADOS = os.path.join(PASTA_PROCESSADOS, "base_modelagem_metadados_2018_2025.json")

COLUNAS_CATEGORICAS = [
    "driver_id",
    "constructor_id",
    "circuit_id",
    "circuit_type",
]

COLUNAS_NUMERICAS = [
    "season",
    "round",
    "grid_position",
    "altitude_m",
    "corners",
    "length_km",
    "lat",
    "long",
    "qualifying_position",
    "Q1_s",
    "Q2_s",
    "Q3_s",
    "registrou_tempo_q1",
    "chegou_no_q2",
    "chegou_no_q3",
    "qualifying_position_imputada",
    "diferenca_grid_qualifying",
]

COLUNAS_IDENTIFICACAO = [
    "season",
    "round",
    "race_name",
    "driver_id",
    "constructor_id",
    "circuit_id",
]


def validar_colunas(base):
    colunas_necessarias = sorted(set(COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS + [COLUNA_ALVO]))
    ausentes = [coluna for coluna in colunas_necessarias if coluna not in base.columns]
    if ausentes:
        raise ValueError("Colunas obrigatorias ausentes: " + ", ".join(ausentes))


def montar_matriz_modelagem(base):
    """Retorna base de modelagem, encoder, metadados e identificadores."""
    validar_colunas(base)

    colunas_modelo_antes_encoding = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS
    validar_sem_vazamento(colunas_modelo_antes_encoding)

    y = pd.to_numeric(base[COLUNA_ALVO], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Alvo {COLUNA_ALVO} possui valores ausentes ou invalidos.")

    X_num = base[COLUNAS_NUMERICAS].copy()
    for coluna in COLUNAS_NUMERICAS:
        X_num[coluna] = pd.to_numeric(X_num[coluna], errors="coerce")

    if X_num.isna().any().any():
        faltantes = X_num.isna().sum()
        faltantes = faltantes[faltantes > 0]
        raise ValueError("Features numericas possuem NaN:\n" + faltantes.to_string())

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.int8,
    )
    dados_codificados = encoder.fit_transform(base[COLUNAS_CATEGORICAS])
    colunas_codificadas = encoder.get_feature_names_out(COLUNAS_CATEGORICAS)
    X_cat = pd.DataFrame(dados_codificados, columns=colunas_codificadas, index=base.index)

    X = pd.concat([X_num, X_cat], axis=1)
    base_modelagem = pd.concat([base[[COLUNA_ALVO]].copy(), X], axis=1)

    validar_base_modelagem(base_modelagem)

    identificadores = base[[coluna for coluna in COLUNAS_IDENTIFICACAO if coluna in base.columns]].copy()
    metadados = {
        "arquivo_entrada": ARQUIVO_ENTRADA,
        "arquivo_saida": ARQUIVO_SAIDA,
        "coluna_alvo": COLUNA_ALVO,
        "colunas_numericas": COLUNAS_NUMERICAS,
        "colunas_categoricas": COLUNAS_CATEGORICAS,
        "colunas_one_hot": colunas_codificadas.tolist(),
        "colunas_modelo": X.columns.tolist(),
        "linhas": int(len(base_modelagem)),
        "colunas": int(len(base_modelagem.columns)),
        "observacao": (
            "Para avaliacao temporal, ajustar o encoder somente no treino "
            "e aplicar no teste."
        ),
    }

    return base_modelagem, encoder, metadados, identificadores


def validar_base_modelagem(base_modelagem):
    if base_modelagem.isna().any().any():
        faltantes = base_modelagem.isna().sum()
        faltantes = faltantes[faltantes > 0]
        raise ValueError("Base de modelagem possui NaN:\n" + faltantes.to_string())

    colunas_object = base_modelagem.select_dtypes(include="object").columns.tolist()
    if colunas_object:
        raise ValueError("Base de modelagem possui colunas textuais: " + ", ".join(colunas_object))

    if base_modelagem.columns.duplicated().any():
        duplicadas = base_modelagem.columns[base_modelagem.columns.duplicated()].tolist()
        raise ValueError("Base de modelagem possui colunas duplicadas: " + ", ".join(duplicadas))

    validar_sem_vazamento([coluna for coluna in base_modelagem.columns if coluna != COLUNA_ALVO])


def mostrar_resumo(base_original, base_modelagem, encoder, metadados):
    print()
    print("=" * 70)
    print("RESUMO DA BASE DE MODELAGEM")
    print("=" * 70)
    print("Linhas:", len(base_modelagem))
    print("Colunas antes:", len(base_original.columns))
    print("Colunas na base de modelagem:", len(base_modelagem.columns))
    print("Features numericas diretas:", len(COLUNAS_NUMERICAS))
    print("Features one-hot:", len(metadados["colunas_one_hot"]))
    print("Alvo:", COLUNA_ALVO)
    print()
    print("Categorias codificadas:")
    for coluna, categorias in zip(COLUNAS_CATEGORICAS, encoder.categories_):
        print(f"- {coluna}: {len(categorias)} categorias")


if __name__ == "__main__":
    print("=" * 70)
    print("ETAPA 6 - BASE DE MODELAGEM")
    print("=" * 70)

    if not os.path.exists(ARQUIVO_ENTRADA):
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {ARQUIVO_ENTRADA}")

    print()
    print("Lendo base tratada...")
    base = pd.read_csv(ARQUIVO_ENTRADA)
    print("Linhas:", len(base), "| Colunas:", len(base.columns))

    print()
    print("Montando matriz numerica sem vazamento...")
    base_modelagem, encoder, metadados, identificadores = montar_matriz_modelagem(base)

    os.makedirs(PASTA_PROCESSADOS, exist_ok=True)
    base_modelagem.to_csv(ARQUIVO_SAIDA, index=False)
    joblib.dump(encoder, ARQUIVO_ENCODER)
    with open(ARQUIVO_METADADOS, "w", encoding="utf-8") as arquivo:
        json.dump(metadados, arquivo, ensure_ascii=False, indent=2)

    mostrar_resumo(base, base_modelagem, encoder, metadados)

    print()
    print("=" * 70)
    print("ETAPA CONCLUIDA")
    print("=" * 70)
    print("Base de modelagem salva em:", ARQUIVO_SAIDA)
    print("Encoder salvo em:", ARQUIVO_ENCODER)
    print("Metadados salvos em:", ARQUIVO_METADADOS)
