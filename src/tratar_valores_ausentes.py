#trata valores ausentes e invalidos da base consolidada antes da modelagem

import os
import numpy as np
import pandas as pd

PASTA_DADOS = "dados"
PASTA_PROCESSADOS = os.path.join(PASTA_DADOS, "processados")

ARQUIVO_ENTRADA = os.path.join(PASTA_PROCESSADOS, "base_consolidada_2018_2025.csv")
ARQUIVO_SAIDA = os.path.join(PASTA_PROCESSADOS, "base_tratada_2018_2025.csv")
PASTA_LOGS = os.path.join(PASTA_DADOS, "logs")
ARQUIVO_LOG_TRATAMENTO = os.path.join(PASTA_LOGS, "log_tratamento_valores_2018_2025.csv")

# colunas de qualifying que podem ficar ausentes quando o piloto nao marcou tempo
COLUNAS_QUALIFYING_TEMPO = ["Q1_s", "Q2_s", "Q3_s"]

# colunas usadas pra marcar disponibilidade de clima
COLUNAS_CLIMA = ["temp_ar_media", "temp_pista_media", "umidade_media", "vento_media", "choveu"]

# colunas preenchidas com historico de corridas anteriores
COLUNAS_MEDIANA_TEMPORAL = [
    "fastf1_avg_lap_time", "fastf1_best_lap_time", "fastf1_num_voltas", "fastf1_num_stints",
    "fastf1_tyre_life_media", "tempo_total_pitstop", "tempo_medio_pitstop",
    "temp_ar_media", "temp_pista_media", "umidade_media", "vento_media",
]
COLUNAS_CONTAGEM = {"fastf1_num_voltas", "fastf1_num_stints"}

# formato do arquivo que registra cada alteração feita na base
COLUNAS_LOG = ["season", "round", "race_name", "circuit_id", "driver_id", "constructor_id",
               "coluna", "tipo_tratamento", "valor_original", "valor_novo", "metodo", "observacao"]


def colunas_identificacao(base):
    """pega as colunas usadas pra identificar cada linha no log"""
    preferidas = ["season", "round", "race_name", "circuit_id", "driver_id", "constructor_id"]
    return [coluna for coluna in preferidas if coluna in base.columns]

def registrar_tratamento(log_tratamento, base, indices, coluna, valores_originais, valores_novos, tipo_tratamento, metodo, observacao=""):
    """guarda no log quais valores foram alterados"""
    identificadores = colunas_identificacao(base)
    for indice in indices:
        registro = {coluna_id: base.at[indice, coluna_id] for coluna_id in identificadores}
        registro.update({
            "coluna": coluna,
            "tipo_tratamento": tipo_tratamento,
            "valor_original": valores_originais.loc[indice],
            "valor_novo": valores_novos.loc[indice],
            "metodo": metodo,
            "observacao": observacao,
        })
        log_tratamento.append(registro)

def somar_preenchimentos(resumo, coluna, quantidade):
    """acumula quantos valores foram preenchidos por coluna"""
    if quantidade == 0:
        return
    resumo[coluna] = resumo.get(coluna, 0) + int(quantidade)

def marcar_invalido(base, coluna, esconder, motivo, log_tratamento):
    """troca valores invalidos por NaN para serem tratados depois"""
    esconder = esconder & base[coluna].notna()
    if not esconder.any():
        return base

    indices = base.index[esconder]
    valores_originais = base[coluna].copy()
    base.loc[esconder, coluna] = np.nan
    registrar_tratamento(log_tratamento, base, indices, coluna,
        valores_originais, base[coluna], tipo_tratamento="invalido",
        metodo="validacao", observacao=motivo,)
    return base

def normalizar_choveu(valor):
    """converte a coluna de chuva para 0, 1 ou NaN"""
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (bool, np.bool_)):
        return int(valor)

    texto = str(valor).strip().lower()
    if texto in {"true", "1", "sim", "yes"}:
        return 1
    if texto in {"false", "0", "nao", "não", "no"}:
        return 0
    return np.nan

def mascara_corridas_anteriores(base, season, round_):
    """seleciona somente corridas anteriores a corrida atual"""
    return (base["season"] < season) | ((base["season"] == season) & (base["round"] < round_))

def mediana_historica(historico, coluna, circuit_id, season):
    """calcula mediana usando circuito, temporada ou historico geral"""
    dados = historico.copy()
    if coluna in {"tempo_total_pitstop", "tempo_medio_pitstop"}:
        dados = dados[dados["num_pitstops"] > 0]

    if circuit_id is not None and "circuit_id" in dados.columns:
        valores = dados.loc[dados["circuit_id"] == circuit_id, coluna].dropna()
        if not valores.empty:
            return valores.median()

    valores = dados.loc[dados["season"] == season, coluna].dropna()
    if not valores.empty:
        return valores.median()

    valores = dados[coluna].dropna()
    if not valores.empty:
        return valores.median()
    return np.nan

def moda_historica(historico, coluna, circuit_id, season):
    """calcula moda usando circuito, temporada ou historico geral"""
    if circuit_id is not None and "circuit_id" in historico.columns:
        valores = historico.loc[historico["circuit_id"] == circuit_id, coluna].dropna()
        if not valores.empty:
            return valores.mode().iloc[0]

    valores = historico.loc[historico["season"] == season, coluna].dropna()
    if not valores.empty:
        return valores.mode().iloc[0]

    valores = historico[coluna].dropna()
    if not valores.empty:
        return valores.mode().iloc[0]
    return np.nan

def preencher_faltantes(base, coluna, mascara, valor, log_tratamento, tipo, metodo, observacao):
    """preenche valores ausentes e registra a alteração no log"""
    if not mascara.any():
        return base

    valores_originais = base[coluna].copy()
    base.loc[mascara, coluna] = valor
    registrar_tratamento(
        log_tratamento,
        base,
        base.index[mascara],
        coluna,
        valores_originais,
        base[coluna],
        tipo_tratamento=tipo,
        metodo=metodo,
        observacao=observacao,
    )
    return base

def tratar_base():
    """executa o tratamento completo da base consolidada"""
    os.makedirs(PASTA_PROCESSADOS, exist_ok=True)
    os.makedirs(PASTA_LOGS, exist_ok=True)

    if not os.path.exists(ARQUIVO_ENTRADA):
        raise FileNotFoundError(f"Arquivo nao encontrado: {ARQUIVO_ENTRADA}")

    base = pd.read_csv(ARQUIVO_ENTRADA)
    log_tratamento = []
    resumo_preenchimentos = {}

    ausentes = base.isna().sum()
    ausentes = ausentes[ausentes > 0]
    print("Tratando valores ausentes...")
    print("Linhas de entrada:", len(base))
    print("Colunas com NaN antes:", len(ausentes))

    base["season"] = pd.to_numeric(base["season"], errors="coerce")
    base["round"] = pd.to_numeric(base["round"], errors="coerce")
    base = base.sort_values(["season", "round"]).reset_index(drop=True)

    colunas_numericas = [
        "num_pitstops", "tempo_total_pitstop", "tempo_medio_pitstop", "fastf1_avg_lap_time",
        "fastf1_best_lap_time", "fastf1_num_voltas", "fastf1_num_stints", "fastf1_tyre_life_media",
        "Q1_s", "Q2_s", "Q3_s", "qualifying_position", "grid_position", "temp_ar_media",
        "temp_pista_media", "umidade_media", "vento_media", "pitstop_dado_disponivel",
    ]
    for coluna in colunas_numericas:
        if coluna in base.columns:
            base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    if "choveu" in base.columns:
        base["choveu"] = base["choveu"].apply(normalizar_choveu)

    if "pitstop_dado_disponivel" in base.columns:
        base["pitstop_dado_disponivel"] = base["pitstop_dado_disponivel"].fillna(0).astype(int)

    for coluna in ["fastf1_avg_lap_time", "fastf1_best_lap_time"]:
        if coluna in base.columns:
            base = marcar_invalido(base, coluna, base[coluna] <= 0, "tempo de volta menor ou igual a zero", log_tratamento)

    for coluna in COLUNAS_QUALIFYING_TEMPO:
        if coluna in base.columns:
            base = marcar_invalido(base, coluna, base[coluna] <= 0, "tempo de classificacao menor ou igual a zero", log_tratamento)

    for coluna in ["num_pitstops", "fastf1_num_voltas", "fastf1_num_stints", "fastf1_tyre_life_media"]:
        if coluna in base.columns:
            base = marcar_invalido(base, coluna, base[coluna] < 0, "valor negativo impossivel", log_tratamento)

    for coluna in ["tempo_total_pitstop", "tempo_medio_pitstop"]:
        if coluna in base.columns:
            base = marcar_invalido(base, coluna, base[coluna] < 0, "tempo de pitstop negativo", log_tratamento)

    if "umidade_media" in base.columns:
        base = marcar_invalido(base, "umidade_media", (base["umidade_media"] < 0) | (base["umidade_media"] > 100), "umidade fora do intervalo 0-100", log_tratamento)

    if "vento_media" in base.columns:
        base = marcar_invalido(base, "vento_media", base["vento_media"] < 0, "velocidade do vento negativa", log_tratamento)

    if "qualifying_position" in base.columns:
        base = marcar_invalido(base, "qualifying_position", base["qualifying_position"] <= 0, "posicao de classificacao invalida", log_tratamento)

    # diferencia ausencia de parada da indisponibilidade dos dados de pit stop
    if "num_pitstops" in base.columns:
        base["num_pitstops_imputado"] = base["num_pitstops"].isna().astype(int)

        if "pitstop_dado_disponivel" in base.columns:
            mascara_sem_parada = base["num_pitstops"].isna() & (base["pitstop_dado_disponivel"] == 1)
            quantidade_sem_parada = mascara_sem_parada.sum()
            if quantidade_sem_parada > 0:
                base = preencher_faltantes(base, "num_pitstops", mascara_sem_parada, 0, log_tratamento, "ausencia_estrutural", "regra_pitstop", "corrida possui dados de pitstop; ausencia do piloto considerada zero paradas")
            somar_preenchimentos(resumo_preenchimentos, "num_pitstops", quantidade_sem_parada)

            mascara_indisponivel = base["num_pitstops"].isna() & (base["pitstop_dado_disponivel"] == 0)
            quantidade_indisponivel = mascara_indisponivel.sum()
            if quantidade_indisponivel > 0:
                base = preencher_faltantes(base, "num_pitstops", mascara_indisponivel, 0, log_tratamento, "ausencia_dado", "sentinela_zero", "dados de pitstop indisponiveis para a corrida; flag pitstop_dado_disponivel preserva a ausencia")
            somar_preenchimentos(resumo_preenchimentos, "num_pitstops", quantidade_indisponivel)
        else:
            mascara = base["num_pitstops"].isna()
            quantidade = mascara.sum()
            if quantidade > 0:
                base = preencher_faltantes(base, "num_pitstops", mascara, 0, log_tratamento, "ausencia_estrutural", "regra_fixa", "sem registro de pitstop; considerado zero paradas")
            somar_preenchimentos(resumo_preenchimentos, "num_pitstops", quantidade)

    for coluna in ["tempo_total_pitstop", "tempo_medio_pitstop"]:
        if coluna not in base.columns:
            continue
        sem_parada = (base["num_pitstops"] == 0) & base[coluna].isna()
        quantidade = sem_parada.sum()
        if quantidade > 0:
            base = preencher_faltantes(base, coluna, sem_parada, 0, log_tratamento, "ausencia_estrutural", "regra_pitstop", "piloto nao realizou pitstop")
        somar_preenchimentos(resumo_preenchimentos, coluna, quantidade)

        zero_invalido = (base["num_pitstops"] > 0) & (base[coluna] == 0)
        base = marcar_invalido(base, coluna, zero_invalido, "tempo igual a zero apesar de existir pitstop", log_tratamento)

    # preserva se houve tempo registrado antes de substituir ausencias por zero
    for nome, flag in [("Q1_s", "registrou_tempo_q1"), ("Q2_s", "chegou_no_q2"), ("Q3_s", "chegou_no_q3")]:
        if nome in base.columns:
            base[flag] = base[nome].notna().astype(int)

    for coluna in COLUNAS_QUALIFYING_TEMPO:
        if coluna not in base.columns:
            continue
        mascara = base[coluna].isna()
        quantidade = mascara.sum()
        if quantidade > 0:
            base = preencher_faltantes(base, coluna, mascara, 0, log_tratamento, "ausencia_estrutural", "sentinela_zero", "zero nao representa tempo real; a coluna binaria correspondente indica se houve tempo registrado")
        somar_preenchimentos(resumo_preenchimentos, coluna, quantidade)

    # registra se os dados climaticos estavam originalmente disponiveis
    colunas_clima_existentes = [coluna for coluna in COLUNAS_CLIMA if coluna in base.columns]
    if colunas_clima_existentes:
        base["clima_dado_disponivel"] = base[colunas_clima_existentes].notna().all(axis=1).astype(int)

    if "qualifying_position" in base.columns and "grid_position" in base.columns:
        base["qualifying_position_imputada"] = base["qualifying_position"].isna().astype(int)

        mascara_grid = base["qualifying_position"].isna() & (base["grid_position"] > 0)
        quantidade_grid = mascara_grid.sum()
        if quantidade_grid > 0:
            valores_originais = base["qualifying_position"].copy()
            base.loc[mascara_grid, "qualifying_position"] = base.loc[mascara_grid, "grid_position"]
            registrar_tratamento(
                log_tratamento,
                base,
                base.index[mascara_grid],
                "qualifying_position",
                valores_originais,
                base["qualifying_position"],
                "aproximacao",
                "grid_position",
                "qualifying_position ausente; utilizada a posicao real de largada",
            )

        mascara_fallback = base["qualifying_position"].isna()
        quantidade_fallback = mascara_fallback.sum()
        if quantidade_fallback > 0:
            valores_originais = base["qualifying_position"].copy()
            tamanho_grid = base.groupby(["season", "round"])["driver_id"].transform("size")
            base.loc[mascara_fallback, "qualifying_position"] = tamanho_grid[mascara_fallback]
            registrar_tratamento(log_tratamento, base, base.index[mascara_fallback], "qualifying_position", valores_originais, base["qualifying_position"], "aproximacao", "ultima_posicao_grid", "qualifying_position e grid_position indisponiveis; utilizada a quantidade de pilotos da corrida")

        somar_preenchimentos(resumo_preenchimentos, "qualifying_position", quantidade_grid + quantidade_fallback)

        base["diferenca_grid_qualifying"] = 0.0
        validas = (base["grid_position"] > 0) & (base["qualifying_position"] > 0)
        base.loc[validas, "diferenca_grid_qualifying"] = base.loc[validas, "grid_position"] - base.loc[validas, "qualifying_position"]

    if "tire_compound_predominante" in base.columns:
        base["tire_compound_predominante"] = base["tire_compound_predominante"].replace(r"^\s*$", np.nan, regex=True)
        base["tire_compound_imputado"] = base["tire_compound_predominante"].isna().astype(int)

    # congela a base antes das imputacoes para nao usar valores imputados como historico
    base_historica = base.copy()
    corridas = base[["season", "round"]].drop_duplicates().sort_values(["season", "round"])

    # usa apenas corridas anteriores, priorizando circuito, temporada e historico geral
    for _, corrida in corridas.iterrows():
        season = corrida["season"]
        round_ = corrida["round"]
        mascara_corrida = (base["season"] == season) & (base["round"] == round_)
        historico = base_historica[mascara_corridas_anteriores(base_historica, season, round_)]
        circuitos = base.loc[mascara_corrida, "circuit_id"].dropna().unique()
        circuit_id = circuitos[0] if len(circuitos) > 0 else None

        for coluna in COLUNAS_MEDIANA_TEMPORAL:
            if coluna not in base.columns:
                continue

            mascara_faltando = mascara_corrida & base[coluna].isna()
            quantidade = mascara_faltando.sum()
            if quantidade == 0:
                continue

            valor = mediana_historica(historico, coluna, circuit_id, season)
            if pd.isna(valor):
                valor, metodo, observacao = 0, "sentinela_zero", "nao existia historico anterior disponivel para calcular a mediana"
            else:
                metodo, observacao = "mediana_historica", "valor calculado somente com corridas anteriores"

            if coluna in COLUNAS_CONTAGEM:
                valor = round(valor)

            base = preencher_faltantes(base, coluna, mascara_faltando, valor, log_tratamento, "imputacao_temporal", metodo, observacao)
            somar_preenchimentos(resumo_preenchimentos, coluna, quantidade)

    if "tire_compound_predominante" in base.columns:
        for _, corrida in corridas.iterrows():
            season = corrida["season"]
            round_ = corrida["round"]
            mascara_corrida = (base["season"] == season) & (base["round"] == round_)
            mascara_faltando = mascara_corrida & base["tire_compound_predominante"].isna()
            quantidade = mascara_faltando.sum()
            if quantidade == 0:
                continue

            historico = base_historica[mascara_corridas_anteriores(base_historica, season, round_)]
            circuitos = base.loc[mascara_corrida, "circuit_id"].dropna().unique()
            circuit_id = circuitos[0] if len(circuitos) > 0 else None
            valor = moda_historica(historico, "tire_compound_predominante", circuit_id, season)

            if pd.isna(valor):
                valor, metodo, observacao = "desconhecido", "sentinela_desconhecido", "nao existia historico anterior de composto de pneu"
            else:
                metodo, observacao = "moda_historica", "composto preenchido somente com corridas anteriores"

            base = preencher_faltantes(base, "tire_compound_predominante", mascara_faltando, valor, log_tratamento, "imputacao_temporal", metodo, observacao)
            somar_preenchimentos(resumo_preenchimentos, "tire_compound_predominante", quantidade)

    if "choveu" in base.columns:
        for _, corrida in corridas.iterrows():
            season = corrida["season"]
            round_ = corrida["round"]
            mascara_corrida = (base["season"] == season) & (base["round"] == round_)
            mascara_faltando = mascara_corrida & base["choveu"].isna()
            quantidade = mascara_faltando.sum()
            if quantidade == 0:
                continue

            historico = base_historica[mascara_corridas_anteriores(base_historica, season, round_)]
            circuitos = base.loc[mascara_corrida, "circuit_id"].dropna().unique()
            circuit_id = circuitos[0] if len(circuitos) > 0 else None
            valor = moda_historica(historico, "choveu", circuit_id, season)

            if pd.isna(valor):
                valor, metodo, observacao = 0, "sentinela_zero", "nao existia historico anterior; clima_dado_disponivel preserva a ausencia original"
            else:
                metodo, observacao = "moda_historica", "valor preenchido somente com corridas anteriores; clima_dado_disponivel preserva a ausencia original"

            base = preencher_faltantes(base, "choveu", mascara_faltando, valor, log_tratamento, "imputacao_temporal", metodo, observacao)
            somar_preenchimentos(resumo_preenchimentos, "choveu", quantidade)

    # garante que a base enviada à modelagem não tenha valores ausentes
    faltando = base.isna().sum()
    faltando = faltando[faltando > 0]

    if len(faltando) > 0:
        print("NaN depois do tratamento:")
        print(faltando)
        raise ValueError("Ainda existem valores NaN na base apos o tratamento.")
    print("NaN depois:", 0)
    print("Valores preenchidos:", sum(resumo_preenchimentos.values()))

    base.to_csv(ARQUIVO_SAIDA, index=False)

    log_df = pd.DataFrame(log_tratamento, columns=COLUNAS_LOG)
    log_df.to_csv(ARQUIVO_LOG_TRATAMENTO, index=False)

    print("Alteracoes registradas no log:", len(log_df))
    print("Base tratada salva em:", ARQUIVO_SAIDA)
    print("Log salvo em:", ARQUIVO_LOG_TRATAMENTO)
    print("Linhas:", len(base), "| Colunas:", len(base.columns))


if __name__ == "__main__":
    tratar_base()
