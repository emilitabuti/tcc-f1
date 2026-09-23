#define quais colunas podem entrar no modelo e quais causariam vazamento

# coluna que o modelo vai tentar prever
COLUNA_ALVO = "finish_position"

COLUNAS_PRE_CORRIDA = [
    "season",
    "round",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "circuit_name",
    "altitude_m",
    "corners",
    "length_km",
    "circuit_type",
    "lat",
    "long",
    "country",
    "code",
    "given_name",
    "family_name",
    "date_of_birth",
    "nationality",
    "grid_position",
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

COLUNAS_POS_CORRIDA = [
    "points",
    "laps",
    "num_pitstops",
    "tempo_total_pitstop",
    "tempo_medio_pitstop",
    "pitstop_dado_disponivel",
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_num_voltas",
    "fastf1_num_stints",
    "fastf1_tyre_life_media",
    "tire_compound_predominante",
    "tire_compound_imputado",
    "temp_ar_media",
    "temp_pista_media",
    "umidade_media",
    "vento_media",
    "choveu",
    "clima_dado_disponivel",
    "status",
    "status_categoria",
    "situacao_tempo_volta",
    "situacao_pitstop",
]

def validar_sem_vazamento(colunas_modelo):
    # verifica se alguma coluna pos-corrida entrou no modelo
    proibidas = sorted(set(colunas_modelo) & set(COLUNAS_POS_CORRIDA))
    if proibidas:
        raise ValueError(
            "Colunas pos-corrida nao podem entrar diretamente no modelo: "
            + ", ".join(proibidas)
        )
