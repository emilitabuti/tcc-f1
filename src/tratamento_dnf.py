from pathlib import Path
import pandas as pd
import numpy as np


#Tratamento de DNFs
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)



# Arquivos de entrada
INPUT_FILE_2018_2024 = PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2024.csv"
INPUT_FILE_2018_2025 = PROCESSED_DIR / "historico_ergast_fastf1_limpo_2018_2025.csv"


# Arquivos de saída
OUTPUT_CLASSIFICADO_2018_2024 = PROCESSED_DIR / "historico_dnf_classificado_2018_2024.csv"
OUTPUT_DNF_EXCLUDED_2018_2024 = PROCESSED_DIR / "historico_dnf_excluded_2018_2024.csv"

OUTPUT_CLASSIFICADO_2018_2025 = PROCESSED_DIR / "historico_dnf_classificado_2018_2025.csv"
OUTPUT_DNF_EXCLUDED_2018_2025 = PROCESSED_DIR / "historico_dnf_excluded_2018_2025.csv"

REPORT_FILE = PROCESSED_DIR / "relatorio_02_tratamento_dnf.txt"
METHODOLOGY_FILE = DOCS_DIR / "metodologia_tratamento_dnf.md"


# Status considerados como corrida concluída
STATUS_CLASSIFICADO_EXATOS = {
    "finished",
    "lapped",
}

# Exemplo de status de classificados com voltas atrás:
# +1 Lap, +2 Laps, +3 Laps etc.



# Status relacionados a erro/incidente do piloto
DNF_PILOTO_KEYWORDS = [
    "accident",
    "collision",
    "spun off",
    "spun-off",
    "spin",
    "crash",
    "damage",
]


# Status relacionados a falhas do carro
DNF_CARRO_KEYWORDS = [
    "engine",
    "gearbox",
    "transmission",
    "clutch",
    "hydraulics",
    "electrical",
    "electronics",
    "ers",
    "power unit",
    "power loss",
    "brakes",
    "brake",
    "suspension",
    "steering",
    "radiator",
    "oil",
    "oil leak",
    "water pressure",
    "water leak",
    "water pump",
    "cooling system",
    "fuel",
    "fuel pressure",
    "fuel pump",
    "fuel leak",
    "out of fuel",
    "turbo",
    "exhaust",
    "mechanical",
    "overheating",
    "puncture",
    "tyre",
    "wheel",
    "wheel nut",
    "driveshaft",
    "differential",
    "battery",
    "undertray",
    "front wing",
    "rear wing",
    "vibrations",
]


# Outros casos de não conclusão
DNF_OUTROS_KEYWORDS = [
    "did not start",
    "dns",
    "withdrew",
    "withdrawn",
    "illness",
    "excluded",
    "disqualified",
    "retired",
    "not classified",
]


# Funções auxiliares
def validar_colunas(df, colunas_obrigatorias, nome_base):
    #Valida se as colunas obrigatórias existem no DataFrame.
    
    colunas_ausentes = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"As seguintes colunas estão ausentes em {nome_base}: "
            f"{colunas_ausentes}"
        )


def repo_relative(path):
    # Registra caminhos portáveis no relatório, independentemente da máquina.
    return path.relative_to(BASE_DIR).as_posix()


def normalizar_status(status):
    #Padroniza o texto do status para facilitar a classificação.

    if pd.isna(status):
        return ""

    return str(status).strip().lower()


def is_status_classificado(status_normalizado):
    #Identifica pilotos que terminaram a corrida ou foram classificados, mesmo com uma ou mais voltas atrás.
    if status_normalizado in STATUS_CLASSIFICADO_EXATOS:
        return True

    # Exemplo: +1 Lap, +2 Laps, +10 Laps
    if status_normalizado.startswith("+") and "lap" in status_normalizado:
        return True

    return False


def contem_keyword(status_normalizado, keywords):
    #Verifica se algum termo da lista aparece no status
    return any(keyword in status_normalizado for keyword in keywords)


def classificar_dnf(status):
    #Classifica o status da corrida em:
    # classificado 
    # dnf_piloto
    # dnf_carro
    # dnf_outros
    status_normalizado = normalizar_status(status)

    if status_normalizado == "":
        return "dnf_outros"

    if is_status_classificado(status_normalizado):
        return "classificado"

    if contem_keyword(status_normalizado, DNF_PILOTO_KEYWORDS):
        return "dnf_piloto"

    if contem_keyword(status_normalizado, DNF_CARRO_KEYWORDS):
        return "dnf_carro"

    if contem_keyword(status_normalizado, DNF_OUTROS_KEYWORDS):
        return "dnf_outros"

    # Qualquer outro status não classificado como "Finished" ou "+x Laps"
    # será tratado como DNF indefinido/outros.
    return "dnf_outros"


def aplicar_tratamento_dnf(df, nome_base):
    #Aplica a classificação de DNF e cria a base DNF Excluded.
    df = df.copy()

    validar_colunas(
        df,
        ["season", "round", "driver_id", "status"],
        nome_base
    )

    df["status_normalizado"] = df["status"].apply(normalizar_status)
    df["dnf_categoria"] = df["status"].apply(classificar_dnf)

    df["is_dnf"] = df["dnf_categoria"].isin(
        ["dnf_piloto", "dnf_carro", "dnf_outros"]
    )
    df["dnf_flag"] = df["is_dnf"].astype(int)
    df["dnf_driver_flag"] = (df["dnf_categoria"] == "dnf_piloto").astype(int)
    df["dnf_car_flag"] = (df["dnf_categoria"] == "dnf_carro").astype(int)
    df["dnf_other_flag"] = (df["dnf_categoria"] == "dnf_outros").astype(int)

    df_dnf_excluded = df[df["dnf_categoria"] == "classificado"].copy()

    return df, df_dnf_excluded


def resumo_dnf(df):
    #Gera resumo da quantidade de registros por categoria DNF.
    return df["dnf_categoria"].value_counts().sort_index()


def salvar_metodologia():
    #Salva a decisão metodológica em Markdown.
    texto = """# Tratamento de DNFs

## Definição

DNF significa *Did Not Finish*, ou seja, pilotos que participaram de uma corrida, mas não a concluíram por algum motivo, como acidente, colisão, erro de pilotagem, falha mecânica ou outro evento externo.

## Variante adotada

Neste trabalho, foi adotada a variante **DNF Excluded**.

Isso significa que os registros de pilotos que não concluíram a corrida foram identificados, classificados e posteriormente removidos da base utilizada para treinamento do modelo.

A escolha foi feita para reduzir ruídos no aprendizado do modelo, pois abandonos podem distorcer a posição final de um piloto. Por exemplo, um piloto poderia apresentar bom desempenho durante a corrida, mas abandonar por falha mecânica e terminar nas últimas posições. Nesse caso, a posição final não representa necessariamente seu desempenho competitivo.

## Classificação dos DNFs

Antes da exclusão, os DNFs foram classificados em três grupos:

### DNF de piloto

Inclui abandonos relacionados a acidentes, colisões ou erros de pilotagem.

Exemplos:

- Collision
- Accident
- Spun off
- Crash

### DNF de carro

Inclui abandonos relacionados a falhas mecânicas ou técnicas do carro.

Exemplos:

- Engine
- Gearbox
- ERS
- Hydraulics
- Brakes
- Suspension
- Power Unit

### DNF outros

Inclui casos que não se enquadram diretamente como erro de piloto ou falha do carro.

Exemplos:

- Did not start
- Withdrew
- Illness
- Disqualified
- Not classified

## Decisão metodológica

A base final utilizada para treinamento segue a abordagem **DNF Excluded**, alinhada ao benchmark RAPM com MAE de 2,3 posições. Dessa forma, apenas pilotos classificados, incluindo aqueles marcados como `Finished`, `Lapped` ou com status de voltas atrás, como `+1 Lap` e `+2 Laps`, são mantidos na base final.

O status `Lapped` é mantido como classificado porque representa pilotos oficialmente classificados com volta(s) atrás, não abandono de corrida.

Os registros de DNF são preservados em uma base intermediária classificada, permitindo análise exploratória e rastreabilidade da decisão metodológica.
"""

    with open(METHODOLOGY_FILE, "w", encoding="utf-8") as f:
        f.write(texto)



# 1. Carregar arquivos da etapa 01
historico_2018_2024 = pd.read_csv(INPUT_FILE_2018_2024)
historico_2018_2025 = pd.read_csv(INPUT_FILE_2018_2025)

print("Arquivos carregados com sucesso.")
print(f"Histórico 2018-2024: {historico_2018_2024.shape}")
print(f"Histórico 2018-2025: {historico_2018_2025.shape}")


# 2. Aplicar tratamento de DNF
classificado_2018_2024, dnf_excluded_2018_2024 = aplicar_tratamento_dnf(
    historico_2018_2024,
    "historico_ergast_fastf1_limpo_2018_2024.csv"
)

classificado_2018_2025, dnf_excluded_2018_2025 = aplicar_tratamento_dnf(
    historico_2018_2025,
    "historico_ergast_fastf1_limpo_2018_2025.csv"
)


# 3. Exibir resumo no terminal
print("\nResumo DNF - 2018 a 2024:")
print(resumo_dnf(classificado_2018_2024))

print("\nResumo DNF - 2018 a 2025:")
print(resumo_dnf(classificado_2018_2025))

print("\nLinhas após DNF Excluded:")
print(f"2018-2024: {len(dnf_excluded_2018_2024)}")
print(f"2018-2025: {len(dnf_excluded_2018_2025)}")


# 4. Salvar bases finais
classificado_2018_2024.to_csv(
    OUTPUT_CLASSIFICADO_2018_2024,
    index=False,
    encoding="utf-8-sig"
)

dnf_excluded_2018_2024.to_csv(
    OUTPUT_DNF_EXCLUDED_2018_2024,
    index=False,
    encoding="utf-8-sig"
)

classificado_2018_2025.to_csv(
    OUTPUT_CLASSIFICADO_2018_2025,
    index=False,
    encoding="utf-8-sig"
)

dnf_excluded_2018_2025.to_csv(
    OUTPUT_DNF_EXCLUDED_2018_2025,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos salvos com sucesso:")
print(OUTPUT_CLASSIFICADO_2018_2024)
print(OUTPUT_DNF_EXCLUDED_2018_2024)
print(OUTPUT_CLASSIFICADO_2018_2025)
print(OUTPUT_DNF_EXCLUDED_2018_2025)


# 5. Salvar relatório
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO - 02 TRATAMENTO DE DNFs\n")
    f.write("=" * 60 + "\n\n")

    f.write("VARIANTE ADOTADA\n")
    f.write("-" * 60 + "\n")
    f.write("DNF Excluded\n\n")

    f.write("JUSTIFICATIVA\n")
    f.write("-" * 60 + "\n")
    f.write(
        "A variante DNF Excluded foi adotada para reduzir ruídos na variável "
        "de resultado, removendo casos em que a posição final foi impactada "
        "por abandono, acidente, falha mecânica ou outro evento externo. "
        "A decisão está alinhada ao benchmark RAPM com MAE de 2,3 posições.\n\n"
    )

    f.write("CRITÉRIOS DE CLASSIFICAÇÃO\n")
    f.write("-" * 60 + "\n")
    f.write("classificado: Finished, Lapped ou status de voltas atrás, como +1 Lap, +2 Laps.\n")
    f.write("dnf_piloto: acidente, colisão, rodada ou erro/incidente do piloto.\n")
    f.write("dnf_carro: falha mecânica, motor, câmbio, ERS, freios, suspensão etc.\n")
    f.write("dnf_outros: DNS, retirada, doença, desclassificação ou casos indefinidos.\n\n")

    f.write("RESUMO 2018-2024\n")
    f.write("-" * 60 + "\n")
    f.write(f"Linhas antes do DNF Excluded: {len(classificado_2018_2024)}\n")
    f.write(f"Linhas após DNF Excluded: {len(dnf_excluded_2018_2024)}\n")
    f.write("\nCategorias:\n")
    f.write(str(resumo_dnf(classificado_2018_2024)))
    f.write("\n\n")

    f.write("RESUMO 2018-2025\n")
    f.write("-" * 60 + "\n")
    f.write(f"Linhas antes do DNF Excluded: {len(classificado_2018_2025)}\n")
    f.write(f"Linhas após DNF Excluded: {len(dnf_excluded_2018_2025)}\n")
    f.write("\nCategorias:\n")
    f.write(str(resumo_dnf(classificado_2018_2025)))
    f.write("\n\n")

    f.write("ARQUIVOS GERADOS\n")
    f.write("-" * 60 + "\n")
    f.write(f"{repo_relative(OUTPUT_CLASSIFICADO_2018_2024)}\n")
    f.write(f"{repo_relative(OUTPUT_DNF_EXCLUDED_2018_2024)}\n")
    f.write(f"{repo_relative(OUTPUT_CLASSIFICADO_2018_2025)}\n")
    f.write(f"{repo_relative(OUTPUT_DNF_EXCLUDED_2018_2025)}\n")

print("\nRelatório salvo em:")
print(REPORT_FILE)


# 6. Salvar documentação metodológica
salvar_metodologia()

print("\nDocumentação metodológica salva em:")
print(METHODOLOGY_FILE)

print("\nEtapa 02 finalizada com sucesso.")
