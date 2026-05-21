# Tratamento de DNFs

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
