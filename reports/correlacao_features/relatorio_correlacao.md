# Relatório de Correlação das Features

Arquivo de entrada: /home/emili-tabuti/Documentos/projects/tcc-f1/data/processed/dataset_features_final_2018_2025.csv
Arquivo tratado gerado: /home/emili-tabuti/Documentos/projects/tcc-f1/data/processed/dataset_features_final_2018_2025_sem_nan.csv
Linhas do dataset: 2943
Colunas do dataset: 130
Features finais analisadas: 21

Tratamento de NaN
Não havia NaN no dataset de entrada.

Após o tratamento, não restaram valores NaN.

Análise de correlação
Foi calculada a matriz de correlação de Pearson entre as features finais canônicas.
Foram destacados pares com correlação absoluta maior que 0.85.
Total de pares com correlação alta: 3

Correlação com o target
- qualifying_position: r=0.7717
- grid_position: r=0.7531
- recent_form_5: r=0.7104
- recent_form_3: r=0.6950
- constructor_coef_rapm: r=-0.6831
- driver_constructor_synergy: r=-0.6629
- driver_coef_rapm: r=-0.5992
- constructor_wins_total: r=-0.4206
- driver_wins_total: r=-0.3499
- driver_experience: r=-0.2072

Principais pares encontrados:
- recent_form_5 x recent_form_3: r=0.9874 | decisão: Revisar manualmente antes de remover.
- grid_position x qualifying_position: r=0.9616 | decisão: Revisar manualmente antes de remover.
- recent_form_5 x driver_constructor_synergy: r=-0.8743 | decisão: Revisar manualmente antes de remover.

Observação metodológica
A análise foi limitada às features finais de docs/lista_features_modelo.md. Colunas auxiliares, z-score/minmax, one-hot intermediário, flags de auditoria e telemetria FastF1 não entram nesta matriz de decisão para modelos tree-based.