# Análise de vendas de videogames e avaliações do Metacritic

Projeto em Python para análise exploratória das vendas registradas de videogames e avaliação do uso de notas de críticos e jogadores na estimativa de **vendas acumuladas** por meio de Machine Learning.

> **Escopo:** este trabalho é um projeto de análise de dados acompanhado de relatório. Não é um sistema comprovado de previsão das vendas na primeira semana de lançamento.

## 0. Links
CANVAS: `canvas/Canvas_Projeto_Vendas_Videogames_Metacritic.pdf`
Overleaf - Prévia do Artigo: `https://www.overleaf.com/project/6b018c08ad13f930d10b3b1f#`

## 1. Perguntas de pesquisa

1. Quais gêneros de videogames venderam mais?
2. Quais plataformas concentraram mais vendas?
3. Quais editoras tiveram maior participação nas vendas registradas?
4. Como as vendas variaram conforme o ano de lançamento?
5. É possível utilizar Machine Learning para estimar as vendas de um jogo?
6. As notas do Metacritic e as avaliações dos jogadores ajudam a estimar o desempenho comercial?
7. É possível utilizar notas da semana de lançamento e informações de jogos anteriores para prever vendas?

**Limitação da pergunta 7:** os datasets não registram necessariamente as notas **na primeira semana**. Os experimentos usam notas disponíveis nas bases e vendas acumuladas; por isso, não validam uma previsão operacional feita na semana de lançamento. A relação encontrada é preditiva/associativa, não causal.

## 2. Fontes de dados

Os arquivos originais são mantidos na pasta `dados/`, em formato ZIP:

- `Metacritic Video Games Reviews (up to sep 2025)(1).zip` (ou variação do mesmo nome), contendo `metacritic_dataset_clean.csv`.
- `Video Game Sales & Industry Data (1980 - 2024).zip`, contendo `Video_Games_Sales_Cleaned.csv`.

A preparação usa as versões *clean* dessas fontes. Os nomes dos arquivos ZIP podem variar, mas `limpeza_dados.py` procura trechos específicos dos nomes.

**Unidade de observação:** registros de jogos por plataforma, conforme as fontes. A variável `total_sales` está expressa em **milhões de unidades**.

## 3. Estrutura do projeto

```text
Projeto_ML_Vendas_Metacritic/
├── dados/                           # ZIPs originais
├── resultados/                      # CSVs de preparação, métricas e gráficos
├── modelos/                         # Reservado para modelos serializados (opcional)
├── analise_inicial.py
├── limpeza_dados.py
├── integracao_dados.py
├── verificar_integracao.py
├── analise_notas_vendas.py
├── analise_comercial.py             # Análise exploratória comercial
├── modelo_regressao.py
├── modelo_random_forest.py
├── modelo_random_forest_log.py
├── analise_erros.py
├── analise_faixas_vendas.py
├── comparar_modelos_faixas.py
├── importancia_variaveis.py
├── modelo_sem_ano.py
├── comparar_importancia_notas.py
├── validacao_temporal.py
├── validacao_referencia.py
├── distribuicao_periodos.py
├── auditoria_integracao.py
├── teste_sensibilidade_integracao.py
├── consolidar_resultados.py
└── README.md
```

Alguns arquivos são experimentos complementares, não etapas obrigatórias para gerar a base integrada. O arquivo `analise_comercial.py` foi acrescentado posteriormente ao pacote original do projeto.

## 4. Instalação e execução

Recomendado: Python 3.12 e ambiente virtual. Na pasta do projeto, no PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

Se a política do PowerShell impedir a ativação, utilize diretamente `\.venv\Scripts\python.exe` nos comandos seguintes.

### Ordem principal

```powershell
python .\analise_inicial.py
python .\limpeza_dados.py
python .\integracao_dados.py
python .\verificar_integracao.py
python .\analise_comercial.py
python .\analise_notas_vendas.py
python .\modelo_regressao.py
python .\modelo_random_forest.py
python .\modelo_random_forest_log.py
python .\validacao_temporal.py
python .\validacao_referencia.py
```

**Importante:** `validacao_referencia.py` lê `resultados/validacao_temporal.csv`; execute `validacao_temporal.py` antes. Os demais scripts complementares podem ser executados depois que as bases e previsões correspondentes estiverem disponíveis. `consolidar_resultados.py` consulta arquivos de métricas previamente gerados.

### Experimentos complementares

- `analise_erros.py` e `analise_faixas_vendas.py`: examinam o **Random Forest original**, não o modelo logarítmico.
- `comparar_modelos_faixas.py`: compara erros por faixa de vendas.
- `importancia_variaveis.py`: importância por permutação no Random Forest Log.
- `modelo_sem_ano.py`: teste de exclusão do ano.
- `comparar_importancia_notas.py`: comparação sem notas, só Metascore e com ambas as notas.
- `distribuicao_periodos.py`: distribuição das vendas nos períodos temporais.
- `auditoria_integracao.py`: divergências entre anos das fontes.
- `teste_sensibilidade_integracao.py`: compara a base completa com o subconjunto de anos coincidentes.

## 5. Bases produzidas

- `resultados/metacritic_preparado.csv`: 21.913 registros após remoção de duplicatas exatas.
- `resultados/vendas_preparadas.csv`: 18.828 registros para análises comerciais.
- `resultados/jogos_integrados.csv`: 6.129 correspondências únicas de nome normalizado, plataforma e ano com diferença máxima de um ano.
- `resultados/correspondencias_ambiguas.csv`: 48 correspondências candidatas excluídas da base integrada.

A integração não garante correspondência perfeita: títulos, plataformas, relançamentos e anos podem apresentar inconsistências.

## 6. Resultados principais

### Análise comercial — base de vendas preparada

- Vendas registradas: **6.598,57 milhões de unidades**.
- Gênero líder: **Sports**, com **1.186,70 milhões** (17,98%).
- Plataforma líder: **PS2**, com **1.025,36 milhões** (15,54%).
- Editora líder: **Activision**, com **722,77 milhões** (10,95%).
- Ano de lançamento com maior soma de vendas acumuladas: **2008**, com **538,09 milhões**.

Saídas: `ranking_generos.csv/.png`, `ranking_plataformas.csv/.png`, `ranking_editoras.csv/.png` e `evolucao_vendas_anuais.csv/.png`.

**Interpretação temporal:** a série agrupa vendas **acumuladas** pelo ano de lançamento, e não vendas realizadas em cada ano. A cobertura é muito reduzida em 2019 e 2020; não se deve inferir queda do mercado global a partir desse gráfico.

### Notas e vendas — base integrada

- Correlação de Pearson entre Metascore e vendas: **0,2903**.
- Correlação de Pearson entre nota dos usuários e vendas: **0,0742**.

### Previsão — divisão temporal principal

Treinamento com jogos até 2015; teste com 916 registros a partir de 2016.

| Modelo | MAE (milhões) | RMSE (milhões) | R² |
|---|---:|---:|---:|
| Referência (mediana) | 0,4426 | 1,1967 | -0,0242 |
| Regressão linear | 0,7295 | 1,1393 | 0,0717 |
| Random Forest | 0,5629 | 1,0067 | 0,2752 |
| Random Forest Log | 0,4742 | 0,9472 | 0,3583 |

O **Random Forest Log** obteve os melhores RMSE e R² nessa comparação, mas **a referência por mediana teve o menor MAE**. Não há um único vencedor em todas as métricas.

Comparação das características no Random Forest Log:

| Configuração | MAE | RMSE | R² |
|---|---:|---:|---:|
| Sem notas | 0,5209 | 1,0675 | 0,1850 |
| Apenas Metascore | 0,4836 | 0,9834 | 0,3084 |
| Metascore e nota dos usuários | 0,4742 | 0,9472 | 0,3583 |

As notas agregaram informação útil à previsão **retrospectiva**, sem demonstrar que avaliações mais altas causam maiores vendas.

## 7. Validação temporal

Foram realizados testes com treinamento histórico crescente:

| Treino até | Teste | Registros de teste | MAE sem notas | MAE com ambas as notas | MAE referência |
|---|---|---:|---:|---:|---:|
| 2012 | 2013–2014 | 423 | 0,7455 | 0,6291 | 0,7706 |
| 2014 | 2015–2016 | 463 | 0,5973 | 0,5805 | 0,5306 |
| 2016 | 2017–2018 | 627 | 0,4727 | 0,4070 | 0,4177 |

A configuração com ambas as notas melhorou o MAE em relação à configuração sem notas nos três períodos, mas superou a referência simples em apenas dois deles.

## 8. Limitações e cuidados

1. **Notas não históricas:** não há garantia de que `metascore` e `user_score` sejam os valores disponíveis na semana de lançamento. Há risco de vazamento temporal para um cenário de previsão real no lançamento.
2. **Vendas acumuladas:** jogos de idades diferentes tiveram tempos diferentes para acumular vendas; a comparação não equivale a vendas de lançamento.
3. **Zeros:** há 1.323 registros com `total_sales = 0` na base comercial e 323 na integrada; o significado desses zeros não foi confirmado na fonte.
4. **Cobertura desigual:** os dados têm pouca representação nos anos finais, apesar do intervalo anunciado no nome do dataset.
5. **Integração:** 367 dos 6.129 registros integrados apresentam diferença de um ano entre as fontes. A correspondência automática pode conter falsos positivos.
6. **Importância do ano:** importância por permutação igual a zero no teste temporal não significa ausência de valor preditivo; árvores treinadas até 2015 não extrapolam o ano como uma regressão linear.
7. **Editoras:** rótulos como `Electronic Arts` e `EA Sports` foram mantidos separados, conforme a fonte; os rankings não representam necessariamente grupos empresariais consolidados.
8. **Interpretação:** correlação e importância preditiva não demonstram causalidade.

## 9. Gráficos e arquivos da entrega

A pasta `resultados/` concentra tabelas, métricas e figuras. Entre os gráficos principais:

- `ranking_generos.png`
- `ranking_plataformas.png`
- `ranking_editoras.png`
- `evolucao_vendas_anuais.png`
- `metascore_vs_vendas.png`
- `usuarios_vs_vendas.png`
- `random_forest_reais_vs_previstas.png` (**modelo original**)
- `random_forest_log_reais_vs_previstas.png` (**modelo logarítmico**, acrescentado posteriormente)
- `importancia_variaveis.png`

Os modelos são treinados pelos scripts. A existência de uma pasta `modelos/` não significa que exista um modelo serializado pronto para uso.

## 10. Reprodutibilidade

- Manter os dois ZIPs originais em `dados/`.
- Executar os scripts a partir da pasta principal do projeto.
- Não alterar manualmente os CSVs produzidos entre as etapas.
- Preservar os valores de `random_state` definidos nos experimentos.
- Registrar as versões das dependências em `requirements.txt` antes da entrega final.
- Não incluir a pasta `.venv/` no pacote de entrega.

Este README descreve os experimentos e resultados realizados; não substitui a verificação da execução integral em um ambiente novo.
