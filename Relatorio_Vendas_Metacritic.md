# Análise de Vendas de Videogames e Avaliações do Metacritic: Uma Abordagem com Ciência de Dados e Machine Learning {#analise-de-vendas-de-videogames-e-avaliacoes-do-metacritic-uma-abordagem-com-ciencia-de-dados-e-machine-learning}

**Eduardo Andrade Correia Lima¹, Leonardo da Fonseca Bastos¹**

¹ Universidade Tiradentes (UNIT)\
Curso de Ciência de Dados e Inteligência Artificial

**Professor:** Yuri Faro Dantas de Sant'Anna

**E-mails:** acrianolima@gmail.com, leofbastos@gmail.com

## Abstract

This study investigates video game sales patterns and examines whether
Metacritic ratings contribute to sales prediction using Data Science and
Machine Learning techniques. Two publicly available datasets were
processed, resulting in 18,828 video game sales records and 21,913
Metacritic records after data preparation. The datasets were integrated
using normalized game titles, platforms, and release years, producing
6,129 matched game-platform records. Exploratory analyses examined sales
distributions across genres, platforms, publishers, and release years.
Linear Regression and Random Forest models were evaluated using a
chronological training and testing split. Among the trained models,
Random Forest with a logarithmic target transformation achieved the best
RMSE (0.9472 million units) and R² (0.3583). However, a median-based
baseline achieved a lower MAE. Additional experiments suggested that
review scores contain useful predictive information. Because review
publication dates were not verified, the results cannot establish
predictive performance during the first week after release.

**Keywords:** Video Game Sales, Metacritic, Machine Learning, Random
Forest, Data Science.

## Resumo

Este trabalho investiga os padrões de vendas de videogames e avalia a
contribuição das notas do Metacritic para a previsão de vendas por meio
de técnicas de Ciência de Dados e Machine Learning. Foram utilizados
dois conjuntos de dados públicos que, após a preparação, apresentaram
18.828 registros de vendas e 21.913 registros do Metacritic. A
integração por nome normalizado, plataforma e ano de lançamento resultou
em 6.129 registros associados de jogos e plataformas. A análise
exploratória investigou a distribuição das vendas entre gêneros,
plataformas, editoras e anos de lançamento. Foram avaliados modelos de
Regressão Linear e Random Forest utilizando uma divisão cronológica
entre treinamento e teste. Entre os modelos treinados, o Random Forest
com transformação logarítmica apresentou o melhor RMSE (0,9472 milhão de
unidades) e R² (0,3583). Entretanto, o modelo de referência baseado na
mediana apresentou menor MAE. Experimentos complementares indicaram que
as avaliações fornecem informações úteis para as previsões. Como não
foram verificadas as datas de publicação das avaliações, os resultados
não permitem estabelecer a capacidade de previsão durante a primeira
semana após o lançamento.

**Palavras-chave:** Vendas de Videogames, Metacritic, Aprendizado de
Máquina, Random Forest, Ciência de Dados.

## 1. Introdução {#introducao}

A indústria de videogames apresenta uma ampla diversidade de produtos,
plataformas, gêneros e modelos de comercialização. Nesse cenário,
compreender os fatores associados ao desempenho comercial dos jogos
representa uma oportunidade de aplicação de técnicas de Ciência de
Dados, permitindo explorar informações históricas e identificar padrões
nas vendas registradas.

Entre as características que podem estar associadas ao desempenho
comercial de um videogame estão seu gênero, a plataforma de lançamento,
a editora responsável e a recepção da crítica especializada e dos
jogadores. O Metacritic reúne avaliações de diferentes fontes,
oferecendo indicadores que podem ser utilizados em estudos quantitativos
sobre a relação entre avaliações e resultados comerciais.

Entretanto, a existência de uma associação estatística entre avaliações
e vendas não significa necessariamente que avaliações positivas
provoquem maiores vendas. Outros fatores, como investimentos em
divulgação, popularidade de franquias, disponibilidade nas plataformas e
características do público consumidor, também podem influenciar os
resultados comerciais.

Diante desse contexto, este estudo investiga padrões de vendas de
videogames e examina a contribuição das avaliações do Metacritic para
modelos de previsão de vendas. Para isso, são utilizados dois conjuntos
de dados públicos obtidos no Kaggle, contendo informações comerciais e
avaliações de jogos.

O desenvolvimento do projeto compreendeu a preparação dos dados, a
padronização de nomes e plataformas, a integração das bases, a análise
exploratória, a construção de modelos de Machine Learning e a realização
de testes de avaliação e validação temporal.

A investigação foi orientada pelas seguintes perguntas de pesquisa:

1.  Quais gêneros de videogames apresentaram maiores vendas registradas?

2.  Quais plataformas concentraram os maiores volumes de vendas?

3.  Quais editoras apresentaram maior participação nas vendas?

4.  Como as vendas acumuladas variaram conforme o ano de lançamento dos
    jogos?

5.  É possível utilizar modelos de Machine Learning para estimar as
    vendas de videogames?

6.  As avaliações do Metacritic e dos usuários contribuem para melhorar
    as previsões?

7.  É possível realizar previsões utilizando avaliações disponíveis na
    primeira semana após o lançamento?

A última pergunta exige uma consideração metodológica importante. Embora
os datasets contenham avaliações e informações sobre vendas, não
apresentam evidências suficientes para estabelecer quando cada nota
estava disponível em relação ao lançamento do jogo. Portanto, os
experimentos realizados permitem avaliar associações retrospectivas e
desempenho preditivo em registros históricos, mas não comprovar
previsões baseadas exclusivamente em informações disponíveis na primeira
semana.

O objetivo geral deste trabalho é analisar padrões de vendas de
videogames e avaliar a contribuição das notas do Metacritic para modelos
de previsão comercial utilizando técnicas de Ciência de Dados e Machine
Learning.

Como objetivos específicos, busca-se identificar os gêneros, plataformas
e editoras com maiores vendas registradas; examinar a distribuição das
vendas por ano de lançamento; comparar diferentes modelos preditivos;
investigar a importância das avaliações; e analisar a estabilidade dos
resultados por meio de experimentos complementares.

O restante do artigo está organizado da seguinte forma: a Seção 2
apresenta a fundamentação teórica; a Seção 3 descreve os materiais e
métodos; a Seção 4 apresenta os resultados e sua discussão; a Seção 5
aborda as limitações do estudo; e a Seção 6 reúne as conclusões.

## 2. Fundamentação Teórica {#fundamentacao-teorica}

### 2.1 Ciência de Dados e Análise Exploratória {#ciencia-de-dados-e-analise-exploratoria}

A Ciência de Dados constitui uma área interdisciplinar que combina
conhecimentos de estatística, computação e análise de informações para
extrair conhecimento de conjuntos de dados. Seu desenvolvimento envolve
diferentes atividades, incluindo coleta, preparação, transformação,
exploração, modelagem e interpretação dos resultados.

Segundo Provost e Fawcett (2013), a análise de dados e os métodos
preditivos permitem identificar padrões relevantes para apoiar processos
de tomada de decisão. Entretanto, a qualidade das conclusões depende
diretamente da adequação dos dados, das técnicas utilizadas e da
interpretação dos resultados.

A Análise Exploratória de Dados (AED) representa uma etapa fundamental
desse processo. Conforme Tukey (1977), a exploração dos dados permite
investigar distribuições, identificar valores atípicos, reconhecer
possíveis relações entre variáveis e formular hipóteses que poderão ser
examinadas posteriormente.

No contexto deste estudo, a análise exploratória foi utilizada para
investigar a distribuição das vendas de videogames entre diferentes
gêneros, plataformas, editoras e anos de lançamento. Também foram
examinadas as relações entre as vendas registradas e as avaliações da
crítica especializada e dos usuários.

### 2.2 Machine Learning e Modelagem Preditiva

Machine Learning, ou Aprendizado de Máquina, corresponde a um conjunto
de métodos computacionais que permitem identificar padrões a partir de
dados e utilizá-los para realizar previsões ou classificações.

De acordo com James et al. (2023), os métodos de aprendizado
supervisionado utilizam observações contendo variáveis explicativas e
uma variável de interesse para estimar relações e produzir previsões.

Neste projeto, o problema foi tratado como uma tarefa de regressão
supervisionada, pois o objetivo consistiu em estimar uma variável
numérica contínua: o volume de vendas acumuladas de videogames, expresso
em milhões de unidades.

As variáveis utilizadas nos experimentos incluíram avaliações da crítica
especializada, notas dos usuários, plataforma, gênero e ano de
lançamento. A utilização dessas características permitiu investigar sua
contribuição para a previsão do desempenho comercial registrado nas
bases históricas.

### 2.3 Regressão Linear {#regressao-linear}

A Regressão Linear é uma técnica estatística utilizada para representar
a relação entre uma variável dependente e uma ou mais variáveis
explicativas.

Sua formulação geral pode ser expressa por:

Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

Em que Y representa a variável de interesse, β₀ corresponde ao
intercepto, β₁ até βₙ representam os coeficientes das variáveis
explicativas e ε representa o termo de erro.

Segundo James et al. (2023), modelos lineares constituem ferramentas
importantes para a interpretação de relações estatísticas e podem
funcionar como referência inicial na comparação com métodos mais
complexos.

Neste estudo, a Regressão Linear foi utilizada como um dos modelos
preditivos, permitindo comparar seu desempenho com o algoritmo Random
Forest.

### 2.4 Random Forest

O Random Forest é um algoritmo de aprendizado supervisionado baseado na
combinação de múltiplas árvores de decisão.

Proposto por Breiman (2001), o método utiliza técnicas de aleatorização
e agregação para produzir previsões a partir de um conjunto de árvores,
reduzindo a dependência de uma única árvore de decisão.

Em problemas de regressão, a previsão final é obtida pela agregação das
previsões individuais das árvores.

Uma vantagem do Random Forest é sua capacidade de representar relações
não lineares e interações entre variáveis, sem exigir que a relação
entre os atributos e a variável de interesse seja estritamente linear.

Neste projeto, foram avaliadas duas abordagens principais: Random Forest
aplicado diretamente às vendas acumuladas e Random Forest com
transformação logarítmica da variável-alvo.

A transformação logarítmica foi investigada devido à distribuição
assimétrica das vendas, caracterizada pela presença de muitos jogos com
volumes comerciais relativamente baixos e um número menor de títulos com
vendas elevadas.

### 2.5 Avaliações do Metacritic e Desempenho Comercial {#avaliacoes-do-metacritic-e-desempenho-comercial}

O Metacritic é uma plataforma que reúne avaliações de diferentes
veículos especializados e apresenta indicadores agregados da recepção
crítica de produtos culturais, incluindo videogames.

Entre os indicadores utilizados neste estudo estão o Metascore,
associado à avaliação da crítica especializada, e o User Score,
relacionado às avaliações dos usuários.

Essas informações permitem investigar possíveis associações entre
recepção crítica e desempenho comercial. Entretanto, uma correlação
estatística não estabelece, isoladamente, uma relação de causa e efeito.

As vendas de um videogame podem estar relacionadas a diversos fatores
que não estão completamente representados nas bases utilizadas,
incluindo investimentos em marketing, popularidade de franquias,
distribuição comercial e características do público consumidor.

Além disso, a utilização das avaliações em modelos preditivos exige
atenção à disponibilidade temporal dessas informações. Uma nota
registrada posteriormente ao lançamento não pode ser considerada
automaticamente disponível no momento em que uma previsão teria sido
realizada.

Dessa forma, o presente trabalho investiga a contribuição das avaliações
para a capacidade preditiva dos modelos em uma perspectiva
retrospectiva, sem atribuir às notas um efeito causal sobre as vendas.

### 2.6 Avaliação de Modelos de Regressão {#avaliacao-de-modelos-de-regressao}

A avaliação de modelos de regressão envolve métricas que permitem
comparar os valores previstos com os valores efetivamente observados.

Neste estudo, foram utilizadas três métricas principais:

**Erro Absoluto Médio (MAE):** representa a média dos valores absolutos
das diferenças entre as previsões e os valores reais.

**Raiz do Erro Quadrático Médio (RMSE):** calcula a raiz quadrada da
média dos erros elevados ao quadrado, atribuindo maior penalização a
erros de grande magnitude.

**Coeficiente de Determinação (R²):** mede o desempenho do modelo em
relação a uma referência baseada na média dos valores observados,
podendo assumir valores negativos quando o modelo apresenta desempenho
inferior a essa referência.

Além dessas métricas, foram realizadas comparações com um modelo de
referência baseado na mediana das vendas e validações temporais
utilizando diferentes períodos de treinamento e teste.

Esses procedimentos permitem avaliar não apenas a capacidade preditiva
dos algoritmos, mas também suas limitações e sua estabilidade em
diferentes condições experimentais.

## 3. Materiais e Métodos {#materiais-e-metodos}

Esta pesquisa possui natureza quantitativa, exploratória e experimental,
utilizando dados secundários de vendas e avaliações de videogames. O
desenvolvimento foi organizado em etapas de obtenção, preparação,
integração e exploração dos dados, seguidas pela construção, comparação
e validação de modelos de aprendizado supervisionado.

O processamento foi realizado em Python, utilizando o Visual Studio Code
(VS Code) como ambiente de desenvolvimento. Os procedimentos foram
implementados em scripts independentes, permitindo executar, verificar e
reproduzir as diferentes etapas do projeto.

### 3.1 Ambiente de Desenvolvimento e Ferramentas

O ambiente computacional utilizado foi configurado com Python 3.12.10 e
um ambiente virtual (`.venv`) para gerenciamento das dependências.

As principais bibliotecas utilizadas foram:

- **Pandas:** leitura, transformação, integração e análise de tabelas.

- **NumPy:** operações numéricas e transformações matemáticas.

- **Matplotlib:** criação de gráficos e visualizações.

- **Seaborn:** apoio à visualização estatística.

- **Scikit-learn:** implementação dos modelos de regressão,
  pré-processamento, métricas e procedimentos de avaliação.

O projeto foi organizado em diretórios destinados aos dados originais
(`dados`), resultados (`resultados`) e modelos (`modelos`), além de
scripts Python para cada etapa.

Também foram produzidos arquivos `README.md` e `requirements.txt`, com
instruções de utilização e dependências do ambiente.

### 3.2 Conjuntos de Dados

Foram utilizados dois conjuntos de dados públicos disponibilizados na
plataforma Kaggle.

O primeiro conjunto, denominado *Video Game Sales & Industry Data
(1980--2024)*, contém informações comerciais de videogames, incluindo
título, plataforma, gênero, editora, ano de lançamento e vendas
acumuladas.

O segundo conjunto, denominado *Metacritic Video Games Reviews (up to
Sep 2025)*, contém informações sobre videogames avaliados no Metacritic,
incluindo identificação dos jogos, plataformas, avaliações da crítica
especializada e avaliações dos usuários.

Os dois conjuntos foram selecionados por apresentarem informações
complementares. Enquanto o primeiro permite investigar o desempenho
comercial registrado, o segundo possibilita examinar a relação entre
avaliações e vendas.

**Tabela 1 -- Quantidade de registros utilizados na preparação dos
dados.**

  Conjunto de dados            Registros após preparação
  -------------------------- ---------------------------
  Vendas de videogames                            18.828
  Avaliações do Metacritic                        21.913
  Registros integrados                             6.129

Fonte: Elaborado pelos autores a partir dos datasets utilizados.

A unidade de observação da base integrada corresponde a um registro
associado a um jogo e uma plataforma. Dessa forma, diferentes versões de
um mesmo título, lançadas para plataformas distintas, podem aparecer
como observações separadas.

### 3.3 Limpeza e Preparação dos Dados {#limpeza-e-preparacao-dos-dados}

A etapa de preparação foi implementada no script `limpeza_dados.py`,
responsável por carregar os conjuntos de dados, padronizar informações e
produzir arquivos adequados às análises posteriores.

Inicialmente, foram utilizados os arquivos CSV disponibilizados nos
pacotes originais. Em seguida, foram aplicados procedimentos de
identificação de registros duplicados, normalização dos nomes dos jogos
e tratamento dos campos necessários à integração.

A normalização dos títulos teve como finalidade reduzir diferenças de
representação textual que poderiam impedir a identificação de jogos
presentes nas duas bases.

Os anos de lançamento também foram convertidos para formatos numéricos
adequados, permitindo comparações entre os registros comerciais e as
informações do Metacritic.

Após a preparação, a base de vendas apresentou 18.828 registros e a base
do Metacritic apresentou 21.913 registros.

Os dados preparados foram armazenados nos arquivos:

- `resultados/vendas_preparadas.csv`

- `resultados/metacritic_preparado.csv`

A base comercial foi preservada para as análises exploratórias gerais,
enquanto a base integrada foi utilizada nos experimentos que exigiam
simultaneamente informações de vendas e avaliações.

Essa distinção é importante porque nem todos os jogos com vendas
registradas possuem correspondência identificada na base do Metacritic.

Além disso, valores de vendas iguais a zero foram mantidos na base
comercial. Entretanto, esses valores devem ser interpretados com
cautela, pois não foi possível estabelecer se representam ausência
efetiva de vendas ou informações comerciais não registradas.

### 3.4 Integração dos Conjuntos de Dados {#integracao-dos-conjuntos-de-dados}

A integração dos conjuntos de dados foi realizada por meio do script
`integracao_dados.py`, com o objetivo de relacionar as informações
comerciais dos videogames às avaliações disponíveis no Metacritic.

Como os conjuntos de dados possuem estruturas e convenções de
nomenclatura distintas, foi necessário estabelecer critérios de
correspondência para identificar registros referentes ao mesmo jogo e à
mesma plataforma.

Inicialmente, foram padronizados os nomes das plataformas, permitindo
relacionar diferentes representações de um mesmo sistema. Por exemplo, a
identificação "PlayStation 4", presente nos dados do Metacritic, foi
convertida para "PS4", utilizada na base comercial.

Posteriormente, foi realizada uma junção interna (*inner join*) entre os
conjuntos de dados, utilizando duas chaves: o nome normalizado do jogo e
a plataforma padronizada.

Após a junção, foram comparados os anos de lançamento registrados nas
duas fontes. Foram mantidas as correspondências cuja diferença absoluta
entre os anos fosse inferior ou igual a um ano.

Esse critério foi adotado para tolerar pequenas divergências entre as
fontes, possivelmente relacionadas a diferenças de datas de lançamento
por região ou de registro das informações. Entretanto, não foi possível
confirmar individualmente a origem dessas divergências.

Para identificar situações em que um registro pudesse estar associado a
múltiplas correspondências, foram criados identificadores para as
observações de cada conjunto de dados.

Os registros que apresentaram repetição desses identificadores nas
correspondências encontradas foram classificados como ambíguos e
separados da base principal.

**Tabela 2 -- Resultados do processo de integração.**

  Etapa                                            Quantidade de registros
  ---------------------------------------------- -------------------------
  Base preparada do Metacritic                                      21.913
  Base preparada de vendas                                          18.828
  Correspondências com diferença de até um ano                       6.177
  Correspondências únicas mantidas                                   6.129
  Correspondências ambíguas separadas                                   48

Fonte: Elaborado pelos autores a partir dos resultados do script
`integracao_dados.py`.

As 6.129 correspondências únicas foram armazenadas no arquivo
`resultados/jogos_integrados.csv`, utilizado como base para as análises
que relacionam avaliações e vendas.

As 48 correspondências classificadas como ambíguas foram armazenadas
separadamente no arquivo `resultados/correspondencias_ambiguas.csv`,
evitando sua inclusão automática nos experimentos.

Cabe destacar que a classificação de uma correspondência como única não
garante, por si só, que os registros associados representem exatamente a
mesma edição de um jogo. Portanto, o procedimento reduz ambiguidades
identificáveis, mas não elimina completamente a possibilidade de
associações incorretas.

### 3.5 Auditoria e Verificação da Integração {#auditoria-e-verificacao-da-integracao}

Após a integração, foram realizados procedimentos adicionais para
verificar a consistência dos registros associados.

O script `verificar_integracao.py` foi utilizado para examinar a
estrutura da base integrada, identificar valores ausentes, verificar
duplicações e analisar características das variáveis disponíveis.

A base integrada apresentou 6.129 registros e 24 colunas, sem valores
ausentes ou duplicações exatas na verificação realizada.

Também foi desenvolvido o script `auditoria_integracao.py`, responsável
por examinar divergências entre os anos de lançamento registrados nas
duas fontes.

Os resultados indicaram que 5.762 correspondências apresentavam anos
coincidentes, enquanto 367 apresentavam diferença de um ano.

**Tabela 3 -- Correspondência dos anos de lançamento.**

  Diferença entre os anos     Registros   Percentual aproximado
  ------------------------- ----------- -----------------------
  Zero ano                        5.762                   94,0%
  Um ano                            367                    6,0%
  Total                           6.129                    100%

Fonte: Elaborado pelos autores a partir da auditoria da integração.

Os resultados mostram que a maior parte das correspondências apresentou
concordância entre os anos de lançamento. Entretanto, essa concordância
não deve ser interpretada como comprovação definitiva da identidade dos
jogos associados.

### 3.6 Teste de Sensibilidade da Integração {#teste-de-sensibilidade-da-integracao}

Para investigar a influência do critério de tolerância de um ano sobre
os resultados preditivos, foi realizado um teste de sensibilidade por
meio do script `teste_sensibilidade_integracao.py`.

Nesse experimento, foram comparados os resultados obtidos com a base
integrada completa e aqueles obtidos utilizando somente registros com
anos de lançamento coincidentes.

**Tabela 4 -- Resultados do teste de sensibilidade.**

  Base utilizada               Registros      MAE     RMSE       R²
  -------------------------- ----------- -------- -------- --------
  Integração completa              6.129   0,4742   0,9472   0,3583
  Apenas anos coincidentes         5.762   0,4877   0,9687   0,3820

Fonte: Elaborado pelos autores a partir dos resultados do teste de
sensibilidade.

Observou-se que a utilização apenas de registros com anos coincidentes
resultou em um R² ligeiramente maior, mas também apresentou valores mais
elevados de MAE e RMSE.

Esses resultados demonstram que diferentes critérios de seleção podem
produzir alterações nas métricas de desempenho.

Entretanto, como a filtragem modifica a composição das amostras
avaliadas, as diferenças observadas não podem ser atribuídas
exclusivamente à qualidade das correspondências. Uma comparação
controlada sobre o mesmo conjunto de teste seria necessária para isolar
esse efeito.

Assim, o experimento foi utilizado como uma verificação complementar da
sensibilidade dos resultados, sem estabelecer superioridade definitiva
de um dos critérios de integração.

### 3.7 Definição do Problema Preditivo {#definicao-do-problema-preditivo}

O problema de Machine Learning foi formulado como uma tarefa de
regressão supervisionada, tendo como variável-alvo o volume de vendas
acumuladas de videogames, representado pelo atributo `total_sales`, em
milhões de unidades.

Foram utilizadas como variáveis explicativas as avaliações da crítica
especializada (`metascore`), as avaliações dos usuários (`user_score`),
a plataforma (`console`), o gênero (`genre`) e o ano de lançamento
(`ano`).

A utilização dessas características permitiu investigar a contribuição
de informações comerciais, temporais e de recepção crítica para a
estimativa das vendas registradas.

A base integrada continha 6.129 registros, que foram organizados em
conjuntos de treinamento e teste conforme o ano de lançamento.

### 3.8 Divisão Temporal dos Dados {#divisao-temporal-dos-dados}

Para avaliar os modelos em jogos lançados posteriormente aos utilizados
no treinamento, foi adotada uma divisão cronológica dos registros.

O conjunto de treinamento foi composto por jogos lançados até 2015,
enquanto o conjunto de teste incluiu jogos lançados a partir de 2016.

**Tabela 5 -- Divisão temporal utilizada nos experimentos principais.**

  Conjunto      Período              Registros
  ------------- ------------------ -----------
  Treinamento   Até 2015                 5.213
  Teste         A partir de 2016           916
  Total         ---                      6.129

Fonte: Elaborado pelos autores a partir da base integrada.

A divisão temporal evita que registros de jogos com anos de lançamento
posteriores sejam utilizados diretamente no treinamento dos modelos
avaliados em períodos anteriores.

Entretanto, essa estratégia não garante, isoladamente, que todas as
variáveis explicativas estivessem disponíveis no momento de uma previsão
real, especialmente no caso das avaliações do Metacritic.

### 3.9 Pré-processamento das Variáveis {#pre-processamento-das-variaveis}

O pré-processamento dos atributos foi realizado com recursos da
biblioteca Scikit-learn.

As variáveis numéricas foram submetidas a procedimentos de imputação
pela mediana, quando necessário. As variáveis categóricas, como
plataforma e gênero, foram transformadas por meio da técnica One-Hot
Encoding.

O One-Hot Encoding representa cada categoria por indicadores binários,
permitindo que os algoritmos utilizem informações categóricas em sua
estrutura numérica.

Os procedimentos de transformação foram integrados aos modelos por meio
de pipelines, permitindo aplicar aos dados de teste as transformações
definidas durante o treinamento.

Essa organização contribui para reduzir inconsistências entre as etapas
de treinamento e avaliação.

### 3.10 Algoritmos de Machine Learning

Foram implementadas diferentes abordagens para estimar as vendas
acumuladas dos videogames.

#### 3.10.1 Modelo de Referência {#modelo-de-referancia}

Foi utilizado um modelo simples baseado na mediana das vendas do
conjunto de treinamento, implementado com `DummyRegressor`.

Esse modelo produz previsões constantes e permite verificar se os
algoritmos de Machine Learning oferecem melhorias em relação a uma
estratégia básica.

#### 3.10.2 Regressão Linear {#regressao-linear-1}

A Regressão Linear foi utilizada como abordagem inicial de aprendizado
supervisionado, permitindo estimar relações entre as variáveis
explicativas e as vendas acumuladas.

O modelo também serviu como referência para comparação com métodos
capazes de representar relações não lineares.

#### 3.10.3 Random Forest

O algoritmo Random Forest foi implementado utilizando um conjunto de 200
árvores de decisão, profundidade máxima de 12 níveis, quantidade mínima
de três observações por folha e semente aleatória igual a 42.

A configuração utilizada foi:

- `n_estimators = 200`

- `max_depth = 12`

- `min_samples_leaf = 3`

- `random_state = 42`

O objetivo foi investigar se um método baseado em múltiplas árvores
conseguiria representar relações mais complexas entre as características
dos jogos e suas vendas.

#### 3.10.4 Random Forest com Transformação Logarítmica {#random-forest-com-transformacao-logaritmica}

Também foi desenvolvido um modelo Random Forest com transformação
logarítmica da variável-alvo.

Nesse procedimento, as vendas foram transformadas pela função `log1p`,
equivalente ao logaritmo natural de um mais o valor observado.

Após a previsão, foi aplicada a transformação inversa `expm1`,
retornando os resultados à escala original de milhões de unidades
vendidas.

A abordagem foi implementada com `TransformedTargetRegressor` e teve
como objetivo reduzir a influência da assimetria da distribuição de
vendas durante o treinamento.

As previsões finais foram limitadas a valores não negativos.

### 3.11 Métricas de Avaliação {#metricas-de-avaliacao}

Os modelos foram avaliados utilizando três métricas principais: Erro
Absoluto Médio (MAE), Raiz do Erro Quadrático Médio (RMSE) e Coeficiente
de Determinação (R²).

O MAE representa a média das diferenças absolutas entre os valores
observados e previstos:

MAE = (1/n) × Σ yᵢ − ŷᵢ

O RMSE representa a raiz quadrada da média dos erros elevados ao
quadrado:

RMSE = √\[(1/n) × Σ (yᵢ − ŷᵢ)²\]

O coeficiente R² foi calculado pela expressão:

R² = 1 − \[Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²\]

Nessas expressões, yᵢ representa o valor observado, ŷᵢ representa a
previsão do modelo, ȳ corresponde à média dos valores observados e n
representa a quantidade de observações.

Para MAE e RMSE, valores menores indicam erros mais baixos. Para R²,
valores mais próximos de 1 indicam melhor desempenho em relação à
referência baseada na média, enquanto valores negativos indicam
desempenho inferior a essa referência.

Como as vendas foram expressas em milhões de unidades, os valores de MAE
e RMSE também foram apresentados nessa unidade.

### 3.12 Experimentos Complementares

Além da comparação principal entre os modelos, foram realizados
experimentos complementares para investigar diferentes aspectos do
desempenho preditivo.

O script `comparar_importancia_notas.py` foi utilizado para comparar
modelos treinados sem avaliações, somente com Metascore e com Metascore
e User Score.

O script `importancia_variaveis.py` foi utilizado para examinar a
importância dos atributos na previsão, por meio da alteração do erro
após a permutação dos valores das variáveis.

Os scripts `analise_erros.py`, `analise_faixas_vendas.py` e
`comparar_modelos_faixas.py` foram empregados para investigar o
comportamento das previsões em diferentes faixas de vendas.

Também foram realizados experimentos de validação temporal, utilizando
diferentes períodos de treinamento e teste, implementados nos scripts
`validacao_temporal.py` e `validacao_referencia.py`.

Essas análises permitiram avaliar a contribuição das notas, a
distribuição dos erros e a estabilidade do desempenho dos modelos em
diferentes períodos.

## 4. Resultados e Discussão {#resultados-e-discussao}

Esta seção apresenta os resultados obtidos por meio das análises
exploratórias e dos experimentos de Machine Learning. Inicialmente, são
examinadas as características comerciais dos videogames, incluindo a
distribuição das vendas por gênero, plataforma, editora e ano de
lançamento.

Posteriormente, são discutidas as relações entre as avaliações do
Metacritic e as vendas, o desempenho dos modelos preditivos e os
resultados dos experimentos complementares de validação.

### 4.1 Distribuição das Vendas por Gênero {#distribuicao-das-vendas-por-ganero}

A análise das vendas por gênero foi realizada utilizando os 18.828
registros da base comercial preparada. As vendas foram agregadas
conforme a classificação de gênero presente no dataset.

Os resultados indicaram que os gêneros Sports, Action e Shooter
apresentaram os maiores volumes de vendas acumuladas registrados na
base.

**Tabela 6 -- Gêneros com maiores volumes de vendas registrados.**

  Posição   Gênero           Vendas (milhões)   Participação
  --------- -------------- ------------------ --------------
            Sports                   1.186,70         17,98%
  2         Action                   1.124,95         17,05%
  3         Shooter                    995,47         15,09%
  4         Misc                       556,89          8,44%
  5         Racing                     523,51          7,93%
  6         Role-Playing               425,86          6,45%

Fonte: Elaborado pelos autores a partir do dataset de vendas.

Os três gêneros com maiores vendas, Sports, Action e Shooter,
concentraram aproximadamente 50,12% do volume total registrado na base.

Esse resultado demonstra uma concentração significativa das vendas
acumuladas em um grupo relativamente pequeno de categorias.

Entretanto, é importante destacar que os valores representam a soma das
vendas registradas dos jogos classificados em cada gênero. Portanto, um
gênero pode apresentar maior volume total tanto por possuir jogos
individualmente muito vendidos quanto por reunir uma quantidade maior de
registros.

Além disso, os resultados refletem a cobertura histórica do dataset
utilizado, não devendo ser interpretados automaticamente como uma
representação completa do mercado mundial de videogames.

<figure>
<p> </p>
<figcaption>Figura 1 – Vendas acumuladas por gênero de
videogame.</figcaption>
</figure>

*Figura 1 -- Distribuição das vendas acumuladas por gênero, em milhões
de unidades. Fonte: Elaborado pelos autores.*

A Figura 1 apresenta visualmente a concentração das vendas entre os
gêneros analisados, destacando a participação de Sports, Action e
Shooter.

Os resultados respondem à primeira pergunta de pesquisa, indicando que
esses três gêneros apresentaram os maiores volumes de vendas acumuladas
na base comercial utilizada.

### 4.2 Distribuição das Vendas por Plataforma {#distribuicao-das-vendas-por-plataforma}

A análise das vendas por plataforma foi realizada utilizando os
registros da base comercial preparada, com o objetivo de identificar os
sistemas de videogames que concentraram os maiores volumes de vendas
acumuladas.

Os resultados indicaram que o PlayStation 2 (PS2) apresentou o maior
volume de vendas registrado, com aproximadamente 1.025,36 milhões de
unidades, correspondendo a 15,54% do total analisado.

Na segunda posição, o Xbox 360 apresentou 859,41 milhões de unidades,
representando 13,02% das vendas. Em seguida, o PlayStation 3 (PS3)
registrou 839,01 milhões de unidades, com participação de 12,72%.

**Tabela 7 -- Plataformas com maiores volumes de vendas registrados.**

  Posição   Plataforma              Vendas (milhões)   Participação
  --------- --------------------- ------------------ --------------
            PlayStation 2 (PS2)             1.025,36         15,54%
  2         Xbox 360 (X360)                   859,41         13,02%
  3         PlayStation 3 (PS3)               839,01         12,72%
  4         PlayStation (PS)                  546,21          8,28%
  5         PlayStation 4 (PS4)               539,92          8,18%
  6         Nintendo Wii                      458,92          6,95%
  7         Nintendo DS                       457,11          6,93%

Fonte: Elaborado pelos autores a partir do dataset de vendas.

As três plataformas com maiores vendas, PS2, Xbox 360 e PS3,
concentraram aproximadamente 41,28% das vendas acumuladas registradas.

Essa concentração evidencia a participação expressiva dessas plataformas
na base histórica analisada.

Entretanto, é necessário considerar que o volume total de vendas
associado a uma plataforma depende de diferentes fatores, incluindo a
quantidade de jogos registrados, o período de comercialização, a
cobertura temporal do dataset e a disponibilidade de informações
comerciais.

Portanto, a liderança de determinada plataforma não deve ser
interpretada automaticamente como evidência de que seus jogos
apresentaram, individualmente, maiores vendas médias.

**Figura 2 -- Vendas acumuladas por plataforma de videogame.**

<figure>
<p> </p>
<figcaption>Gráfico de vendas acumuladas por plataforma</figcaption>
</figure>

*Fonte: Elaborado pelos autores a partir do dataset de vendas.*

A Figura 2 apresenta a distribuição dos volumes de vendas entre as
principais plataformas.

Observa-se a liderança do PlayStation 2, seguida pelo Xbox 360 e pelo
PlayStation 3, enquanto outras plataformas também apresentam
participações relevantes no conjunto de dados.

Os resultados respondem à segunda pergunta de pesquisa, indicando que
PS2, Xbox 360 e PS3 foram as plataformas com maiores volumes de vendas
acumuladas entre os registros analisados.

### 4.3 Distribuição das Vendas por Editora {#distribuicao-das-vendas-por-editora}

A análise das vendas por editora teve como objetivo identificar as
empresas ou identificações editoriais que concentraram os maiores
volumes de vendas acumuladas nos registros comerciais utilizados.

Para essa investigação, foram considerados os 18.828 registros da base
de vendas preparada, agrupando-se os valores do atributo `total_sales`
conforme a editora identificada no campo `publisher`.

Os resultados indicaram que a Activision apresentou o maior volume de
vendas acumuladas, com aproximadamente 722,77 milhões de unidades,
correspondendo a 10,95% do total registrado na base.

Na segunda posição, a Electronic Arts apresentou 644,13 milhões de
unidades, representando 9,76% das vendas. Em seguida, a categoria EA
Sports registrou 485,66 milhões de unidades, com participação de 7,36%.

**Tabela 8 -- Editoras com maiores volumes de vendas registrados.**

  Posição   Editora                         Vendas (milhões)   Participação
  --------- ----------------------------- ------------------ --------------
            Activision                                722,77         10,95%
  2         Electronic Arts                           644,13          9,76%
  3         EA Sports                                 485,66          7,36%
  4         Ubisoft                                   462,57          7,01%
  5         THQ                                       320,89          4,86%
  6         Sony Computer Entertainment               311,08          4,71%

Fonte: Elaborado pelos autores a partir do dataset de vendas.

As três primeiras categorias do ranking concentraram aproximadamente
28,07% das vendas acumuladas registradas.

Esse resultado demonstra uma concentração relevante das vendas em
determinadas identificações editoriais presentes no conjunto de dados.

Entretanto, é necessário considerar que a classificação das editoras foi
preservada conforme os registros originais. Dessa forma, identificações
como Electronic Arts e EA Sports foram tratadas separadamente, sem a
realização de um processo de consolidação de grupos empresariais.

Consequentemente, os resultados representam a distribuição das vendas
entre as categorias editoriais presentes no dataset, não correspondendo
necessariamente à participação consolidada de cada grupo econômico no
mercado mundial.

Também é importante destacar que os volumes totais podem ser
influenciados pela quantidade de jogos registrados para cada editora,
pelos períodos de lançamento e pela cobertura comercial disponível na
base.

**Figura 3 -- Vendas acumuladas por editora de videogames.**

<figure>
<p> </p>
<figcaption>Gráfico de vendas acumuladas por editora</figcaption>
</figure>

*Fonte: Elaborado pelos autores a partir do dataset de vendas.*

A Figura 3 apresenta visualmente a distribuição das vendas acumuladas
entre as principais editoras registradas.

Observa-se que a Activision ocupa a primeira posição, seguida pelas
categorias Electronic Arts e EA Sports. Ubisoft, THQ e Sony Computer
Entertainment também apresentam volumes expressivos.

Os resultados respondem à terceira pergunta de pesquisa, identificando
as categorias editoriais com maiores vendas acumuladas no conjunto de
dados analisado.

### 4.4 Distribuição das Vendas por Ano de Lançamento {#distribuicao-das-vendas-por-ano-de-lancamento}

A análise temporal foi realizada agrupando-se os registros comerciais
conforme o ano de lançamento dos videogames. Para cada ano, foram
somadas as vendas acumuladas dos jogos registrados na base.

O maior volume foi identificado em 2008, com aproximadamente 538,09
milhões de unidades associadas a 1.624 registros de jogos e plataformas.

**Figura 4 -- Vendas acumuladas dos jogos por ano de lançamento.**

<figure>
<p> </p>
<figcaption>Vendas acumuladas por ano de lançamento</figcaption>
</figure>

*Fonte: Elaborado pelos autores a partir do dataset de vendas.*

A Figura 4 apresenta diferenças expressivas entre os volumes associados
aos diferentes anos de lançamento. Essas variações podem estar
relacionadas à quantidade de jogos registrados, à popularidade das
plataformas e à cobertura histórica da base utilizada.

É importante destacar que o gráfico não representa as vendas realizadas
em cada ano, mas a soma das vendas acumuladas dos jogos classificados
pelo respectivo ano de lançamento.

Além disso, os anos mais recentes apresentam menor quantidade de
registros e possíveis limitações de cobertura. Por exemplo, 2019 contém
35 registros e 2020 contém 28 registros na base analisada. Portanto, os
menores volumes observados nesses períodos não permitem concluir que
ocorreu uma redução equivalente nas vendas totais da indústria.

Os resultados respondem à quarta pergunta de pesquisa, demonstrando como
os volumes comerciais registrados se distribuem entre os anos de
lançamento dos jogos.

### 4.5 Relação entre Avaliações do Metacritic e Vendas {#relacao-entre-avaliacoes-do-metacritic-e-vendas}

A relação entre as avaliações dos videogames e seus resultados
comerciais foi investigada utilizando os 6.129 registros da base
integrada.

Foram calculados coeficientes de correlação de Pearson entre as vendas
acumuladas e duas variáveis: Metascore, correspondente à avaliação da
crítica especializada, e User Score, correspondente à avaliação dos
usuários.

**Tabela 9 -- Correlação entre avaliações e vendas acumuladas.**

  Variável                Correlação de Pearson
  --------------------- -----------------------
  Metascore × Vendas                     0,2903
  User Score × Vendas                    0,0742

Fonte: Elaborado pelos autores a partir da base integrada.

O Metascore apresentou correlação positiva fraca com as vendas, enquanto
o User Score apresentou correlação positiva muito fraca.

Esses resultados indicam que notas mais elevadas não estão
necessariamente associadas a volumes proporcionalmente maiores de
vendas.

**Figura 5 -- Relação entre Metascore e vendas acumuladas.**

<figure>
<p> </p>
<figcaption>Dispersão entre Metascore e vendas</figcaption>
</figure>

*Fonte: Elaborado pelos autores.*

A Figura 5 mostra uma concentração de jogos nas faixas de menores
vendas, mesmo quando apresentam avaliações relativamente elevadas. Isso
indica que a avaliação crítica, isoladamente, não explica a maior parte
da variação comercial observada.

**Figura 6 -- Relação entre User Score e vendas acumuladas.**

<figure>
<p> </p>
<figcaption>Dispersão entre User Score e vendas</figcaption>
</figure>

*Fonte: Elaborado pelos autores.*

Na Figura 6, observa-se uma associação linear ainda menor entre as
avaliações dos usuários e as vendas acumuladas.

A comparação sugere que o Metascore apresenta maior associação linear
com as vendas do que o User Score no conjunto analisado.

Entretanto, correlação não implica causalidade. Não é possível afirmar,
a partir desses resultados, que avaliações positivas provocam aumento
das vendas.

Além disso, como as avaliações e as vendas foram obtidas de registros
históricos, não foi possível assegurar que as notas estivessem
disponíveis antes da realização das vendas.

Portanto, os resultados devem ser interpretados como associações
retrospectivas, e não como evidência de capacidade de previsão das
vendas no momento do lançamento.

### 4.6 Comparação dos Modelos Preditivos {#comparacao-dos-modelos-preditivos}

Os experimentos de Machine Learning foram realizados utilizando 5.213
registros para treinamento e 916 registros para teste, respeitando a
divisão temporal estabelecida na metodologia.

Foram comparados quatro modelos: DummyRegressor, Regressão Linear,
Random Forest e Random Forest com transformação logarítmica da
variável-alvo.

**Tabela 10 -- Desempenho dos modelos no conjunto de teste.**

  Modelo                       MAE         RMSE           R²
  ------------------- ------------ ------------ ------------
  DummyRegressor        **0,4426**       1,1967      -0,0242
  Regressão Linear          0,7295       1,1393       0,0717
  Random Forest             0,5629       1,0067       0,2752
  Random Forest Log         0,4742   **0,9472**   **0,3583**

Fonte: Elaborado pelos autores a partir dos experimentos de Machine
Learning.

O modelo Random Forest com transformação logarítmica apresentou o menor
RMSE (0,9472) e o maior R² (0,3583), indicando melhor desempenho nessas
métricas entre os algoritmos avaliados.

O R² obtido indica que o modelo apresentou uma redução de
aproximadamente 35,83% na soma dos erros quadráticos em comparação com
uma referência baseada na média das vendas do conjunto de teste.

Entretanto, o DummyRegressor apresentou o menor MAE (0,4426), superando
os demais modelos nessa métrica.

Esse comportamento demonstra que a escolha do melhor algoritmo depende
do critério de avaliação utilizado. Enquanto o MAE atribui peso
proporcional aos erros absolutos, o RMSE penaliza mais intensamente
previsões com erros elevados.

A transformação logarítmica contribuiu para melhorar o desempenho do
Random Forest em relação à versão sem transformação, reduzindo o RMSE de
1,0067 para 0,9472 e aumentando o R² de 0,2752 para 0,3583.

**Figura 7 -- Valores reais e previstos pelo Random Forest.**

<figure>
<p> </p>
<figcaption>Comparação entre vendas reais e previstas pelo Random
Forest</figcaption>
</figure>

*Fonte: Elaborado pelos autores.*

A Figura 7 apresenta a relação entre as vendas observadas e as previsões
produzidas pelo Random Forest sem transformação logarítmica.

**Figura 8 -- Valores reais e previstos pelo Random Forest com
transformação logarítmica.**

<figure>
<p> </p>
<figcaption>Comparação entre vendas reais e previstas pelo Random Forest
Log</figcaption>
</figure>

*Fonte: Elaborado pelos autores.*

A Figura 8 apresenta os resultados do modelo com transformação
logarítmica. Embora tenha obtido o melhor RMSE entre os modelos
avaliados, ainda foram observadas diferenças relevantes entre valores
reais e previstos.

Os resultados indicam que as características utilizadas apresentam
capacidade preditiva limitada para estimar as vendas acumuladas dos
videogames.

Além disso, como as avaliações foram extraídas de registros históricos,
o desempenho obtido corresponde a uma avaliação retrospectiva e não
comprova a capacidade de realizar previsões no momento do lançamento dos
jogos.

### 4.7 Importância das Variáveis e Contribuição das Avaliações {#importancia-das-variaveis-e-contribuicao-das-avaliacoes}

A importância das variáveis foi investigada por meio da técnica de
importância por permutação, que avalia a alteração no desempenho do
modelo após o embaralhamento dos valores de cada atributo.

No experimento realizado com o Random Forest com transformação
logarítmica, o Metascore apresentou o maior aumento do erro absoluto
médio (MAE), seguido pelas variáveis plataforma, gênero e User Score.

**Figura 9 -- Importância das variáveis na previsão das vendas.**

<figure>
<p> </p>
<figcaption>Importância das variáveis no modelo preditivo</figcaption>
</figure>

*Fonte: Elaborado pelos autores.*

A Figura 9 mostra que o Metascore apresentou a maior importância por
permutação, com aumento aproximado de 0,1019 milhão de unidades no MAE.
Plataforma, gênero e User Score apresentaram aumentos de 0,0397, 0,0389
e 0,0328 milhão, respectivamente.

A variável ano apresentou importância nula nesse experimento.
Entretanto, esse resultado não permite concluir que o ano de lançamento
seja irrelevante para as vendas, pois a avaliação foi realizada em um
período posterior ao utilizado no treinamento.

Para investigar a contribuição das avaliações, foram comparadas três
configurações do modelo: sem notas, utilizando somente Metascore e
utilizando Metascore e User Score.

**Tabela 11 -- Desempenho conforme as avaliações utilizadas.**

  Configuração                      MAE         RMSE           R²
  ------------------------ ------------ ------------ ------------
  Sem avaliações                 0,5209       1,0675       0,1850
  Somente Metascore              0,4836       0,9834       0,3084
  Metascore e User Score     **0,4742**   **0,9472**   **0,3583**

Fonte: Elaborado pelos autores a partir dos experimentos de Machine
Learning.

A inclusão do Metascore melhorou as três métricas em comparação com a
configuração sem avaliações. A inclusão adicional do User Score também
produziu melhorias, embora menores.

Esses resultados indicam que as avaliações contribuíram para o
desempenho preditivo retrospectivo do modelo, mesmo apresentando
correlações lineares relativamente baixas com as vendas.

Entretanto, não é possível estabelecer uma relação causal entre
avaliações e resultados comerciais. Também não foi possível comprovar
que essas notas estariam disponíveis no momento de uma previsão
realizada durante o lançamento dos jogos.

### 4.8 Validação Temporal dos Modelos {#validacao-temporal-dos-modelos}

Para avaliar a estabilidade dos resultados em diferentes períodos, foram
realizados três experimentos de validação temporal, utilizando conjuntos
de treinamento compostos por jogos lançados antes dos respectivos
períodos de teste.

Os experimentos consideraram o Random Forest com transformação
logarítmica, utilizando Metascore e User Score como variáveis
explicativas, juntamente com as demais características selecionadas.

**Tabela 12 -- Resultados da validação temporal do Random Forest Log.**

  Treinamento   Teste          Registros de teste      MAE     RMSE       R²
  ------------- ------------ -------------------- -------- -------- --------
  Até 2012      2013--2014                    423   0,6291   1,4300   0,4770
  Até 2014      2015--2016                    463   0,5805   1,1972   0,0326
  Até 2016      2017--2018                    627   0,4070   0,8594   0,4562

Fonte: Elaborado pelos autores a partir dos experimentos de validação
temporal.

Os resultados demonstram diferenças de desempenho entre os períodos
analisados. O maior R² foi observado no teste de 2013--2014, enquanto o
menor ocorreu no período de 2015--2016.

No período de 2017--2018, o modelo apresentou os menores valores
absolutos de MAE e RMSE entre os três experimentos. Entretanto, como os
conjuntos de teste possuem distribuições e características diferentes,
essa redução não comprova, isoladamente, uma melhoria geral do algoritmo
ao longo do tempo.

Também foi realizada uma comparação com o modelo de referência baseado
na mediana. O Random Forest Log com ambas as avaliações apresentou MAE
inferior ao modelo de referência em dois dos três períodos avaliados,
mas não no período de 2015--2016.

Esses resultados indicam que o desempenho preditivo depende do período
considerado e reforçam a importância de utilizar avaliações temporais
complementares.

Por fim, embora a divisão cronológica reduza o risco de utilizar jogos
futuros no treinamento, ela não garante que as avaliações do Metacritic
estivessem disponíveis no momento do lançamento. Dessa forma, os
experimentos continuam representando avaliações retrospectivas.

## 5. Limitações do Estudo {#limitacoes-do-estudo}

Apesar dos resultados obtidos, esta pesquisa apresenta limitações
relacionadas à cobertura dos dados, à integração das fontes e à
capacidade de generalização dos modelos preditivos.

Inicialmente, os conjuntos de dados utilizados possuem cobertura
histórica limitada e não representam necessariamente a totalidade dos
jogos lançados ou das vendas realizadas no mercado mundial. Além disso,
os valores comerciais correspondem a vendas acumuladas, e não às vendas
realizadas exclusivamente no ano de lançamento.

A integração entre as bases de vendas e Metacritic também constitui uma
limitação. Embora tenham sido aplicados procedimentos de normalização,
padronização de plataformas, comparação de anos e exclusão de
correspondências ambíguas, não foi realizada uma validação manual de
todas as associações. Dessa forma, podem existir correspondências
incorretas não identificadas.

Outra limitação relevante envolve a disponibilidade temporal das
avaliações. Como os dados foram coletados retrospectivamente, não foi
possível determinar se as notas da crítica especializada e dos usuários
estavam disponíveis no momento em que as previsões seriam realizadas.

Consequentemente, os resultados não comprovam a possibilidade de prever
vendas na semana de lançamento utilizando essas avaliações, nem permitem
estabelecer relações causais entre notas e desempenho comercial.

Também foram observadas diferenças de desempenho entre os períodos de
validação temporal. Além disso, os modelos apresentaram dificuldades
para representar toda a variabilidade das vendas, especialmente em um
cenário caracterizado pela concentração de registros com baixos volumes
comerciais e pela existência de jogos com vendas muito elevadas.

Por fim, a pesquisa utilizou um conjunto limitado de variáveis
explicativas. Fatores como orçamento de marketing, preço, franquia,
distribuição, concorrência e campanhas promocionais não foram
incorporados aos modelos.

Essas limitações indicam que os resultados devem ser interpretados como
uma análise exploratória e preditiva retrospectiva, sendo necessários
dados adicionais e validações específicas para aplicações comerciais
prospectivas.

## 6. Conclusão {#conclusao}

Este trabalho apresentou uma análise das vendas de videogames e de sua
relação com as avaliações do Metacritic, utilizando técnicas de Ciência
de Dados e Machine Learning. O objetivo foi identificar padrões
comerciais, investigar a associação entre avaliações e vendas e comparar
algoritmos de regressão para estimar o desempenho comercial dos jogos.

A análise exploratória de 18.828 registros comerciais identificou
Sports, Action e Shooter como os gêneros com maiores volumes de vendas
acumuladas. Entre as plataformas, destacaram-se PlayStation 2, Xbox 360
e PlayStation 3, enquanto Activision, Electronic Arts e EA Sports
lideraram as categorias editoriais registradas na base.

Após a integração dos conjuntos de dados, foram obtidos 6.129 registros
para investigar a relação entre avaliações e vendas. O Metascore
apresentou correlação positiva fraca com as vendas (0,2903), enquanto o
User Score apresentou correlação positiva muito fraca (0,0742).

Nos experimentos preditivos, o Random Forest com transformação
logarítmica apresentou o menor RMSE (0,9472) e o maior R² (0,3583).
Entretanto, o modelo de referência DummyRegressor apresentou o menor MAE
(0,4426), demonstrando que a escolha do melhor modelo depende da métrica
considerada.

Os experimentos complementares indicaram que a inclusão das avaliações
contribuiu para melhorar o desempenho retrospectivo do Random Forest.
Contudo, a validação temporal demonstrou variações entre diferentes
períodos, evidenciando limitações na generalização dos resultados.

Conclui-se que as avaliações do Metacritic fornecem informações úteis
para a análise retrospectiva das vendas, mas não são suficientes,
isoladamente, para explicar o desempenho comercial dos videogames. Além
disso, a ausência de informações sobre o momento de disponibilização das
avaliações impede confirmar a capacidade de previsão das vendas no
lançamento.

Como trabalhos futuros, recomenda-se incorporar variáveis relacionadas a
preços, marketing, franquias e concorrência, além de utilizar dados
comerciais e avaliações com registros temporais verificáveis. Essas
melhorias poderão contribuir para o desenvolvimento de modelos mais
adequados à previsão prospectiva de vendas.

## Referências {#referancias}

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5--32.
https://doi.org/10.1023/A:1010933404324.

James, G., Witten, D., Hastie, T., Tibshirani, R. e Taylor, J. (2023).
*An Introduction to Statistical Learning: With Applications in Python*.
Springer. https://doi.org/10.1007/978-3-031-38747-0.

Provost, F. e Fawcett, T. (2013). *Data Science for Business: What You
Need to Know About Data Mining and Data-Analytic Thinking*. O'Reilly
Media.

Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
