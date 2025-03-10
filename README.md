# Visão Geral

Este projeto contém scripts para o pré-processamento, treinamento e análise de dados de posts em redes sociais dos candidatos Jair Bolsonaro e Luiz Inácio Lula da Silva. O objetivo é investigar técnicas para prever o engajamento (quantidade de likes)
a partir dos embeddings gerados a partir das postagens.

O código principal está nos notebooks Jupyter (`*.ipynb`). Arquivos de Python puro (`*.py`) definem funções chamadas pelos notebooks. 

# 1 - Visão Geral

## Código Principal

O código está distribuído nos notebooks abaixo. Eles podem ser executados na ordem indicada.

- **`1.1-Preprocessamento_A`** - Responsável pela limpeza dos dados e selecão das colunas relevantes.
- **`1.2-Preprocessamento_B`** - Focado na análise das colunas e na criação de opções de variáveis alvos (inclusive para classificação). (Acho que é possível fazer o merge)
- **`2-Classificacao-50p-Treinamento`** - Otimiza hiperparâmetros e treina modelos de classificação binária para o atributo 
`Curtidas-2Classes-50p` (definido com base na mediana das curtidas).
- **`3-Classificacao-50p-Analise-Resultados`** - Analisa os resultados obtidos no notebook anterior.
- **`4-Regressao`** - Esboço de código para treinar modelos para regressão. (Não concluído).


## Códigos Auxiliares

Estes arquivos definem as funções usadas nos notebooks explicados acima.

- **`classification_train_util.py`** - Define a principal função para o treinamento de modelos de classificação, que realiza validação cruzada aninhada com grid search.
- **`data_transformations_util.py`** - Define transformações de dados personalizadas para pipelines de machine learning. Nela, foi criado um target encoding misto (de classificação/regressão).
- **`ensemble_train_util.py`** - Define funções para o treinamento de modelos de *ensemble*, formado pela combinações de modelos de tipos quaisquer.
- **`util.py`** - Define funções utilitárias para filtragem de datasets e para salvar e carregar resultados dos treinamentos.


# 2 - Mais Detalhes do Código Principal (Notebooks)

## Notebook `1.1-Preprocessamento_A.ipynb`

- **Concatenação**: Dados de Bolsonaro e Lula são unidos.

- **Limpeza**: Colunas irrelevantes são removidas e linhas com valores ausentes são excluídas.

- **Cálculo de Novas Variáveis**
   - *Dias Decorridos*: Número de dias entre a data de coleta e a data do post.
   - *Candidato*: Identifica o candidato.

## Notebook `1.2-Preprocessamento_B.ipynb`

- **Análises de correlação**: Matriz de correlação entre curtidas, plays, comentários, e compartilhamentos, incluindo a versão logarítmica.

- **Gráficos de Distribuição de Curtidas**:
   - *Scatter Plots*: Gráficos de dispersão para curtidas em escala logarítmica.
   - *Histograma*: Plotagem da distribuição de curtidas e suas transformações.

- **Criação de Nova Variável de Regressão** a partir da coluna `Curtidas`: 
   - *Curtidas-Log*: o logaritmo da quantidade de curtidas
   - *Curtidas-MinMax* e *Curtidas-Log-MinMax*: versões normalizadas de forma linear entre 0 e 1 

- **Criação de Novas Variáveis de Classificação**:
  - *Curtidas-2Classes-75p*: Indica se as curtidas estão acima do 75º percentil.
  - *Curtidas-2Classes-50p*: Indica se as curtidas estão acima da mediana.

- **Análise de Variáveis Categóricas**:
   - Gera histogramas e verifica a distribuição de curtidas por variáveis categóricas.

- **Testes Estatísticos**:
   - *ANOVA* e *Kruskal-Wallis*: para avaliar a influência das variáveis categóricas na quantidade de curtidas.


## Notebook `2-Classificacao-50p-Treinamento.ipynb`

Define modelos e hiperparâmetros dos modelos e realiza a otimização e o treinamento deles 
para prever as classes binárias baseadas nas curtidas (e.g. high/low).

1. **Configuração do Dataset**: Filtra o dataset, permitindo treinar em posts de Lula, Bolsonaro ou ambos. .
1. **Modelos e Hiperparâmetros**: Define vários modelos, incluindo `MLP Neural Network`, `SVM`, `Random Forest`, `Logistic Regression`, e `KNN`.
1. **Treinamento e Validação Cruzada**: Realiza treinamento com validação cruzada aninhada e treino de ensemble para combinar os melhores modelos.
1. **Salvar Resultados**: Salva os resultados e metadados em um arquivo `.npy`.


## Notebook `3-Classificao-50p-Analise-Resultados.ipynb`

Notebook usado para analisar os resultados do treinamento correspondente, permitindo a visualização e comparação das métricas 
de desempenho dos modelos.

1. **Análise de Métricas**: Ordena modelos por métricas como F1-score, precisão (*precision*), e revocação (*recall*).
1. **Comparação com Classificadores Aleatórios**: Compara os resultados obtidos com classificadores aleatórios.
1. **Análise de Hiperparâmetros**: Visualiza os melhores hiperparâmetros de cada modelo.
1. **Plotagem de Importância de Features**: Plota a importância das features extraídas de modelos baseados em árvores de decisão.


## Notebook `4-Regressao.ipynb`

Para avaliar modelos de regressão, para prever a quantidade de likes (ou o log dessa quantidade).

Não concluído.
