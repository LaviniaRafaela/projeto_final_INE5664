# Projeto Final – INE5664

## Implementação de uma Rede Neural Artificial Multicamadas (MLP) do Zero em Python

Este projeto consiste na implementação de uma **Rede Neural Artificial Multicamadas (Multilayer Perceptron – MLP)** desenvolvida integralmente em **Python e NumPy**, sem o uso de bibliotecas de Deep Learning como TensorFlow, PyTorch ou Keras.

O objetivo é compreender o funcionamento interno de uma rede neural, implementando manualmente todas as etapas do treinamento:

- propagação direta (*forward propagation*);
- cálculo das funções de perda;
- retropropagação (*backpropagation*);
- atualização dos pesos por Gradiente Descendente com Mini-Batches;
- diferentes funções de ativação;
- diferentes funções de perda;
- inicializações de pesos.

Além da implementação da biblioteca, o projeto apresenta sua aplicação em **três problemas reais relacionados à corrosão atmosférica**, comparando o desempenho da MLP implementada com modelos do **scikit-learn**.

---

# Estrutura do Projeto 
PROJETO_FINAL_INE5664
    datasets/
        dataset_cl.xlsx
    dataset_SO2.xlsx
    NNRaw/   
        neural_network.py

    notebooks/
        1_regressao_cloreto.ipynb
        2_multiclasse_so2.ipynb
        3_binaria_cloreto.ipynb
    README.md


---

# Biblioteca Implementada

O arquivo "neural_network.py" implementa toda a biblioteca de redes neurais.

## Funcionalidades implementadas

### Funções de ativação

- Linear (identidade)
- ReLU
- Leaky ReLU
- ELU
- Sigmoid
- Tanh
- Softmax

---

### Funções de perda

- Mean Squared Error (MSE)
- Binary Cross Entropy
- Categorical Cross Entropy

---

### Inicialização dos pesos

- Aleatória
- Xavier
- He

---

### Arquitetura

A biblioteca permite criar qualquer arquitetura MLP utilizando camadas conectadas.
Exemplo:
rede = NN.RedeNeural(perda='mse')
rede.adicionar_camada(6,16,'relu','he')
rede.adicionar_camada(16,8,'relu','he')
rede.adicionar_camada(8,1,'linear','he')

---

# Problemas Resolvidos

## 1. Regressão: Predição da Deposição Seca de Cloreto

Notebook: 1_regressao_cloreto.ipynb

Objetivo: Prever numericamente a deposição seca de cloreto (Cl⁻), importante indicador da corrosividade atmosférica.

Variáveis de entrada:
- distância da costa;
- altitude;
- massa de aerossol marinho;
- velocidade do vento;
- tempo de exposição ao vento;
- rugosidade do terreno.

Modelos comparados:
- Nossa MLP
- MLPRegressor (scikit-learn)
- Random Forest
- SVR

Métricas:
- R^2
- MAE
- RMSE

---

## 2. Classificação Multiclasse – Categoria de Poluição por SO₂

Notebook: 2_multiclasse_so2.ipynb

Objetivo: Classificar uma estação segundo as categorias de poluição da ISO 9223:
- P0
- P1
- P2
- P3

Modelos comparados:
- Nossa MLP
- MLPClassifier (scikit-learn)
- Random Forest
- KNN

Métricas:
- Accuracy
- F1-score Macro

---

## 3. Classificação Binária – Categoria de Salinidade

Notebook: 3_binaria_cloreto.ipynb

Objetivo: Classificar ambientes em:
- Baixa/Média salinidade
- Alta salinidade

Modelos comparados:
- Nossa MLP
- MLPClassifier (scikit-learn)
- Regressão Logística
- SVM

Métricas:
- Accuracy
- F1-score
- AUC-ROC

---

# Metodologia Experimental

Todos os notebooks seguem a mesma metodologia.

## 1. Pré-processamento

- seleção das variáveis;
- divisão treino/teste por projetos;
- normalização Min-Max.

---

## 2. Validação Cruzada

Os modelos são avaliados utilizando **Monte Carlo Cross Validation (Random Subsampling Estratificado)**. Em cada notebook são realizados 100 splits estratificados. 
Essa etapa é utilizada para:
- comparar os modelos;
- estimar desempenho médio;
- selecionar hiperparâmetros.

---

## 3. Treinamento Final

Após a validação cruzada, cada modelo é treinado novamente utilizando **todo o conjunto de treinamento**.

---

## 4. Avaliação Final

Os modelos são avaliados em um conjunto de teste composto por projetos nunca vistos durante o treinamento.

São apresentados:
- métricas de treino;
- métricas de teste;
- gráficos comparativos;
- curvas de perda;
- matrizes de confusão (problemas de classificação);
- gráficos Real e Predito (problema de regressão).

---

# Tecnologias Utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# Como Executar

## 1. Clone o projeto

```bash
git clone <repositorio>
```

---

## 2. Instale as dependências

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl notebook
```

---

## 3. Abra os notebooks

```bash
jupyter notebook
```

Execute os notebooks na seguinte ordem:

1. `1_regressao_cloreto.ipynb`
2. `2_multiclasse_so2.ipynb`
3. `3_binaria_cloreto.ipynb`

---

# Autores
Lavínia Rafaela De Marco e Pedro Philippi Araujo

Projeto desenvolvido para a disciplina **INE5664**.

Universidade Federal de Santa Catarina (UFSC).