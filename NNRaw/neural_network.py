import numpy as np


# =============================================================================
# FUNÇÕES DE ATIVAÇÃO
# Cada função retorna também sua derivada, usada no backpropagation.
# =============================================================================

class Linear:
    """
    Ativação identidade: f(x) = x
    """
    def forward(self, z):
        return z

    def backward(self, grad):
        return grad

    def __repr__(self):
        return "Linear"
    
class ReLU:
    """
    Rectified Linear Unit (ReLU): f(x) = max(0, x)
    Use para todas as camadas ocultos

    """
    def forward(self, z):
        self.z = z  # guarda para usar na derivada
        return np.maximum(0, z)

    def backward(self, grad):
        # Derivada: 1 onde z > 0, 0 onde z <= 0
        return grad * (self.z > 0).astype(float)

    def __repr__(self):
        return "ReLU"


class Sigmoid:
    """
    Sigmoid: f(x) = 1 / (1 + e^(-x))
    - Saída entre 0 e 1, boa para classificação binária na camada final
    """
    def forward(self, z):
        # Clipa para evitar overflow numérico
        self.output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return self.output

    def backward(self, grad):
        # Derivada: sigmoid(z) * (1 - sigmoid(z))
        s = self.output
        return grad * s * (1 - s)

    def __repr__(self):
        return "Sigmoid"


class Tanh:
    """
    Tangente hiperbólica: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """
    def forward(self, z):
        self.output = np.tanh(z)
        return self.output

    def backward(self, grad):
        # Derivada: 1 - tanh²(z)
        return grad * (1 - self.output ** 2)

    def __repr__(self):
        return "Tanh"


class Softmax:
    """
    Softmax: f(x_i) = e^(x_i) / Σ e^(x_j)
    - Usada na camada final para classificação multiclasse
    """
    def forward(self, z):
        # Subtrai o máximo para estabilidade numérica (evita overflow)
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.output = e / np.sum(e, axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        # Quando usada com Cross-Entropy, o gradiente já vem simplificado
        return grad

    def __repr__(self):
        return "Softmax"


class LeakyReLU:
    """
    Leaky ReLU: f(x) = x se x > 0, senão alfa * x
    - Permite gradiente negativo quando x <= 0
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, z):
        self.z = z
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, grad):
        dz = np.where(self.z > 0, 1.0, self.alpha)
        return grad * dz

    def __repr__(self):
        return f"LeakyReLU(alfa = {self.alpha})"


class ELU:
    """
    Exponential Linear Unit (ELU) f(x) = x se x > 0, senão alfa*(e^x - 1).
    - Empurra a média das ativações para perto de zero e acelera o aprendizado.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, z):
        self.z = z
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def backward(self, grad):
        dz = np.where(self.z > 0, 1.0, self.alpha * np.exp(self.z))
        return grad * dz

    def __repr__(self):
        return f"ELU(alfa = {self.alpha})"


# Mapa de strings para classes de ativação
ATIVACOES = {
    "linear": Linear,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
}


# =============================================================================
# FUNÇÕES DE PERDA (LOSS)
# Medem o quão errada está a rede. O gradiente da perda inicia o backprop.
# =============================================================================

class MSE:
    """
    Erro Quadrático Médio: L = (1/n) * Σ (y_pred - y_true)²
    - Usada para regressão
    """
    def calcular(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def gradiente(self, y_pred, y_true):
        # dL/dy_pred = 2 * (y_pred - y_true) / n
        return 2 * (y_pred - y_true) / y_true.shape[0]


class CrossEntropyBinaria:
    """
    Log Loss binária: L = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]
    - Usada para classificação binária (com Sigmoid na saída)
    """
    def calcular(self, y_pred, y_true):
        eps = 1e-12  # evita log(0)
        p = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def gradiente(self, y_pred, y_true):
        eps = 1e-12
        p = np.clip(y_pred, eps, 1 - eps)
        return (p - y_true) / (p * (1 - p) * y_true.shape[0])


class CrossEntropyCategorica:
    """
    Cross-Entropy multiclasse: L = -(1/n) * Σ Σ y_true * log(y_pred)
    - Usada para classificação multiclasse (com Softmax na saída)
    """
    def calcular(self, y_pred, y_true):
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0)
        return -np.mean(np.sum(y_true * np.log(p), axis=1))

    def gradiente(self, y_pred, y_true):
        # Gradiente combinado Softmax + Cross-Entropy (forma simplificada)
        return (y_pred - y_true) / y_true.shape[0]


PERDAS = {
    "mse":                    MSE,
    "cross_entropy_binaria":  CrossEntropyBinaria,
    "cross_entropy_categorica": CrossEntropyCategorica,
}


# =============================================================================
# CAMADA DENSA (FULLY CONNECTED)
# O bloco fundamental da rede: z = X·W + b
# =============================================================================

class CamadaDensa:
    """
    Camada totalmente conectada.

    Parâmetros treináveis:
      - W: matriz de pesos  (n_entrada × n_neuronios)
      - b: vetor de bias    (1 × n_neuronios)

    Inicialização:
      - "xavier"
      - "he"
      - "aleatorio"
    """

    def __init__(self, n_entrada, n_neuronios, ativacao, inicializacao):
        self.ativacao_fn = ATIVACOES[ativacao]()

        # --- Inicialização dos pesos ---
        if inicializacao == "xavier":
            # Controla a variância para manter o gradiente estável com Sigmoid/Tanh
            limite = np.sqrt(6 / (n_entrada + n_neuronios))
            self.W = np.random.uniform(-limite, limite, (n_entrada, n_neuronios))
        elif inicializacao == "he":
            # Recomendado para ReLU: compensa que metade dos neurônios ficam zerados
            self.W = np.random.randn(n_entrada, n_neuronios) * np.sqrt(2 / n_entrada)
        else:
            self.W = np.random.randn(n_entrada, n_neuronios) * 0.01

        self.b = np.zeros((1, n_neuronios))

        # Acumuladores de gradiente (preenchidos no backward)
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        Propagação direta:
          z = X · W + b   (combinação linear)
          a = f(z)        (ativação não-linear)
        """
        self.entrada = X          # guarda para o backprop
        self.z = X @ self.W + self.b
        self.saida = self.ativacao_fn.forward(self.z)
        return self.saida

    def backward(self, grad_saida):
        # Retropropagação
        # Gradiente após a ativação
        grad_z = self.ativacao_fn.backward(grad_saida)

        # Gradientes dos parâmetros desta camada
        self.dW = self.entrada.T @ grad_z        # dL/dW = X^T · δ
        self.db = np.sum(grad_z, axis=0, keepdims=True)  # dL/db = Σ δ

        # Gradiente que flui para a camada anterior: dL/dX = δ · W^T
        return grad_z @ self.W.T

    def atualizar(self, taxa_aprendizado):
        #Atualiza pesos e bias pelo gradiente descendente
        self.W -= taxa_aprendizado * self.dW
        self.b -= taxa_aprendizado * self.db


# =============================================================================
# REDE NEURAL (MLP)
# Orquestra as camadas, o treinamento e a avaliação.
# =============================================================================

class RedeNeural:
    """
    Perceptron Multicamadas (MLP).

    Exemplo de uso:
        rede = RedeNeural(perda="mse")
        rede.adicionar_camada(n_entrada=2, n_neuronios=8, ativacao="relu")
        rede.adicionar_camada(n_entrada=8, n_neuronios=1, ativacao="sigmoid")
        rede.treinar(X, y, epocas=100, taxa_aprendizado=0.01)
    """

    def __init__(self, perda):
        #armazena camadas em um array
        self.camadas = []
        #armazena as funcoes de perda
        self.fn_perda = PERDAS[perda]()
        #cria um historico de perda de treinamento e validacao
        self.historico = {"perda_treino": [], "perda_val": []}

    def adicionar_camada(self, n_entrada, n_neuronios, ativacao, inicializacao):
        """Adiciona uma camada densa à rede."""
        camada = CamadaDensa(n_entrada, n_neuronios, ativacao, inicializacao)
        self.camadas.append(camada)
        return self  # permite encadeamento: rede.adicionar(...).adicionar(...)

    def forward(self, X):
        """Propagação direta por todas as camadas."""
        saida = X
        for camada in self.camadas:
            saida = camada.forward(saida)
        return saida

    def backward(self, y_pred, y_true):
        """Retropropagação: calcula gradientes de trás para frente."""
        # Gradiente inicial vem da função de perda
        grad = self.fn_perda.gradiente(y_pred, y_true)

        # Propaga o gradiente pelas camadas em ordem inversa
        for camada in reversed(self.camadas):
            grad = camada.backward(grad)

    def atualizar_pesos(self, taxa_aprendizado):
        """Atualiza parâmetros de todas as camadas."""
        for camada in self.camadas:
            camada.atualizar(taxa_aprendizado)

    def treinar(self, X, y, epocas, taxa_aprendizado,
                tamanho_batch=32, X_val=None, y_val=None, verbose=True):
        
        # Loop de treinamento com mini-batches.

        n = X.shape[0]

        for epoca in range(1, epocas + 1):
            # Embaralha os dados a cada época (evita viés na ordem)
            indices = np.random.permutation(n)
            X_emb, y_emb = X[indices], y[indices]

            perda_epoca = 0.0
            n_batches = 0

            # Divide em mini-batches
            for inicio in range(0, n, tamanho_batch):
                fim = inicio + tamanho_batch
                X_batch = X_emb[inicio:fim]
                y_batch = y_emb[inicio:fim]

                # 1. Forward pass
                y_pred = self.forward(X_batch)

                # 2. Calcula perda do batch
                perda_epoca += self.fn_perda.calcular(y_pred, y_batch)
                n_batches += 1

                # 3. Backward pass (gradientes)
                self.backward(y_pred, y_batch)

                # 4. Atualiza pesos
                self.atualizar_pesos(taxa_aprendizado)

            # Perda média da época
            perda_media = perda_epoca / n_batches
            self.historico["perda_treino"].append(perda_media)

            # Validação (sem atualizar pesos)
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                perda_val = self.fn_perda.calcular(y_val_pred, y_val)
                self.historico["perda_val"].append(perda_val)

            # Exibe progresso
            if verbose and (epoca % max(1, epocas // 10) == 0 or epoca == 1):
                msg = f"Época {epoca:>5}/{epocas} | Perda treino: {perda_media:.6f}"
                if X_val is not None:
                    msg += f" | Perda val: {perda_val:.6f}"
                print(msg)

    def prever(self, X):
        # Retorna as saídas brutas da rede (logits ou probabilidades).
        return self.forward(X)

    def resumo(self):
        # Resumo da arquitetura da rede
        print(f"{'ARQUITETURA DA REDE NEURAL':^55}")
        total_params = 0
        for i, camada in enumerate(self.camadas):
            n_w = camada.W.size
            n_b = camada.b.size
            total = n_w + n_b
            total_params += total
            print(f"  Camada {i+1}: {camada.W.shape[0]:>4} → {camada.W.shape[1]:<4} "
                  f"| Ativação: {camada.ativacao_fn} "
                  f"| Parâmetros: {total}")
        print(f"  {'Total de parâmetros:':45} {total_params}")
        print(f"  {'Função de perda:':45} {type(self.fn_perda).__name__}")

