# Biblioteca que implementa operações de algebra linear
import numpy as np
# Biblioteca utilizada para exibir gráficos
import matplotlib.pyplot as plt
# Importando uma função que ajuda a criar 
# datasets fake, para test
# Informações: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
from sklearn.datasets import make_classification

class NeuralNetwork():
    _biases = []
    _weights = []
    _n_layers = 0

    def __init__(self, input_size, output_size):
        self._biases = []
        self._weights = []
        self._n_layers = 0
        self.input_size = input_size
        self.output_size = output_size

    # Adiciona uma layer com _neurons_ neurônios.
    #
    # Pra isso ele cria uma matriz de pesos aletórios com dimensão 
    # (quantidadde de neurônios da camada anterior, _neurons_)
    # e adiciona no vetor _witghts_.
    def add_layer(self, neurons):
        self._biases.append( np.random.randn(neurons) )

        prev_layer_size = self.input_size
        if self._n_layers > 0:
            # Quantidade de neurônios na camada annterior
            #
            # Número de linhas da camada anterior, é a dimensão
            # da saida.
            prev_layer_size = self._weights[-1].shape[0]

        self._weights.append( np.random.randn(neurons, prev_layer_size) )
        self._n_layers += 1

    # O compile da rede é adicionar a camada de output.
    def compile(self):
        self.add_layer(self.output_size)
        self._n_layers += 1

    # Predict para uma matriz de atributos X.
    # 
    # Para isso, ele chamada o feedfoward para cada X 
    # e retonar como predict o valor de ativação da última camada.
    def predict(self, X):
        predictions = []
        
        for x in X:
            _, activations = self.feedfoward(x)
            predictions.append(activations[-1])

        return predictions

    # Implementação do algoritmo feedfoward
    #
    # Ele armazena as ativações e os z de cada 
    # camada para uso posterior.
    def feedfoward(self, x):
        # Verificacao se a dimensão do x passada
        # é a mesma do input da rede reural.
        assert self.input_size == len(x)
        a = x
        zs = []
        activations = [a]

        for weight, bias in zip(self._weights, self._biases):
            # np.dot calcula o produto de matrizes. 
            # Documentação em: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
            z = np.dot(weight, a.transpose()) + bias
            zs.append(z)
            a = self.activation(z)
            activations.append(a)

        return zs, activations

    # Implementação do algoritmos de backpropagate.
    def backpropagate(self, x, y):
        news_b = [ np.zeros(b.shape) for b in self._biases ]
        news_w = [ np.zeros(w.shape) for w in self._weights ]

        # Primeiro aplico o feedfoward para o x
        zs, activations = self.feedfoward(x)

        # Calculo o custo dessa "operação".
        delta = self.cost_of_derivative(activations[-1], y)

        # Atualiza o erro cometido na última camada.
        news_b[-1] = delta
        news_w[-1] = self.cost_matrix_over_weitght(activations[-2], delta)

        for l in range(2, self._n_layers):
            z = zs[-l]
            sp = self.activation_prime(z)

            # Propaga o erro para a camada anterior.
            delta = np.dot(self._weights[-l+1].transpose(), delta) * sp

            # Atualiza o erro cometido nessa camada.
            news_b[-l] = delta
            news_w[-l] = self.cost_matrix_over_weitght(activations[-l-1], delta)


        # Retorna as matrizes de pesos e bias para os erros
        # comentidos ao tentar preve o x.
        return news_w, news_b

    # A função do mini_batch é aplicar o backprogate 
    # para cada x, e atualizar os pessoas pela medias 
    # de cada matriz retornada.
    def update_mini_batch(self, mini_batch, eta):
        news_b = [np.zeros(b.shape) for b in self._biases]
        news_w = [np.zeros(w.shape) for w in self._weights]
        
        for x, y in mini_batch:
            # Calcula o backpropagate para cada para (x,y)
            delta_w, delta_b = self.backpropagate(x, y)

            # Soma os valores encontrados em cada iteração.
            news_b = [ nb+db for nb, db in zip(news_b, delta_b)]
            news_w = [ nw+dw for nw, dw in zip(news_w, delta_w)]

        m = len(mini_batch)

        # Atualiza os pesos pela média dos resultados do backpropagate
        # de cada elemento em mini_batch.
        self._weights = [ w - (eta/m)*nw for w, nw in zip(self._weights, news_w)]
        self._biases = [ b - (eta/m)*nb for b, nb in zip(self._biases, news_b)]

    # Treino: O treino é feito da seguinta maneira:
    # - Cada _epoch_ eu selecionio _batch_size_ do
    #   conjunto de dados.
    # - Atualizo as matrizes de pesos _weights_
    #   e os vetores de _bias_.
    # - Começo uma nova _epoch_
    def train(self, X, y, epochs=100, eta=0.01, batch_size=30):
        print("Start training..")
        r = np.arange(len(y))
        
        costs = []

        for _ in range(epochs):
            costs.append(self.cost(X, y))
            indexes = np.random.choice(r, batch_size)
            mini_batch = [ (X[i], y[i]) for i in indexes ]
            self.update_mini_batch(mini_batch, eta)

        print("End training")
        return costs

    # Nesse método eu construo a matriz de gradientes da rede
    # com relação aos pesos.
    # Dada pela seguinte equação: http://neuralnetworksanddeeplearning.com/chap2.html#eqtnBP4.
    # A implementação é a definição. Deve ter jeitos mais rápidos de fazer.
    def cost_matrix_over_weitght(self, activation, delta):
        rows = []
        for delta_i in delta:
            rows.append(activation*delta_i)

        return np.array(rows)
        
    # Calcula o custo atual da rede.
    # Como função de custo estou usado a distância quadratica.
    # https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function
    def cost(self, X, y):
        c_cum = 0
        m = len(y)
        y_hat = self.predict(X)

        for i in range(m):
            c_cum = np.linalg.norm(y_hat[i] - y[i]) ** 2

        return c_cum/(2*m)

    # Calcula a derivada da função de custo em um ponto.
    # A função aqui é a distância ao quadrado.
    def cost_of_derivative(self, output, y):
        return output - y 

    # Calcula a função de ativação de neurônio.
    # A função aque é a sigmoid
    # https://en.wikipedia.org/wiki/Sigmoid_function
    def activation(self, z):
        return 1/(1 + np.exp(-z))

    # Calcula a derivada da função de ativação
    def activation_prime(self, z):
        return self.activation(z)*(1-self.activation(z))


if __name__ == "__main__":
    n = NeuralNetwork(4, 1)
    n.add_layer(2)
    n.compile()

    X, y = make_classification(n_features=4, n_samples=1000)

    _ = n.train(X, y, epochs=100, eta=4, batch_size=100)

    y_predic = n.predict(X)
    y_predic = np.array([ round(y_hat[0]) for y_hat in y_predic])
    print("Acerto de {}".format(np.sum(y == y_predic)/len(y)))