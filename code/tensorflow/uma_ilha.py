# Biblioteca para operações com matrizes
import numpy as np
# Biblioteca para trabalhar com redes neurais
import tensorflow as tf
# Bibliteca para exebir gráficos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setando uma seed para que "reproducibilidade" do código
np.random.seed(123)
tf.random.set_seed(1234)


def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gen_data(n=30):
    ## Gera n pontos equidistantes no intervalo [-3, 3]
    x_range = np.linspace(-3, 3, n)
    ## Cria um grid com [-3, 3] x [-3, 3]
    xx, yy = np.meshgrid(x_range, x_range, indexing="ij")

    x = []
    y = []

    c1 = 0
    c2 = 0
    for i in range(len(xx)):
        for j in range(len(yy)):
            dist = np.power(xx[i,j], 2) + np.power(yy[i,j], 2)

            ## Pontos no circulo de raio 2
            if dist <= 2:
                ## Essa verificação se faz necessária para criar uma margem
                ## entre a ilha e os pontos ao seu redor. Neste caso, a distência 
                ## é de 0.7
                if dist <= 1.3:
                    y.append((0, 1))
                    x.append((xx[i, j], yy[i,j]))
                    c1+=1
            else:
                if np.random.uniform(0,1) >= 0.4: continue
                y.append((1,0))
                x.append((xx[i, j], yy[i,j]))
                c2+=1

    return np.array(x), np.array(y), c1, c2, len(x)

def main():
    x, y, n_c1, n_c2, n = gen_data(30)

    ## Criação de uma rede com a camada de input
    ## camada intermediária com 3 neurônios e 
    ## a camada de saida.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(
            units=3, 
            activation="sigmoid",
            bias_initializer="random_uniform",
            name="internal"
        ),
        tf.keras.layers.Dense(
            units=2, 
            activation="sigmoid", 
            bias_initializer="random_uniform",
            name="output"
        )
    ])

    ## Criação do otimizador e compilação da rede.
    optim = tf.keras.optimizers.Adam(learning_rate=0.031)
    model.compile(
        optimizer=optim,
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    ## Treino da rede.
    _ = model.fit(x, y, epochs=200)

    print("")
    print("Quantidade de dados: {}".format(n))
    print("Porcentagem da class 1 (ilha): {:.2f}".format(n_c1/n))
    print("Porcentagem da class 2: {:.2f}".format(n_c2/n))

    # primeira caamada
    l1_w = model.layers[0].weights[0].numpy() ## matriz de pesos
    l1_b = model.layers[0].bias.numpy() ## bias
    z_1 = np.dot(x, l1_w) + l1_b 
    a_1 = sigmoid(z_1)

    # camada de saida
    l2_w = model.layers[1].weights[0].numpy() ## matriz de pesos
    l2_b = model.layers[1].bias.numpy() ## bias
    z_2 = np.dot(a_1, l2_w) + l2_b
    a_2 = sigmoid(z_2)


    ### Plot das imagens.
    classes = np.argmax(y, axis=1)
    colors = []
    for c in classes:
        if c == 1:
            colors.append("r")
        else:
            colors.append("b")

    fig = plt.figure()

    # ax = fig.add_subplot(131)
    # plt.scatter(x[:, 0], x[:, 1], c=colors)    

    ax = fig.add_subplot(121, projection='3d')
    plt.scatter(a_1[:, 0], a_1[:, 1], a_1[:, 2], c=colors)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(122)
    plt.scatter(a_2[:, 0], a_2[:, 1], c=colors)    

    plt.show()



if __name__ == "__main__":
    main()
