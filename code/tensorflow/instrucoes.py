import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class WeightsMonitor(tf.keras.callbacks.Callback):
    weigts = []
    bias = []
    losses = []

    def on_train_batch_begin(self, batch, logs=None):
        #print(self.model.layers)
        for layer in self.model.layers:
            if len(layer.weights) > 0:
                self.weigts.append(layer.weights[0].numpy().reshape((2, )))
                self.bias.append(layer.weights[1].numpy())

def gen_data(n):
    X = []
    y = []

    ## Gera n ponto equidistantes no intervalo [-2, 2]
    interval = np.linspace(-2, 2, n)

    for i in range(n):
        X.append( (-1, interval[i]) )
        y.append(-1)
        X.append( (1, interval[i]) )
        y.append(1)

    return np.array(X), np.array(y)

def main():
    ## Gera os dados conforme o pdf
    X, y = gen_data(5)
    wm = WeightsMonitor()

    ## Visualiza os dados
    # plt.title("Dados gerados")
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    ## Montamos uma rede com a unica camada.
    #only_layer = tf.keras.layers.Dense(1, activation="tanh")
    only_layer = tf.keras.layers.Dense(1, )

    model = tf.keras.models.Sequential([
        only_layer
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss='mean_squared_error',
        metrics=["mae"]
    )

    ## Treino da rede
    training_history = model.fit(
        X, y,
        epochs=300,
        verbose=1,
        callbacks=[wm]
    )


    wm.weigts = np.array(wm.weigts)
    wm.bias = np.array(wm.bias)
    
    plt.figure(figsize=(16, 9))
    ax = plt.subplot(111, projection="3d")
    plt.scatter(
        wm.weigts[:, 0], 
        wm.weigts[:, 1],
        zs=training_history.history['loss'][1:],
        s=10,
        label="Caminho"
    )
    plt.scatter(1, 0, zs=0, s=10, label="Esperado")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("loss")
    plt.legend()
    plt.show()

    print("Loss conseguido pós treino:", training_history.history['loss'][-1])
    print("Pesos pós treino")
    print(only_layer.weights[0].numpy())
    print("Bias pós treino")
    print(only_layer.weights[1].numpy())
    print("")
    print("Predicts")
    print(model.predict(X))
    print("Real")
    print(y)


    # ## ------- Minimizando -----------
    # ## Aqui vou minimizar na "unha"
    # weights = tf.Variable( np.random.random(size=(2, )), dtype=tf.float32 )
    # bias = tf.Variable( np.random.random(size=1), dtype=tf.float32 )
    
    # ## Definição da função loss
    # def loss_fn():
    #     n = len(X)
    #     s = 0
    #     for i in range(n):
    #         x = X[i]
    #         _y = y[i]
    #         b = weights[0]*x[0] + weights[1]*x[1] + bias
    #         #b = tf.math.tanh(b)
    #         s += tf.math.pow( _y - b , 2 )

    #     return s/(2*n)

    # print()
    # print('Pesos e bias antes da minimização')
    # print(weights.numpy(), bias.numpy())

    # print()
    # print("minimizando..")
    # opt = tf.optimizers.SGD()

    # # equanto o loss for alto
    # i = 0
    # while loss_fn().numpy() > 1e-9:
    #     opt.minimize(loss=loss_fn, var_list=[weights, bias])

    #     i += 1 
    #     if i % 100 == 0:
    #         print("Loss:", loss_fn().numpy())

    # print("Iterações:", opt.iterations.numpy())
    # print()
    # print('Pesos e bias depois da minimização')
    # print(weights.numpy(), bias.numpy())
    # print("Loss: ", loss_fn().numpy())


if __name__ == "__main__":
    main()