import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# gera os dados como uma função (um linha)
def gen_data(n):
    X = []
    y = []

    ## Gera n ponto equidistantes no intervalo [-2, 2]

    for i in np.linspace(-2, 2, n):
        ## A
        X.append( (-1, i) )
        y.append([1, 0])

        ## B
        X.append( (1, i) )
        y.append([0, 1])

    return np.array(X), np.array(y)

def main():
    X, y = gen_data(30)

    w00s = []
    w01s = []
    w10s = []
    w11s = []
    b0s = []
    b1s = []

    for i in range(1):
        print("Rede {}".format(i+1))
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                2, 
                input_shape=(2, ),
                activation="sigmoid"
            )
        ])

        model.compile(
            loss='mean_squared_error',
            metrics=["accuracy"]
        )

        hist = model.fit(X, y, epochs=100, batch_size=5, verbose=0)

        acc = hist.history["accuracy"][-1]
        print("Accuracy:", acc)
        if acc < 1: 
            print("não convergiu")
            continue

        W = model.layers[0].weights[0].numpy()
        b = model.layers[0].weights[1].numpy()

        # print(X.shape, W.shape)

        # print(X[0], W)

        # z_1 = np.dot(X, W) + b 
        # a_1 = tf.keras.activations.sigmoid(z_1)

        # print((model.predict(X) - a_1) <= 1e-6)

        w00 = W[0][0] # entrada 0 saida 0
        w01 = W[0][1] # entrada 0 saida 1
        w10 = W[1][0] # entrada 1 saida 0
        w11 = W[1][1] #  entrada 1 saida 1

        b0 = b[0]
        b1 = b[1]

        print("A_0({}x + {}y + {})".format(w00, w10, b0))
        print("A_1({}x + {}y + {})".format(w01, w11, b1))

        w00s.append(w00)
        w01s.append(w01)
        w10s.append(w10)
        w11s.append(w11)

        b0s.append(b0)
        b1s.append(b1)
        print("")

    plt.subplot(321)
    plt.hist(w00s)
    plt.xlabel("entrada 0 saida 0")

    plt.subplot(322)
    plt.hist(w01s)
    plt.xlabel("entrada 0 saida 1")

    plt.subplot(323)
    plt.hist(w10s)
    plt.xlabel("entrada 1 saida 0")

    plt.subplot(324)
    plt.hist(w11s)
    plt.xlabel("entrada 1 saida 1")

    plt.subplot(325)
    plt.hist(b0s)
    plt.xlabel("b0")

    plt.subplot(326)
    plt.hist(b1s)
    plt.xlabel("b1")

    plt.show()


if __name__ == "__main__":
    main()