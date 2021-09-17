import numpy as np
import tensorflow as tf

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

    # Rede 1
    # Accuracy: 1.0
    # A_0(-1.8427765369415283x + -0.03800506889820099y + 0.0005597827257588506)
    # A_1(0.02591600827872753x + 0.7730266451835632y + 0.0021445956081151962)

    z1 = -1.8427765369415283*X[:, 0] -0.03800506889820099*X[:, 1] + 0.0005597827257588506
    z2 = 0.02591600827872753*X[:, 0] +0.7730266451835632*X[:, 1] + 0.0021445956081151962

    print(X[:, 0])
    print("Antes de ativar")
    print(z1)
    print(z2)
    print("Depois de ativar")
    print(tf.keras.activations.sigmoid(z1))
    print(tf.keras.activations.sigmoid(z2))


if __name__ == "__main__":
    main()