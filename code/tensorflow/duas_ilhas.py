import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(123)
tf.random.set_seed(1234)

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gen_data(n=30):
    x_range = np.linspace(-5, 5, n)
    xx, yy = np.meshgrid(x_range, x_range, indexing="ij")

    x = []
    y = []

    n_c1 = 0
    n_c2 = 0
    for i in range(len(xx)):
        for j in range(len(yy)):
            dist_circ1 = np.power(xx[i,j]-2, 2) + np.power(yy[i,j], 2)
            dist_circ2 = np.power(xx[i,j]+2, 2) + np.power(yy[i,j], 2)

            if dist_circ1 <= 2 or dist_circ2 <= 2:
                if dist_circ1 <= 1.3 or dist_circ2 <= 1.3:
                    y.append((0, 1))
                    x.append((xx[i, j], yy[i,j]))
                    n_c1+=1
            else:
                if np.random.uniform(0,1) >= 0.4: continue
                y.append((1,0))
                x.append((xx[i, j], yy[i,j]))
                n_c2+=1

    

    return np.array(x), np.array(y), n_c1, n_c2, len(x)


def main():
    x, y, n_c1, n_c2, n = gen_data(30)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        #tf.keras.layers.Dense(4, activation="relu"),
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

    optim = tf.keras.optimizers.Adam(learning_rate=0.031)
    model.compile(
        optimizer=optim,
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    # model.summary()
    _ = model.fit(x, y, epochs=100)

    print("")
    print("n: {}".format(n))
    print("c1: {:.2f}".format(n_c1/n))
    print("c2: {:.2f}".format(n_c2/n))

    # primeira caamada
    l1_w = model.layers[0].weights[0].numpy()
    l1_b = model.layers[0].bias.numpy()
    z_1 = np.dot(x, l1_w) + l1_b 
    a_1 = sigmoid(z_1)

    # camada de saida
    l2_w = model.layers[-1].weights[0].numpy()
    l2_b = model.layers[-1].bias.numpy()
    z_2 = np.dot(a_1, l2_w) + l2_b
    a_2 = sigmoid(z_2)

    classes = np.argmax(y, axis=1)
    colors = []
    for c in classes:
        if c == 1:
            colors.append("r")
        else:
            colors.append("b")

    fig = plt.figure()

    ax = fig.add_subplot(131)
    plt.scatter(x[:, 0], x[:, 1], c=colors)    

    ax = fig.add_subplot(132, projection='3d')
    plt.scatter(a_1[:, 0], a_1[:, 1], a_1[:, 2], c=colors)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(133)
    plt.scatter(a_2[:, 0], a_2[:, 1], c=colors)    

    plt.show()



if __name__ == "__main__":
    main()