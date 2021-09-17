import numpy as np
import tensorflow as tf
from NNGraph import NNGraph
from DataGen import DataGen

dataGenerators = DataGen()
X, y = dataGenerators.uma_ilha(30)
paths_accum = {}

for i in range(10):
    print("Treinando rede {}..".format(i))
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

    optim = tf.keras.optimizers.Adam(learning_rate=0.031)
    model.compile(
        optimizer=optim,
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    hist = model.fit(
        X,
        y,
        epochs=200,
        #batch_size=3,
        verbose=0
    )

    g = NNGraph(model.layers)
    print("Accuracy: {}".format(hist.history["accuracy"][-1]))
    print()
    g.build_nn_graph()
    paths_data = g.compute_paths()
    for path_data in paths_data:
        path_edges = ",".join(path_data[0])
        path_weight = path_data[1]

        path_weights = paths_accum.get(path_edges, [])
        path_weights.append(path_weight)

        paths_accum[path_edges] = path_weights

total = 0
especific = 0
for path, weights in paths_accum.items():
    mean = np.mean(weights)

    total += 1
    if abs(mean) < 1:
        especific += 1
    print("Descrição caminho:", path, mean, np.std(weights))

print("não preciso percorrer {} da rede".format(especific/total))