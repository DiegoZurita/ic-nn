import numpy as np
import tensorflow as tf


def gen_data_2_classe_na_reta(n = 100):
    eps = 0.8
    inicio = 0

    i1 = np.linspace(inicio, inicio+ 2, n)
    i2 = np.linspace(inicio + 2 + eps, inicio + 4, n)

    d1 = [ (x, 1) for x in i1 ]
    d2 = [ (x, 0) for x in i2 ]

    data = np.concatenate((d1, d2))

    return data[:, 0], data[:, 1]

def fit_model(x, y, neurons):
    print("##################################")
    print("Treinando para {} neuronions".format(neurons))
    print("##################################")


    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            input_shape=(1, ),
            units=neurons, 
            name="internal",
        ),

        tf.keras.layers.Dense(
            units=1, 
            name="output"
        )
    ])

    optim = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=optim,
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    print(model.summary())

    hist = model.fit(
        x, 
        y,
        batch_size=16,
        epochs=500, 
        verbose=0
    )

    acc = hist.history["accuracy"]
    print("##################################")
    print("Acurácia final: {}".format(acc[-1]))
    print("##################################")

    return model


def main():
    x, y = gen_data_2_classe_na_reta()
    neurons = 7
    k = 2
    model = fit_model(x, y, neurons)
    layers = model.layers

    s = 0
    index_especifico = np.random.randint(0, len(x))
    x_especifico = x[index_especifico]
    
    w2 = layers[0].weights[0].numpy()
    b2 = layers[0].weights[1].numpy()

    w3 = layers[1].weights[0].numpy()
    b3 = layers[1].weights[1].numpy()

    sum_on_neurons = 0
    print()
    print("X escolhido: {}, previsto: {}, real: {}".format(
        x_especifico, 
        model.predict([x_especifico]),
        y[index_especifico]
    ))
    print("w2")
    print(w2)
    print("b2")
    print(b2)

    print("w3")
    print(w3)
    print("b3")
    print(b3)

    for i in range(neurons):
        sum_on_neurons += w3[i, 0] * (x_especifico*w2[0, i] + b2[i])

    sum_on_k = 0
    for i in range(k):
        sum_on_k += w3[i, 0] * (x_especifico*w2[0, i] + b2[i])
    print()
    print("hipotese: ")
    print(sum_on_neurons, sum_on_k, np.abs(sum_on_neurons - sum_on_k))



if __name__ == "__main__":
    main()