## 
## Expero que até um certo n, a prob seja 0
## depois, fique uniforme.
##

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from DataGen import DataGen
import tensorflow as tf

def likelihood_neural(X_train, X_test, y_train, y_test, neurons):
    print("##################################")
    print("Treinando para {} neuronions".format(neurons))
    print("##################################")


    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(
            units=neurons, 
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

    ## Treino da rede.
    _ = model.fit(X_train, y_train, epochs=100, verbose=0)

    evaluate_result = model.evaluate(X_test, y_test)
    print("Likelihood founded: ", evaluate_result[1])
    print()
    return evaluate_result[1] ## representa a acuracia



def main():
    N = 25
    n_data_points = 50
    parametro_uniforme_discreta = 7

    dataGen = DataGen() 
    X, y = dataGen.uma_ilha(n_data_points)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    posteriories = {}
    priories = []
    

    ## Calcula posteriories
    for i in range(N):
        priori = np.random.choice(np.arange(parametro_uniforme_discreta) + 1)
        likelihood = likelihood_neural(X_train, X_test, y_train, y_test, priori)
        posteriori = likelihood # * priori = roubalheira

        posteiori_until_now = posteriories.get(priori, [])
        posteiori_until_now.append( posteriori )

        posteriories[priori] = posteiori_until_now
        priories.append(priori)


    ## Exibe priori e posterior
    plt.subplot(121)

    plt.title("Priori para os neuronios: UnifDiscreta(1, {})".format(parametro_uniforme_discreta))
    plt.hist(priories)

    plt.subplot(122)
    plt.title("\"Posteriorie\" para os neuronios")
    for i, estimations in posteriories.items():
        plt.scatter(i, np.mean(estimations), c="r")

    plt.show()



if __name__ == "__main__":
    main()