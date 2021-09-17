import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def gen_data(n=30):
    x = np.linspace(-3, 3, num=n)
    y = np.zeros(n)

    y[(x >= -1) & (x <= 1)] = 1


    return x, y

def main():
    x, y = gen_data()
    colors = []

    for i in y:
        if i == 1:
            colors.append("r")
        else:
            colors.append("b")


    plt.scatter(x, [3]*len(x), c=colors)
    plt.show()

if __name__ == "__main__":
    main()