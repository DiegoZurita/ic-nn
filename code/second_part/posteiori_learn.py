import numpy as np
from typing import Mapping, List
import matplotlib.pyplot as plt

N = 10000
posteriori: Mapping[int, List[int]] = {}

for i in range(N):
    priori = np.random.poisson(1)
    likelihood = np.random.exponential(priori)
    i_posteriori = priori*likelihood

    post_until_now =  posteriori.get(priori, [])
    post_until_now.append(i_posteriori)

    posteriori[priori] = post_until_now

plt.subplot(121)

plt.title("Priori")
plt.hist(np.random.poisson(3, 1000))

plt.subplot(122)

plt.title("Posteiori")
for i, estimations in posteriori.items():
    # for estimation in estimations:
    #     plt.scatter(i, estimation, c="b")

    plt.scatter(i, np.mean(estimations), c="r")

plt.show()