import numpy as np
import matplotlib.pyplot as plt

n=np.arange(1,11)
eps=0.05
plt.plot(n, 1-(1-eps)**(1/n))
plt.show()
