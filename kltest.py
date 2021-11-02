import numpy as np
import matplotlib.pyplot as plt

q = np.linspace(0.01,0.15,1000)

def klber(p,q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

err = 1e-3
rhs = 0.05477230080755461

plt.plot(q, klber(err,q))
plt.plot(q,q,color='k')
#plt.plot(q, klber(.025,q))
#plt.plot(q, klber(.05,q))
plt.plot([q[0],q[-1]], [rhs, rhs])
plt.show()
