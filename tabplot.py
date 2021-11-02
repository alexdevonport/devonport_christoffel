import matplotlib.pyplot as plt
import numpy as np
from datetime import date

t = np.linspace(-1,1,100)
y = np.exp(t)

N = 1000
m = 100
eps = .05
delta=1e-9

paramcell = [
        ['$N$', '{:d}'.format(N)],
        ['$m$', '{:d}'.format(m)],
        ['$\epsilon$', '{:.3g}'.format(eps)],
        ['$\delta$', '{:.3g}'.format(delta)],
        ['samp time', '80.11'],
        ['Nys time', '5.11'],
        ['KL time', '80.11'],
        ['CF eval time', '80.11']
]

plt.plot(t,y)
plt.subplots_adjust(right=0.7)
plt.table(paramcell, loc='right', colWidths=[0.2,0.2], edges='')

plt.savefig('tabplot.png', dpi=300)
