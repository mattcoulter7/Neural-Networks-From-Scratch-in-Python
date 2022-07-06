import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x

x = np.array(range(5))
y = f(x)

plt.plot(x,y)
plt.show()