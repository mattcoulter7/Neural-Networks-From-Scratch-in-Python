import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

x = np.arange(0,5,0.01)
y = f(x)
plt.plot(x,y)

def approximate_tangent_line(x,approximate_derivative):
    return (approximate_derivative * x) + b

def gradient(x1,x2,y1,y2):
    return (y2-y1) / (x2-x1)

def offset(x,y,approximate_derivative):
    return y-approximate_derivative * x

colors = ['k','g','r','b','c']

tanget_line_length = 2
p2_delta = 0.0001
for i in range(5):
    x1 = i
    x2 = x1 + p2_delta
    
    y1 = f(x1)
    y2 = f(x2)

    approximate_derivative = gradient(x1,x2,y1,y2)
    b = offset(x1,y1,approximate_derivative)

    to_plot = [x1 - tanget_line_length/2,x1,x1 + tanget_line_length/2]

    plt.scatter(x1,y1,c=colors[i % len(colors)])
    plt.plot(
        [point for point in to_plot],
        [approximate_tangent_line(point,approximate_derivative) for point in to_plot]
    )


plt.show()