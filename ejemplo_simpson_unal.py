import numpy as np
import matplotlib.pyplot as plt

# Función a integrar
def f(x):
    return np.exp(-x**2)

# Intervalo de integración
a, b = 0, 2

# Número de subintervalos (debe ser par)
n = 10
if n % 2 != 0:
    n += 1

# Nodos equiespaciados
x = np.linspace(a, b, n + 1)
y = f(x)
h = (b - a) / n

# Regla de Simpson compuesta
S = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]))

# Dominio fino para graficar la función
x_fine = np.linspace(a, b, 500)
y_fine = f(x_fine)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_fine, label=r'$f(x) = e^{-x^2}$', color='blue', linewidth=2)
plt.scatter(x, y, color='black', zorder=5, label='Nodos de interpolación')

# Dibujar parábolas por cada par de subintervalos
for i in range(0, n, 2):
    xi = np.linspace(x[i], x[i+2], 100)
    x0, x1, x2 = x[i], x[i+1], x[i+2]
    y0, y1, y2 = y[i], y[i+1], y[i+2]
    
    # Polinomio de Lagrange grado 2
    def lagrange_segment(xi):
        L0 = ((xi - x1)*(xi - x2)) / ((x0 - x1)*(x0 - x2))
        L1 = ((xi - x0)*(xi - x2)) / ((x1 - x0)*(x1 - x2))
        L2 = ((xi - x0)*(xi - x1)) / ((x2 - x0)*(x2 - x1))
        return y0*L0 + y1*L1 + y2*L2

    yi = lagrange_segment(xi)
    plt.fill_between(xi, yi, alpha=0.3, color='green')

# Mostrar resultado
plt.title(f'Regla de Simpson compuesta (n={n}) — Aproximación ≈ {S:.6f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
