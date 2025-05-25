import numpy as np
import matplotlib.pyplot as plt

# 1. Datos: 4 puntos igualmente espaciados en [-1, 1]
x = np.array([-1.0, -0.3333, 0.3333, 1.0])
f = lambda x: np.exp(x)
y = f(x)

# 2. Construcción manual de la matriz de Vandermonde (grado 3)
n = len(x)
V = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        V[i, j] = x[i]**j  # cada fila: [1, x_i, x_i^2, x_i^3]

# 3. Resolución del sistema V a = y
a = np.linalg.solve(V, y)

# 4. Evaluación del polinomio interpolante
x_plot = np.linspace(-1.2, 1.2, 200)
y_interp = sum(a[j] * x_plot**j for j in range(n))
y_true = f(x_plot)

# 5. Gráfica
plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_true, label='Función $f(x) = e^x$', linewidth=2)
plt.plot(x_plot, y_interp, '--', label='Interpolación polinómica (grado 3)')
plt.scatter(x, y, color='black', zorder=5, label='Nodos de interpolación')
plt.title('Interpolación polinómica de grado 3')
plt.xlabel('$x$')
plt.ylabel('$f(x)$ / Aproximación')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("grafica_interpolacion.png", dpi=300)
plt.show()
