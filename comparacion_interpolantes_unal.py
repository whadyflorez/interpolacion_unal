import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import barycentric_interpolate

# Función continua pero no diferenciable en x = 0
def f(x):
    return np.abs(x) + 0.3 * np.sin(4 * x)

# Nodos de interpolación
n = 11
x_nodes = np.linspace(-1, 1, n)
y_nodes = f(x_nodes)

# Malla para graficar
x_plot = np.linspace(-1.0, 1.0, 1000)
y_true = f(x_plot)

# 1. Interpolación polinómica (grado n - 1)
poly_coeffs = np.polyfit(x_nodes, y_nodes, deg=n - 1)
y_poly = np.polyval(poly_coeffs, x_plot)

# 2. Interpolación trigonométrica explícita
def trig_basis_matrix(x, n_terms):
    rows = []
    for xi in x:
        row = [1]
        for k in range(1, n_terms // 2 + 1):
            row.append(np.sin(k * xi))
            row.append(np.cos(k * xi))
        rows.append(row)
    return np.array(rows)

A_trig = trig_basis_matrix(x_nodes, n)
c_trig = np.linalg.solve(A_trig, y_nodes)
A_plot = trig_basis_matrix(x_plot, n)
y_trig = A_plot @ c_trig

# 3. Interpolación racional (barycentric)
y_rat = barycentric_interpolate(x_nodes, y_nodes, x_plot)

# Gráfica comparativa
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, label='Función original $f(x) = |x| + 0.3\\sin(4x)$', linewidth=2)
plt.plot(x_plot, y_poly, '--', label='Interpolación polinómica')
plt.plot(x_plot, y_trig, ':', label='Interpolación trigonométrica')
plt.plot(x_plot, y_rat, '-.', label='Interpolación racional')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodos')
plt.title('Interpolación de una función no diferenciable (n = 11)')
plt.xlabel('$x$')
plt.ylabel('$f(x)$ / aproximación')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
