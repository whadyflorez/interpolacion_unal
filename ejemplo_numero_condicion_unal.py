import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond, solve

# Función objetivo
def f(x):
    return np.exp(-x**2) * np.cos(3 * x)

# Dominio para graficar
x_plot = np.linspace(-2, 2, 1000)
y_true = f(x_plot)

# Diferentes cantidades de nodos
node_counts = [5, 10, 15, 20]
condition_numbers = []

# Preparar figuras
fig_interp, axes_interp = plt.subplots(2, 2, figsize=(14, 10))
fig_error, axes_error = plt.subplots(2, 2, figsize=(14, 10))
axes_interp = axes_interp.ravel()
axes_error = axes_error.ravel()

for i, n in enumerate(node_counts):
    x_nodes = np.linspace(-2, 2, n)
    y_nodes = f(x_nodes)

    # Construir matriz de Vandermonde y resolver
    V = np.vander(x_nodes, N=n, increasing=True)
    coeffs = solve(V, y_nodes)
    V_plot = np.vander(x_plot, N=n, increasing=True)
    y_interp = V_plot @ coeffs

    # Calcular número de condición y error
    cond_V = cond(V)
    condition_numbers.append(cond_V)
    error = np.abs(y_true - y_interp)

    # Gráfica de interpolación
    ax_i = axes_interp[i]
    ax_i.plot(x_plot, y_true, label=r'$f(x) = e^{-x^2} \cos(3x)$', linewidth=2)
    ax_i.plot(x_plot, y_interp, '--', label='Interpolante', linewidth=2)
    ax_i.scatter(x_nodes, y_nodes, color='black', label='Nodos', zorder=5)
    ax_i.set_title(f'{n} nodos - Cond(V) ≈ {cond_V:.2e}')
    ax_i.set_xlabel('$x$')
    ax_i.set_ylabel('$f(x)$')
    ax_i.grid(True)
    ax_i.legend()

    # Gráfica de error
    ax_e = axes_error[i]
    ax_e.plot(x_plot, error, color='purple', linewidth=2)
    ax_e.set_title(f'Error absoluto |f - p| con {n} nodos')
    ax_e.set_xlabel('$x$')
    ax_e.set_ylabel('Error')
    ax_e.grid(True)

# Títulos generales
fig_interp.suptitle('Interpolación polinómica con matriz de Vandermonde', fontsize=14)
fig_interp.tight_layout(rect=[0, 0, 1, 0.95])
fig_error.suptitle('Errores de interpolación según el número de nodos', fontsize=14)
fig_error.tight_layout(rect=[0, 0, 1, 0.95])

# Gráfico del número de condición
plt.figure(figsize=(8, 5))
plt.semilogy(node_counts, condition_numbers, 'o-', color='red', linewidth=2)
plt.title('Número de condición de la matriz de Vandermonde')
plt.xlabel('Número de nodos')
plt.ylabel('Condición (escala log)')
plt.grid(True)
plt.tight_layout()
plt.show()
