import numpy as np
import matplotlib.pyplot as plt

# Definir la función de Runge
def runge(x):
    return 1 / (1 + 25 * x**2)

# Dominio para graficar la función original y los interpolantes
x_plot = np.linspace(-1, 1, 1000)
y_exact = runge(x_plot)

# Cantidades de nodos a comparar
node_counts = [5, 11, 17]

# Crear la figura con 3 subgráficas
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, n in zip(axes, node_counts):
    # Nodos equiespaciados en [-1, 1]
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge(x_nodes)

    # Interpolación polinómica con np.polyfit y evaluación
    coeffs = np.polyfit(x_nodes, y_nodes, deg=n - 1)
    y_interp = np.polyval(coeffs, x_plot)

    # Graficar
    ax.plot(x_plot, y_exact, label='Función de Runge', linewidth=2)
    ax.plot(x_plot, y_interp, '--', label=f'Interpolante polinómico (n={n})')
    ax.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodos')
    ax.set_title(f'{n} nodos equiespaciados')
    ax.set_xlabel('$x$')
    ax.grid(True)

axes[0].set_ylabel('$f(x)$')
axes[1].legend()
plt.suptitle('Fenómeno de Runge en interpolación polinómica')
plt.tight_layout()
plt.show()
