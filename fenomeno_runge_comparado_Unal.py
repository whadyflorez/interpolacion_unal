import numpy as np
import matplotlib.pyplot as plt

# Función de Runge
def runge(x):
    return 1 / (1 + 25 * x**2)

# Malla fina para evaluar
x_plot = np.linspace(-1, 1, 1000)
y_true = runge(x_plot)

# Métodos para generar diferentes tipos de nodos
def equidistant_nodes(n):
    return np.linspace(-1, 1, n)

def chebyshev_nodes(n):
    i = np.arange(n)
    return np.cos((2*i + 1) * np.pi / (2*n))

def cluster_ends_nodes(n):
    s = np.linspace(-1, 1, n)
    return np.sign(s) * s**2  # Agrupar artificialmente más cerca de ±1

# Diccionario de funciones generadoras de nodos
node_generators = {
    "Equiespaciados": equidistant_nodes,
    "Chebyshev": chebyshev_nodes,
    "Agrupados en extremos": cluster_ends_nodes
}

# Número de nodos
n = 15

# Crear gráficas por separado para cada distribución
for label, node_func in node_generators.items():
    # Generar nodos y evaluar función
    x_nodes = node_func(n)
    y_nodes = runge(x_nodes)

    # Ajustar polinomio interpolante
    coeffs = np.polyfit(x_nodes, y_nodes, deg=n-1)
    y_interp = np.polyval(coeffs, x_plot)
    error = np.abs(y_true - y_interp)

    # Gráfica: función + interpolante
    plt.figure(figsize=(7, 4))
    plt.plot(x_plot, y_true, label='Función de Runge', linewidth=2, color='orange')
    plt.plot(x_plot, y_interp, '--', label='Interpolante polinómico', color='orangered')
    plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodos')
    plt.title(f'Interpolación con nodos {label} (n={n})')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica: error absoluto
    plt.figure(figsize=(7, 4))
    plt.plot(x_plot, error, color='purple', linewidth=2)
    plt.title(f'Error absoluto $|f(x) - p(x)|$ con nodos {label} (n={n})')
    plt.xlabel('$x$')
    plt.ylabel('Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
