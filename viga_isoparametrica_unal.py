import numpy as np
import matplotlib.pyplot as plt

# Definir la geometría objetivo: una curva suave
curve_x = np.linspace(0, 3, 300)
curve_y = -0.5 * np.sin(curve_x)  # curva original

# Nodos físicos para 3 elementos cuadráticos (7 nodos)
x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
y_nodes = -0.5 * np.sin(x_nodes)

# Funciones de forma cuadráticas (en coordenadas naturales)
def N1(xi): return 0.5 * xi * (xi - 1)
def N2(xi): return 1 - xi**2
def N3(xi): return 0.5 * xi * (xi + 1)

# Interpolación para cada elemento
xi_vals = np.linspace(-1, 1, 100)
x_all = []
y_all = []

for i in range(0, len(x_nodes) - 2, 2):  # 3 elementos: nodos 0-2, 2-4, 4-6
    x0, x1, x2 = x_nodes[i:i+3]
    y0, y1, y2 = y_nodes[i:i+3]

    def x_interp(xi):
        return N1(xi)*x0 + N2(xi)*x1 + N3(xi)*x2

    def y_interp(xi):
        return N1(xi)*y0 + N2(xi)*y1 + N3(xi)*y2

    x_elem = x_interp(xi_vals)
    y_elem = y_interp(xi_vals)
    x_all.append(x_elem)
    y_all.append(y_elem)

# Gráfica
plt.figure(figsize=(10, 5))
plt.plot(curve_x, curve_y, label='Geometría curva original', color='gray', linewidth=2)

for i in range(len(x_all)):
    plt.plot(x_all[i], y_all[i], label=f'Elemento {i+1}', linewidth=2)
    plt.fill_between(x_all[i], y_all[i], alpha=0.2)

plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodos de interpolación')
plt.title("Aproximación de una geometría curva con elementos isoparamétricos")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

