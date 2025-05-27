import numpy as np
import matplotlib.pyplot as plt

# 1. Definir la curva cerrada: ecuación paramétrica del corazón
t_vals = np.linspace(0, 2 * np.pi, 500)
x_curve = 16 * np.sin(t_vals)**3
y_curve = 13 * np.cos(t_vals) - 5 * np.cos(2*t_vals) - 2 * np.cos(3*t_vals) - np.cos(4*t_vals)

# 2. Elegir nodos para los elementos cuadráticos
num_elements = 8  # número de elementos isoparamétricos
node_indices = np.linspace(0, len(t_vals) - 1, 2 * num_elements + 1, dtype=int)
x_nodes = x_curve[node_indices]
y_nodes = y_curve[node_indices]

# 3. Definir funciones de forma cuadráticas
def N1(xi): return 0.5 * xi * (xi - 1)
def N2(xi): return 1 - xi**2
def N3(xi): return 0.5 * xi * (xi + 1)

# 4. Interpolación isoparamétrica para cada elemento
xi_vals = np.linspace(-1, 1, 100)
x_all = []
y_all = []

for i in range(0, len(x_nodes) - 2, 2):
    x0, x1, x2 = x_nodes[i], x_nodes[i+1], x_nodes[i+2]
    y0, y1, y2 = y_nodes[i], y_nodes[i+1], y_nodes[i+2]

    def x_interp(xi): return N1(xi)*x0 + N2(xi)*x1 + N3(xi)*x2
    def y_interp(xi): return N1(xi)*y0 + N2(xi)*y1 + N3(xi)*y2

    x_elem = x_interp(xi_vals)
    y_elem = y_interp(xi_vals)
    x_all.append(x_elem)
    y_all.append(y_elem)

# 5. Graficar la curva y su aproximación por elementos
plt.figure(figsize=(8, 8))
plt.plot(x_curve, y_curve, color='gray', linewidth=2, label='Curva de referencia (corazón)')
for i in range(len(x_all)):
    plt.plot(x_all[i], y_all[i], linewidth=2, label=f'Elemento {i+1}')
    plt.fill_between(x_all[i], y_all[i], alpha=0.2)

plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodos de interpolación')
plt.title("Aproximación de una curva cerrada con elementos isoparamétricos")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
