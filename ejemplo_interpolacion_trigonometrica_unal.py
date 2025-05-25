import numpy as np
import matplotlib.pyplot as plt

# Número de nodos de interpolación y funciones base
N = 10
x_nodes = np.linspace(0, 2 * np.pi, N, endpoint=False)

# Función periódica a interpolar
def f(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

y_nodes = f(x_nodes)

# Definir las 10 funciones base: 1, cos(kx), sin(kx) para k = 1, 2, 3, 4, más cos(5x)
def phi_0(x): return np.ones_like(x)
def phi_1(x): return np.cos(x)
def phi_2(x): return np.sin(x)
def phi_3(x): return np.cos(2 * x)
def phi_4(x): return np.sin(2 * x)
def phi_5(x): return np.cos(3 * x)
def phi_6(x): return np.sin(3 * x)
def phi_7(x): return np.cos(4 * x)
def phi_8(x): return np.sin(4 * x)
def phi_9(x): return np.cos(5 * x)

# Lista de funciones base
basis_funcs = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8, phi_9]

# Construir la matriz del sistema A[i,j] = phi_j(x_i)
A = np.column_stack([phi(x_nodes) for phi in basis_funcs])

# Resolver el sistema A @ a = y_nodes para obtener los coeficientes
coeffs = np.linalg.solve(A, y_nodes)

# Construir el interpolante como combinación lineal
def interpolante(x):
    return sum(coeffs[j] * basis_funcs[j](x) for j in range(N))

# Malla para graficar
x_plot = np.linspace(0, 2 * np.pi, 400)
y_true = f(x_plot)
y_interp = interpolante(x_plot)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_true, label='Función original $f(x)$', linewidth=2)
plt.plot(x_plot, y_interp, '--', label='Interpolante trigonométrico (10 funciones base)')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodos de interpolación')
plt.title('Interpolación trigonométrica con 10 nodos y 10 funciones base')
plt.xlabel('$x$')
plt.ylabel('$f(x)$ / aproximación')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
