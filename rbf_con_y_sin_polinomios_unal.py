import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

# Función de Runge
def runge(x):
    return 1 / (1 + 25 * x**2)

# Función base multicuádrica
def phi(r):
    c = 2.0
    return np.sqrt(1 + (c * r)**2)

# Puntos de interpolación
N = 11
x_nodes = np.linspace(-1, 1, N)
y_nodes = runge(x_nodes)

# Matriz de distancias
def distance_matrix(x1, x2):
    return np.abs(x1[:, None] - x2[None, :])

# Interpolación RBF sin polinomios
r = distance_matrix(x_nodes, x_nodes)
A = phi(r)
alpha_rbf = solve(A, y_nodes)

# Interpolación RBF con términos polinómicos (lineales: 1 y x)
P = np.vstack((np.ones_like(x_nodes), x_nodes)).T
A_aug = np.block([
    [A, P],
    [P.T, np.zeros((2, 2))]
])
rhs_aug = np.concatenate([y_nodes, np.zeros(2)])
alpha_aug = solve(A_aug, rhs_aug)
alpha_rbf_poly = alpha_aug[:N]
lambda_poly = alpha_aug[N:]

# Evaluación de los interpolantes
x_eval = np.linspace(-1.2, 1.2, 400)
r_eval = distance_matrix(x_eval, x_nodes)
phi_eval = phi(r_eval)

# RBF sin polinomios
y_interp_rbf = phi_eval @ alpha_rbf

# RBF con polinomios
P_eval = np.vstack((np.ones_like(x_eval), x_eval)).T
y_interp_poly = phi_eval @ alpha_rbf_poly + P_eval @ lambda_poly

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x_eval, runge(x_eval), 'k-', label='Función original (Runge)')
plt.plot(x_eval, y_interp_rbf, 'r--', label='RBF sin polinomios')
plt.plot(x_eval, y_interp_poly, 'b-.', label='RBF con polinomios')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodos')
plt.title('Comparación: RBF sin y con términos polinómicos')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
