
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

# Parámetros
num_points = 200
c = 1.0

# Función original y derivada exacta respecto a x
def f(x, y):
    return np.sin(x) * np.cos(y)

def df_dx_exact(x, y):
    return np.cos(x) * np.cos(y)

# Puntos de interpolación
np.random.seed(42)
x_vals = np.random.uniform(-2, 2, num_points)
y_vals = np.random.uniform(-2, 2, num_points)
z_vals = f(x_vals, y_vals)

# Funciones base RBF y polinomio
def phi(x, xi):
    r = norm(x - xi)
    return np.sqrt(1 + c**2 * r**2)

def phi_x(x, xi):
    r = norm(x - xi)
    return c**2 * (x[0] - xi[0]) / np.sqrt(1 + c**2 * r**2)

def poly(x):
    return np.array([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])

def poly_x(x):
    return np.array([0, 1, 0, 2*x[0], x[1], 0])

# Construcción de la matriz del sistema
A = np.zeros((num_points + 6, num_points + 6))
b = np.zeros(num_points + 6)

for i in range(num_points):
    xi = np.array([x_vals[i], y_vals[i]])
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        A[i, j] = phi(xi, xj)
    A[i, num_points:] = poly(xi)
    A[num_points:, i] = poly(xi)

b[:num_points] = z_vals
a = solve(A, b)

# Interpolantes
def interpolant(x):
    z = np.zeros(num_points + 6)
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        z[j] = phi(x, xj)
    z[num_points:] = poly(x)
    return np.dot(a, z)

def interpolant_x(x):
    z = np.zeros(num_points + 6)
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        z[j] = phi_x(x, xj)
    z[num_points:] = poly_x(x)
    return np.dot(a, z)

# Evaluación sobre la malla
xg = np.linspace(-2, 2, 50)
yg = np.linspace(-2, 2, 50)
grid_x, grid_y = np.meshgrid(xg, yg)

df_dx_original = df_dx_exact(grid_x, grid_y)
df_dx_interp = np.zeros_like(grid_x)

for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        x_ij = np.array([grid_x[i, j], grid_y[i, j]])
        df_dx_interp[i, j] = interpolant_x(x_ij)

error_dx = np.abs(df_dx_original - df_dx_interp)

# Graficar resultados
fig = plt.figure(figsize=(18, 5))

# Derivada exacta
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(grid_x, grid_y, df_dx_original, cmap='viridis',alpha=0.8)
ax1.scatter(x_vals, y_vals, df_dx_exact(x_vals, y_vals), color='red', s=40)
ax1.set_title('Derivada exacta $\\partial f/\\partial x$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel(r'$\frac{\partial f}{\partial x}$')

# Derivada interpolada
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
interp_vals_x = np.array([interpolant_x(np.array([x, y])) for x, y in zip(x_vals, y_vals)])
ax2.plot_surface(grid_x, grid_y, df_dx_interp, cmap='plasma',alpha=0.8)
ax2.scatter(x_vals, y_vals, interp_vals_x, color='red', s=40)
ax2.set_title('Derivada interpolada RBF')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Interpolado')

# Error absoluto
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(grid_x, grid_y, error_dx, cmap='inferno')
ax3.set_title('Error absoluto $|\\partial f/\\partial x - \\hat{f}_x|$')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Error')

plt.tight_layout()
plt.show()
