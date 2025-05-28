import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve

# Función original
def f(x, y):
    return np.sin(x) * np.cos(y)

# Puntos de interpolación
np.random.seed(42)
num_points = 200
x_vals = np.random.uniform(0, 2*np.pi, num_points)
y_vals = np.random.uniform(0, 2*np.pi, num_points)
z_vals = f(x_vals, y_vals)

# Parámetro multicuádrico
c = 0.1

# Base RBF multicuádrica
def phi(x, xi):
    r = norm(x - xi)
    return np.sqrt(1 + c**2 * r**2)

# Derivada de la base con respecto a x
def phi_x(x, xi):
    r = norm(x - xi)
    return c**2 * (x[0] - xi[0]) / np.sqrt(1 + c**2 * r**2)

# Matriz del sistema (sin augmentación polinómica)
F = np.zeros((num_points, num_points))
for i in range(num_points):
    xi = np.array([x_vals[i], y_vals[i]])
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        F[i, j] = phi(xi, xj)

# Solución del sistema
a = solve(F, z_vals)

# Interpolante
def interpolant(x):
    return sum(a[j] * phi(x, np.array([x_vals[j], y_vals[j]])) for j in range(num_points))

# Derivada interpolada con respecto a x
def interpolant_x(x):
    return sum(a[j] * phi_x(x, np.array([x_vals[j], y_vals[j]])) for j in range(num_points))

# Malla de evaluación
grid_x, grid_y = np.meshgrid(np.linspace(0, 2*np.pi, 100),
                             np.linspace(0, 2*np.pi, 100))
grid_z_original = f(grid_x, grid_y)
grid_z_interp = np.vectorize(lambda x, y: interpolant(np.array([x, y])))(grid_x, grid_y)
error_abs = np.abs(grid_z_interp - grid_z_original)

# Graficar
fig = plt.figure(figsize=(18, 5))

# Gráfica 1: Función original
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(grid_x, grid_y, grid_z_original, cmap='viridis', alpha=0.8)
ax1.scatter(x_vals, y_vals, z_vals, color='red', s=40)
ax1.set_title('Función original $f(x,y)$')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('f(x,y)')

# Gráfica 2: Interpolación RBF
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(grid_x, grid_y, grid_z_interp, cmap='plasma', alpha=0.8)
ax2.scatter(x_vals, y_vals, z_vals, color='red', s=40)
ax2.set_title('Interpolación RBF (sin términos polinómicos)')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('Interpolado')

# Gráfica 3: Error absoluto
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(grid_x, grid_y, error_abs, cmap='inferno')
ax3.set_title('Error absoluto $|f - \\hat{f}|$')
ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('Error')

plt.tight_layout()
plt.show()

