import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve

# Definir la función original
def f(x, y):
    return x**2-y**2


# Generar puntos de interpolación
np.random.seed(42)
num_points = 200
x_vals = np.random.uniform(-2, 2, num_points)
y_vals = np.random.uniform(-2, 2, num_points)
x_herm = np.random.uniform(-2, 2, num_points)
y_herm = np.random.uniform(-2, 2, num_points)
z_vals = f(x_vals, y_vals)

# Parámetro de la función RBF (multicuádrica)
c = 0.1

# Función base radial multicuádrica
def phi(x, xi):
    r = norm(x - xi)
    return np.sqrt(1 + c**2 * r**2)

def phi_lap(x, xi):
    r = norm(x - xi)
    df=c**2*(2+c**2*r**2)/phi(x,xi)**3
    return df

def phi_bih(x, xi):
    r = norm(x - xi)
    df=c**4*(c**4*r**4+8*c**2*r**2-8)/\
    (phi(x,xi)*(c**6*r**6+3*c**4*r**4+3*c**2*r**2+1))
    return df

# Base polinómica de segundo orden
def poly(x):
    return np.array([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])

def poly_lap(x):
    return np.array([0, 0, 0, 2, 0, 2])


# Construcción de la matriz del sistema aumentado
F = np.zeros((2*num_points + 6, 2*num_points + 6))
for i in range(num_points):
    xi = np.array([x_vals[i], y_vals[i]])
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        F[i, j] = phi(xi, xj)
    for j in range(num_points):
        xj = np.array([x_herm[j], y_herm[j]])
        F[i, num_points+j] = phi_lap(xi, xj)
    F[i, 2*num_points:2*num_points+6] = poly(xi)
for i in range(num_points):
    xi = np.array([x_herm[i], y_herm[i]])
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        F[num_points+i, j] = phi_lap(xi, xj)
    for j in range(num_points):
        xj = np.array([x_herm[j], y_herm[j]])
        F[num_points+i, num_points+j] = phi_bih(xi, xj)
    F[num_points+i, 2*num_points:2*num_points+6] = poly_lap(xi)
F[2*num_points:2*num_points+6, 0:2*num_points]=F[0:2*num_points,\
                                    2*num_points:2*num_points+6].T

# Vector del lado derecho
RHS = np.zeros(2*num_points + 6)
RHS[:num_points] = z_vals

# Resolver el sistema para obtener los coeficientes
a = solve(F, RHS)

# Función interpolante final
def interpolant(x):
    z = np.zeros(2*num_points + 6)
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        z[j] = phi(x, xj)
    for j in range(num_points):
        xj = np.array([x_herm[j], y_herm[j]])
        z[num_points+j] = phi_lap(x, xj)
    z[2*num_points:] = poly(x)
    return np.dot(a, z)

def interpolant_lap(x):
    z = np.zeros(2*num_points + 6)
    for j in range(num_points):
        xj = np.array([x_vals[j], y_vals[j]])
        z[j] = phi_lap(x, xj)
    for j in range(num_points):
        xj = np.array([x_herm[j], y_herm[j]])
        z[num_points+j] = phi_bih(x, xj)
    z[2*num_points:] = poly_lap(x)
    return np.dot(a, z)

#verificacion
x=np.array([x_vals[0],y_vals[0]])
x1=np.array([x_herm[0],y_herm[0]])

print(interpolant(x))
print(interpolant_lap(x))

# Crear malla para visualización
grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 100),
                             np.linspace(-2, 2, 100))
grid_z_original = f(grid_x, grid_y)
grid_z_interp = np.vectorize(lambda x, y: interpolant(np.array([x, y])))(grid_x, grid_y)
error_abs = np.abs(grid_z_interp - grid_z_original)
grid_z_interp_lap = np.vectorize(lambda x, y: interpolant_lap(np.array([x, y])))(grid_x, grid_y)

# Graficar
fig = plt.figure()

# Gráfica 1: Función original
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(grid_x, grid_y, grid_z_original, cmap='viridis', alpha=0.8)
ax1.scatter(x_vals, y_vals, z_vals, color='red', s=40)
ax1.set_title(r'$f(x,y)=x^2-y^2$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')

# Gráfica 2: Interpolación RBF
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(grid_x, grid_y, grid_z_interp, cmap='plasma', alpha=0.8)
ax2.scatter(x_vals, y_vals, z_vals, color='red', s=40)
ax2.set_title('Interpolación RBF Hermite')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Interpolado')

# Gráfica 3: Error absoluto en la funcion
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(grid_x, grid_y, error_abs, cmap='plasma')
ax3.set_title('Error absoluto $|f - \hat{f}|$')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Error')

# Gráfica 3: laplaciao interpolado
ax3 = fig.add_subplot(2, 2, 4, projection='3d')
ax3.plot_surface(grid_x, grid_y,grid_z_interp_lap , cmap='plasma')
ax3.set_title(r'$\nabla^2 f=0$')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Error')
