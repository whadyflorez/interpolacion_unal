
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
N = 10
x_nodes = np.linspace(0, 2 * np.pi, N, endpoint=False)

def f(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

y_nodes = f(x_nodes)

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

basis_funcs = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8, phi_9]

A_trig = np.column_stack([phi(x_nodes) for phi in basis_funcs])
coeffs_trig = np.linalg.solve(A_trig, y_nodes)

def interpolante_trig(x):
    return sum(coeffs_trig[j] * basis_funcs[j](x) for j in range(N))

poly_coeffs = np.polyfit(x_nodes, y_nodes, deg=N-1)
def interpolante_poly(x):
    return np.polyval(poly_coeffs, x)

x_eval = np.linspace(0.1, 2 * np.pi - 0.1, 10)
f_exact = f(x_eval)
f_trig = interpolante_trig(x_eval)
f_poly = interpolante_poly(x_eval)
error_trig = np.abs(f_exact - f_trig)
error_poly = np.abs(f_exact - f_poly)

x_plot = np.linspace(0, 2 * np.pi, 400)
plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), label='Función original $f(x)$', linewidth=2)
plt.plot(x_plot, interpolante_trig(x_plot), '--', label='Interpolación trigonométrica')
plt.plot(x_plot, interpolante_poly(x_plot), ':', label='Interpolación polinómica')
plt.scatter(x_nodes, y_nodes, color='black', label='Nodos de interpolación', zorder=5)
plt.scatter(x_eval, f_exact, color='red', label='Puntos de evaluación', zorder=5)
plt.xlabel('$x$')
plt.ylabel('$f(x)$ / Aproximación')
plt.title('Comparación: Interpolación trigonométrica vs polinómica (N = 10)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
