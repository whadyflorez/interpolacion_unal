import sympy as sp

# Definir variables simbólicas
r, eps = sp.symbols('r epsilon', positive=True)

# Función base multicuádrica
phi = sp.sqrt(1 + (eps * r)**2)

# Derivadas sucesivas
phi_1 = sp.diff(phi, r)
phi_2 = sp.diff(phi_1, r)
phi_3 = sp.diff(phi_2, r)
phi_4 = sp.diff(phi_3, r)

# Bi-Laplaciano en coordenadas cilíndricas
bi_laplacian = phi_4 + (2 / r) * phi_3 - (1 / r**2) * phi_2 + (1 / r**3) * phi_1

# Generar código LaTeX
latex_expr = sp.latex(sp.simplify(bi_laplacian))

# Mostrar como texto normal
print("Expresión en LaTeX:")
print(latex_expr)
