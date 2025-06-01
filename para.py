import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure page
st.set_page_config(page_title="Parametrization Visualizer", layout="centered")

# Input section
st.title(" :, Parametrization Visualizer!")
st.markdown("Enter a Cartesian equation, and this app will:")
st.markdown("1. Parametrize it, 2. Show the parametrization, and 3. Plot the curve/surface.")

eq_input = st.text_input("Enter a Cartesian equation (e.g., `y - x**2` or `z - x**2 - y**2`):", value="y - x**2")
mode = st.radio("Select dimensionality", ["2D", "3D"])

# Define symbols
x, y, z, t = sp.symbols('x y z t')
u, v = sp.symbols('u v')

# Processing
if eq_input:
    try:
        expr = sp.sympify(eq_input)

        if mode == "2D":
            sol = sp.solve(expr, y)
            if sol:
                y_expr = sol[0]
                st.latex(f"x = t,\quad y = {sp.latex(y_expr.subs(x, t))}")
                x_param = sp.lambdify(t, t, 'numpy')
                y_param = sp.lambdify(t, y_expr.subs(x, t), 'numpy')
                
                t_vals = np.linspace(-10, 10, 400)
                x_vals = x_param(t_vals)
                y_vals = y_param(t_vals)

                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals)
                ax.set_title("2D Parametrized Curve")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.grid(True)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.error("Could not solve for y in terms of x.")

        elif mode == "3D":
            sol = sp.solve(expr, z)
            if sol:
                z_expr = sol[0]
                st.latex(f"x = u,\quad y = v,\quad z = {sp.latex(z_expr)}")

                z_func = sp.lambdify((u, v), z_expr, 'numpy')
                u_vals = np.linspace(-5, 5, 100)
                v_vals = np.linspace(-5, 5, 100)
                U, V = np.meshgrid(u_vals, v_vals)
                Z = z_func(U, V)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(U, V, Z, cmap='viridis', edgecolor='none')
                ax.set_title("3D Parametrized Surface")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                st.pyplot(fig)
            else:
                st.error("Could not solve for z in terms of x and y.")

    except Exception as e:
        st.error(f"Error parsing or solving the equation: {e}")
