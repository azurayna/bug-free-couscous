import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objs as go
import re

st.set_page_config(page_title="Parametrization Visualizer", layout="centered")

st.title(":, Parametrization Visualizer!")
st.markdown("### Input example: `y = x^2 + sqrt(x) + e^x`")

# UI
equation_input = st.text_input("Equation (e.g., `y = x^2 + sqrt(x) + e^x`):", value="y = x^2")
mode = st.radio("Equation Type", ["2D (y = f(x))", "3D (z = f(x, y))"])

# === Symbols ===
x, y, z, t = sp.symbols('x y z t')
u, v = sp.symbols('u v')
e = sp.E  # Euler's number
i = sp.I  # Imaginary unit, if needed

def preprocess_input(user_input):
    # Replace '^' with '**'
    expr = user_input.replace('^', '**')

    # Replace e^... with exp(...)
    expr = re.sub(r'\be\*\*(\([^)]+\)|[a-zA-Z0-9_]+)', r'exp(\1)', expr)

    return expr

def parse_equation(eq_str):
    """Turn input like y = x^2 into sympy expression y - x**2"""
    processed = preprocess_input(eq_str)
    if "=" not in processed:
        raise ValueError("Equation must contain '='")
    lhs_str, rhs_str = processed.split("=")
    lhs = sp.sympify(lhs_str.strip())
    rhs = sp.sympify(rhs_str.strip())
    return lhs - rhs

# Try to process and plot
try:
    expr = parse_equation(equation_input)

    if mode.startswith("2D"):
        sol = sp.solve(expr, y)
        if not sol:
            st.error("Couldn't solve for y in terms of x.")
        else:
            y_expr = sol[0]
            st.latex(f"x = t,\quad y = {sp.latex(y_expr.subs(x, t))}")

            t_vals = np.linspace(-10, 10, 500)
            x_vals = t_vals
            y_func = sp.lambdify(t, y_expr.subs(x, t), 'numpy')
            y_vals = y_func(t_vals)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Curve'))
            fig.update_layout(title="2D Parametrized Curve", xaxis_title="x", yaxis_title="y")
            st.plotly_chart(fig, use_container_width=True)

    elif mode.startswith("3D"):
        sol = sp.solve(expr, z)
        if not sol:
            st.error("Couldn't solve for z in terms of x and y.")
        else:
            z_expr = sol[0]
            st.latex(f"x = u,\quad y = v,\quad z = {sp.latex(z_expr)}")

            u_vals = np.linspace(-5, 5, 100)
            v_vals = np.linspace(-5, 5, 100)
            U, V = np.meshgrid(u_vals, v_vals)
            z_func = sp.lambdify((u, v), z_expr, 'numpy')
            Z = z_func(U, V)

            fig = go.Figure(data=[
                go.Surface(z=Z, x=U, y=V, colorscale='Viridis')
            ])
            fig.update_layout(
                title="3D Parametrized Surface",
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='y',
                    zaxis_title='z'
                )
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
