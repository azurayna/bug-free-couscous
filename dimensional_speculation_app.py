
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objs as go
import re

# === SETUP ===
st.set_page_config(page_title="Dimensional Speculation Visualizer", layout="wide")

st.title("🔭 Dimensional Speculation: 2D to 3D Extensions")
st.markdown("Explore how a 2D equation might embed into 3D space.")

# === SYMBOLS ===
x, y, z, t = sp.symbols('x y z t')
e = sp.E
i = sp.I

# === USER INPUT ===
with st.sidebar:
    st.header("🧮 Equation & Restrictions")
    equation_input = st.text_input("2D Equation (e.g., y = x^2 + sqrt(x))", value="y = x^2")
    restriction_input = st.text_area("Optional Restriction (e.g., z < x + y)", value="", height=100)

# === PARSING ===
def preprocess_input(expr):
    expr = expr.replace('^', '**')
    expr = re.sub(r'\be\*\*(\([^)]*\)|[a-zA-Z0-9_]+)', r'exp(\1)', expr)
    return expr

def parse_equation(eq_str):
    processed = preprocess_input(eq_str)
    if "=" not in processed:
        raise ValueError("Equation must contain '='")
    lhs_str, rhs_str = processed.split("=")
    lhs = sp.sympify(lhs_str.strip())
    rhs = sp.sympify(rhs_str.strip())
    return lhs - rhs

# === PROCESSING ===
try:
    expr2d = parse_equation(equation_input)
    sol = sp.solve(expr2d, y)
    if not sol:
        st.error("Could not solve for y in terms of x.")
        st.stop()
    y_expr = sol[0]
    param_eqs = {
        "z = 0": 0,
        "z = t": t,
        "z = y": y_expr.subs(x, t),
        "z = sqrt(x^2 + y^2)": sp.sqrt(t**2 + y_expr.subs(x, t)**2),
        "z = sin(t)": sp.sin(t),
        "z = exp(t)": sp.exp(t)
    }

    # === PLOT SETUP ===
    t_vals = np.linspace(-10, 10, 500)
    fig = go.Figure()
    visible_curves = []
    latex_strings = []

    for label, z_expr in param_eqs.items():
        x_vals = t_vals
        y_vals = sp.lambdify(t, y_expr.subs(x, t), 'numpy')(t_vals)
        z_vals = sp.lambdify(t, z_expr, 'numpy')(t_vals)

        # Apply restriction if given
        keep = np.full_like(t_vals, True, dtype=bool)
        if restriction_input.strip():
            try:
                r_expr = preprocess_input(restriction_input.strip())
                r_sym = sp.sympify(r_expr)
                r_func = sp.lambdify((x, y, z), r_sym, 'numpy')
                keep = r_func(x_vals, y_vals, z_vals)
            except:
                st.warning(f"Could not apply restriction: `{restriction_input}`")

        fig.add_trace(go.Scatter3d(
            x=x_vals[keep], y=y_vals[keep], z=z_vals[keep],
            mode='lines',
            name=label,
            line=dict(width=3)
        ))
        latex_strings.append(f"x(t) = t, \\ y(t) = {sp.latex(y_expr.subs(x, t))}, \\ {label}")

    fig.update_layout(
        title="Possible 3D Dimensionalizations",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        width=1000, height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        st.header("📋 Parametrizations")
        for tex in latex_strings:
            st.latex(tex)

except Exception as e:
    st.error(f"Error: {e}")
