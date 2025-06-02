import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

st.set_page_config(page_title="Parametrized Math Explorer", layout="centered")
st.title(":,Parametrization Explorer!")

mode = st.radio("Choose mode:", ["Curve (y = f(x))", "Surface (z = f(x, y))"])

# Define symbols
x, y, z = sp.symbols("x y z")

# Common input range
x_min = st.number_input("x min", value=-2.0)
x_max = st.number_input("x max", value=2.0)

if x_min >= x_max:
    st.error("x min must be less than x max.")
    st.stop()

try:
    if mode == "Curve (y = f(x))":
        curve_input = st.text_input("y = f(x):", value="x^2")
        z_input = st.text_input("Optional: z = f(x, y):", value="sin(x*y)")

        curve_input = curve_input.replace("^", "**")
        z_input = z_input.replace("^", "**")

        y_expr = sp.sympify(curve_input)
        y_func = sp.lambdify(x, y_expr, modules="numpy")

        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = y_func(x_vals)

        if z_input.strip() == "":
            z_vals = np.zeros_like(x_vals)

                # Generate frames for the animated point
frames = []
for i in range(len(x_vals)):
    frame = go.Frame(
        data=[
            go.Scatter3d(
                x=x_vals[:i+1],
                y=y_vals[:i+1],
                z=z_vals[:i+1],
                mode="lines",
                line=dict(color="blue", width=4),
                showlegend=False
            ),
            go.Scatter3d(
                x=[x_vals[i]],
                y=[y_vals[i]],
                z=[z_vals[i]],
                mode="markers",
                marker=dict(size=6, color="red"),
                name="Moving Point"
            )
        ],
        name=str(i)
    )
    frames.append(frame)

# Initial figure
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=[x_vals[0]],
            y=[y_vals[0]],
            z=[z_vals[0]],
            mode="lines",
            line=dict(color="blue", width=4),
            showlegend=False
        ),
        go.Scatter3d(
            x=[x_vals[0]],
            y=[y_vals[0]],
            z=[z_vals[0]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Moving Point"
        )
    ],
    layout=go.Layout(
        title="3D Parametrized Curve with Moving Particle",
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])],
            showactive=False
        )],
        margin=dict(l=0, r=0, b=0, t=40),
        scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
    ),
    frames=frames
)

st.plotly_chart(fig, use_container_width=True)
        else:
            z_expr = sp.sympify(z_input)
            z_func = sp.lambdify((x, y), z_expr, modules="numpy")
            z_vals = z_func(x_vals, y_vals)


        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines', line=dict(width=4),
            name="Curve"
        ))
        fig.update_layout(
            title="3D Parametrized Curve",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Surface mode
        y_min = st.number_input("y min", value=-2.0)
        y_max = st.number_input("y max", value=2.0)

        if y_min >= y_max:
            st.error("y min must be less than y max.")
            st.stop()

        surface_input = st.text_input("z = f(x, y):", value="sin(x^2 + y^2)")
        surface_input = surface_input.replace("^", "**")

        z_expr = sp.sympify(surface_input)
        z_func = sp.lambdify((x, y), z_expr, modules="numpy")

        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = z_func(X, Y)

        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
        fig.update_layout(
            title="3D Surface",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")

