import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

st.set_page_config(page_title="Parametrized Math Explorer", layout="centered")
st.title(":,Parametrization Explorer!")

mode = st.radio("Choose mode:", ["Curve (y = f(x))", "Surface (z = f(x, y))", "Parametric Curve (x(t), y(t))"])

# Define symbols
x, y, z, t = sp.symbols("x y z t")

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
        else:
            z_expr = sp.sympify(z_input)
            z_func = sp.lambdify((x, y), z_expr, modules="numpy")
            z_vals = z_func(x_vals, y_vals)

        frames = []
        for i in range(len(x_vals)):
            frame = go.Frame(
                data=[
                    go.Scatter3d(x=x_vals[:i+1], y=y_vals[:i+1], z=z_vals[:i+1],
                                 mode="lines", line=dict(color="blue", width=4), showlegend=False),
                    go.Scatter3d(x=[x_vals[i]], y=[y_vals[i]], z=[z_vals[i]],
                                 mode="markers", marker=dict(size=6, color="red"), name="Moving Point")
                ],
                name=str(i)
            )
            frames.append(frame)

        fig = go.Figure(
            data=[
                go.Scatter3d(x=x_vals, y=y_vals, z=z_vals,
                             mode="lines", line=dict(color="blue", width=4), name="Curve"),
                go.Scatter3d(x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
                             mode="markers", marker=dict(size=6, color="red"), name="Moving Point")
            ],
            layout=go.Layout(
                title="3D Parametrized Curve with Moving Particle",
                scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="▶️ Play", method="animate", args=[None])],
                    showactive=False
                )],
                margin=dict(l=0, r=0, b=0, t=40),
                scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            frames=frames
        )
        st.plotly_chart(fig, use_container_width=True)

    elif mode == "Surface (z = f(x, y))":
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

    elif mode == "Parametric Curve (x(t), y(t))":
        
    st.markdown("Define x(t), y(t), and optionally z(t).")

    t_min = st.number_input("t min", value=0.0)
    t_max = st.number_input("t max", value=2 * np.pi)

    x_input = st.text_input("x(t):", value="cos(t)")
    y_input = st.text_input("y(t):", value="sin(t)")
    z_input = st.text_input("z(t):", value="t/2")

    x_input = x_input.replace("^", "**")
    y_input = y_input.replace("^", "**")
    z_input = z_input.replace("^", "**")

    x_expr = sp.sympify(x_input)
    y_expr = sp.sympify(y_input)
    z_expr = sp.sympify(z_input)

    # Velocity components
    dx_dt = sp.diff(x_expr, t)
    dy_dt = sp.diff(y_expr, t)
    dz_dt = sp.diff(z_expr, t)

    x_func = sp.lambdify(t, x_expr, modules="numpy")
    y_func = sp.lambdify(t, y_expr, modules="numpy")
    z_func = sp.lambdify(t, z_expr, modules="numpy")
    dx_func = sp.lambdify(t, dx_dt, modules="numpy")
    dy_func = sp.lambdify(t, dy_dt, modules="numpy")
    dz_func = sp.lambdify(t, dz_dt, modules="numpy")

    t_vals = np.linspace(t_min, t_max, 400)
    x_vals = x_func(t_vals)
    y_vals = y_func(t_vals)
    z_vals = z_func(t_vals)

    # Sample fewer points for vectors
    arrow_t = np.linspace(t_min, t_max, 20)
    arrow_x = x_func(arrow_t)
    arrow_y = y_func(arrow_t)
    arrow_z = z_func(arrow_t)
    arrow_u = dx_func(arrow_t)
    arrow_v = dy_func(arrow_t)
    arrow_w = dz_func(arrow_t)

    # Plot cone velocity vectors
    cones = go.Cone(
        x=arrow_x,
        y=arrow_y,
        z=arrow_z,
        u=arrow_u,
        v=arrow_v,
        w=arrow_w,
        sizemode="scaled",
        sizeref=0.5,
        anchor="tail",
        colorscale="Reds",
        showscale=False,
        name="Velocity Vectors"
    )

    # Animation frames for motion
    frames = []
    for i in range(len(t_vals)):
        frame = go.Frame(
            data=[
                go.Scatter3d(x=x_vals[:i+1], y=y_vals[:i+1], z=z_vals[:i+1],
                             mode="lines", line=dict(color="blue", width=4), showlegend=False),
                go.Scatter3d(x=[x_vals[i]], y=[y_vals[i]], z=[z_vals[i]],
                             mode="markers", marker=dict(size=6, color="red"), name="Moving Point"),
                cones
            ],
            name=str(i)
        )
        frames.append(frame)

    fig = go.Figure(
        data=[
            go.Scatter3d(x=x_vals, y=y_vals, z=z_vals,
                         mode="lines", line=dict(color="blue", width=4), name="Parametric Curve"),
            go.Scatter3d(x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
                         mode="markers", marker=dict(size=6, color="red"), name="Moving Point"),
            cones
        ],
        layout=go.Layout(
            title="3D Parametric Curve with Velocity Vectors",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="▶️ Play", method="animate", args=[None])],
                showactive=False
            )],
            margin=dict(l=0, r=0, b=0, t=40),
            scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        frames=frames
    )

    st.plotly_chart(fig, use_container_width=True)


except Exception as e:
    st.error(f"Error: {e}")
